#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-test: bench your edited tree, and optionally bench a pristine copy to get a real ratio.

NOT the sealed verifier — this is the cheap measurement you use to decide whether the edits in
`repo/` recovered the regression.

WHY IT CAN BENCH BOTH ARMS. The scored baseline is re-benched from the pinned tree at scoring time,
so there is no number committed here to compare against — a stale one would be worse than none.
`--with-baseline` reconstructs the pristine arm from the `inference_base` tag in `repo/`'s own git
history and benches it the same way, which is the only way to get a ratio that means anything
before you submit.

HOW IT MEASURES. A fresh `vllm serve` per SCORED shape, arm and round, booted with the deployment's
flags on a clean compile cache, and driven with `vllm bench serve`: the anchor shape first, on its
own server, then the guard shapes against that same server; then each other scored shape on a server
of its own. The reading per shape is total tokens/s; the score is the geometric mean over the scored
shapes of candidate over baseline, and a guard shape must not read slower on the candidate.

TWO CAUTIONS:

  * A single unrotated sample lies. Benching A then B once can put the difference in the WRONG
    direction on this workload. `--rounds N` alternates the arms; use it.
  * The first server in a fresh process pays one-time initialisation. A warm-up (one server per
    arm, the anchor shape) is discarded by default (`--no-warmup` if you know why you want to skip it).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Not the port the scored run uses, so a self-test cannot collide with a server it did not start.
DEFAULT_PORT = 8811
BOOT_TIMEOUT_SEC = int(os.environ.get("INFB_BOOT_TIMEOUT_SEC", "1200"))


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path(__file__).resolve().parents[1]


def _engine_python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def _matrix(ws: Path) -> dict:
    """Every shape of public/workloads.json: the anchor, the other scored shapes, the guard shapes."""
    doc = json.loads((ws / "public" / "workloads.json").read_text())
    points = {w["name"]: w for w in doc.get("workloads", []) if type(w) is dict and "name" in w}
    anchor = doc.get("primary_workload") or next(iter(points))
    scored = [str(n) for n in doc.get("scored_workloads") or [anchor]]
    guards = [str(n) for n in doc.get("guard_workloads") or []]
    benched = [anchor] + [n for n in scored if n != anchor] + [n for n in guards
                                                              if type(points[n].get("from")) is not str]
    return {"anchor": anchor, "scored": scored, "guards": guards, "points": points, "benched": benched}


def _workload(ws: Path) -> dict:
    m = _matrix(ws)
    return m["points"][m["anchor"]]


def _geomean(values):
    values = [v for v in values if type(v) in (int, float) and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link.resolve())
    raise SystemExit("no model: set INFB_MODEL_PATH or stage <workspace>/model")


def _seed(requested: int | None) -> int:
    """The prompt seed, shared by both arms and every shape of one run: `--seed`, else
    INFB_BENCH_SEED, else time-based, so two runs draw different request streams unless pinned."""
    if requested is not None:
        return int(requested)
    env = os.environ.get("INFB_BENCH_SEED")
    return int(env) if env and env.strip() else (int(time.time()) % (2 ** 31))


# THIS SCRIPT AND THE VERIFIER RUN THE SAME SERVER AND THE SAME CLIENT. Both boot `vllm serve` with
# the flags below and drive every shape of workloads.json with `vllm bench serve`. That is
# deliberate -- a self-test that quietly measures something else is worse than no self-test.
#
# What differs is the tree, not the measurement: the verifier stages its own pristine copy of the
# pinned source, so its absolute numbers will not match yours. Compare ratios, not throughputs.


def _serve_cmd(model: str, max_len: int, port: int) -> list[str]:
    # SAME FLAGS THE VERIFIER USES. The ones that are not defaults and are not preferences:
    #
    #   --speculative-config   the deployment under test runs with the speculator ON. Both arms do;
    #                          switching it off measures a different deployment, not a fix.
    #   -O.cudagraph_mode      PIECEWISE. The FULL decode-graph path cannot start at this pin -- its
    #                          capture sizes have to be multiples of num_spec+1 = 3 and the resolved
    #                          defaults are not -- and forcing it to work by feeding a multiples-of-3
    #                          ladder captures the decode path into a graph, which moves the engine's
    #                          per-step work somewhere the deployment under test does not have it.
    #                          Not de-tuning: not eager, symmetric across arms, and the mode this
    #                          model plus this speculator actually runs in.
    #   --cuda-graph-sizes     max_num_seqs * (1 + num_speculative_tokens). One element of this flag
    #                          is a CEILING, expanded to [1,2,4] + range(8, N+1, 8). Its default,
    #                          min(max_num_seqs*2, 512), is sized for one query token per request and
    #                          does not reach a speculative decode batch, which leaves the largest
    #                          decode steps outside every captured graph. Set in both arms.
    #   --max-num-seqs 256     the deployment's admission cap, the same for every shape; the shapes
    #                          vary the client's concurrency, never the server.
    #   --max-num-batched-tokens 16384  the scheduler budget the deployment runs; the API server's
    #                          own default is smaller, so it is set rather than inherited.
    #   --no-enable-chunked-prefill  as the deployment under test runs.
    #   --load-format auto     real weights. Under `dummy` the draft head predicts from random
    #                          parameters and its acceptance rate collapses, which changes the
    #                          behaviour being measured rather than merely simplifying it.
    max_seqs = 256
    return [
        _engine_python(), "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--port", str(port),
        "--tensor-parallel-size", "4",
        "--max-num-seqs", str(max_seqs),
        "--max-num-batched-tokens", "16384",
        "--max-model-len", str(max_len),
        "--speculative-config", '{"method":"qwen3_next_mtp","num_speculative_tokens":2}',
        "-O.cudagraph_mode=PIECEWISE",
        "--cuda-graph-sizes", str(max_seqs * 3),
        "--gpu-memory-utilization", "0.9",
        "--load-format", "auto",
        "--no-enable-chunked-prefill",
        "--seed", "12345",
        "--disable-log-stats",
        "--trust-remote-code",
    ]


def _bench_cmd(model: str, wl: dict, out_json: Path, port: int, seed: int) -> list[str]:
    return [
        _engine_python(), "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--model", model, "--tokenizer", model,
        "--port", str(port), "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(int(wl["input_len"])), "--random-output-len", str(int(wl["output_len"])),
        # Zero, explicitly: every shape runs exactly the declared lengths.
        "--random-range-ratio", "0.0",
        "--max-concurrency", str(int(wl.get("max_concurrency") or wl["num_prompts"])),
        "--num-prompts", str(int(wl["num_prompts"])),
    ]
    if wl.get("request_rate") is not None:
        # a finite arrival rate (requests/s, Poisson): the batch changes shape continuously instead of in waves
        cmd += ["--request-rate", str(float(wl["request_rate"])), "--burstiness", "1.0"]
    cmd += [
        "--ignore-eos",
        "--seed", str(seed),
        "--save-result", "--result-filename", str(out_json),
        # Sampling set explicitly on both arms rather than read off the checkpoint's generation_config.
        "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0",
        "--trust-remote-code",
    ]


def _wait_ready(server: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm serve exited early rc={server.returncode}; see the .serve.log")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 -- not up yet
            pass
        time.sleep(5)
    raise RuntimeError(f"vllm serve never became ready within {timeout:.0f}s; see the .serve.log")


def _stop(server: subprocess.Popen) -> None:
    if server.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        try:
            server.wait(timeout=180)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                server.wait(timeout=30)


def _read_record(out_json: Path, wl: dict) -> dict | None:
    """One shape's reading, or None when the run did not deliver the declared shape."""
    if not out_json.is_file():
        return None
    doc = json.loads(out_json.read_text())
    tps = doc.get("total_token_throughput")
    if int(doc.get("completed") or 0) != int(wl["num_prompts"]) or int(doc.get("failed") or 0):
        return None
    if type(tps) not in (int, float) or tps <= 0:
        return None
    return {"total_token_throughput": float(tps), "output_throughput": doc.get("output_throughput"),
            "mean_ttft_ms": doc.get("mean_ttft_ms"), "mean_tpot_ms": doc.get("mean_tpot_ms"),
            "mean_itl_ms": doc.get("mean_itl_ms"), "completed": doc.get("completed")}


def _bench(tree: Path, ws: Path, raw: Path, label: str, points: list[str], seed: int,
           port: int = DEFAULT_PORT) -> dict[str, dict | None]:
    """Every named shape on `tree`: each scored shape on a fresh server of its own, the guard shapes
    against the anchor's server after it; each shape's record, or None if it failed."""
    matrix = _matrix(ws)
    guards = [n for n in points if n in matrix["guards"]]
    docs: dict[str, dict | None] = {}
    for host in [n for n in points if n not in matrix["guards"]]:
        riders = guards if host == matrix["anchor"] else []
        docs.update(_serve_once(tree, ws, raw, f"{label}_{host}", [host] + riders, seed, port))
    return docs


def _serve_once(tree: Path, ws: Path, raw: Path, label: str, points: list[str], seed: int,
                port: int = DEFAULT_PORT) -> dict[str, dict | None]:
    """One fresh server on `tree`, the named shapes against it in order; a record per shape, or None."""
    matrix = _matrix(ws)
    anchor = matrix["points"][matrix["anchor"]]
    max_len = int(anchor.get("max_model_len") or 0) or max(
        4096, int(anchor["input_len"]) + int(anchor["output_len"]) + 256
    )
    model = _model(ws)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}:{env.get('PYTHONPATH', '')}".rstrip(":")
    # A COLD COMPILE CACHE PER SERVER, which is what the verifier gives each of its servers too. Keyed
    # on the tree being benched so the two arms never collide, and wiped before every boot so each
    # server starts where the verifier's would. The shapes of one server share it, as in the deployment.
    scratch = Path(tempfile.gettempdir()) / f"infb_jit_{abs(hash(str(tree))) % (10 ** 10)}"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "triton").mkdir(parents=True, exist_ok=True)
    (scratch / "vllm").mkdir(parents=True, exist_ok=True)
    env["TRITON_CACHE_DIR"] = str(scratch / "triton")
    env["VLLM_CACHE_ROOT"] = str(scratch / "vllm")
    docs: dict[str, dict | None] = {n: None for n in points}
    with (raw / f"{label}.serve.log").open("w") as fh:
        server = subprocess.Popen(_serve_cmd(model, max_len, port), env=env, cwd="/tmp",
                                  stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            t0 = time.time()
            _wait_ready(server, port, BOOT_TIMEOUT_SEC)
            print(f"[run_benchmark] {label}: server ready after {time.time() - t0:.0f}s", flush=True)
            for name in points:
                wl = matrix["points"][name]
                out_json = raw / f"{label}_{name}.json"
                out_json.unlink(missing_ok=True)
                t1 = time.time()
                with (raw / f"{label}_{name}.log").open("w") as pl:
                    subprocess.run(_bench_cmd(model, wl, out_json, port, seed), env=env, cwd="/tmp",
                                   stdout=pl, stderr=subprocess.STDOUT)
                docs[name] = _read_record(out_json, wl)
                tps = docs[name]["total_token_throughput"] if docs[name] else None
                print(f"[run_benchmark] {label:22s} {name:34s} "
                      + (f"{tps:9.1f} tok/s" if tps else "FAILED (see iterations/.raw/)")
                      + f"   [{time.time() - t1:.0f}s]", flush=True)
                if server.poll() is not None:
                    print(f"[run_benchmark] {label}: the server died; see {label}.serve.log", flush=True)
                    break
        except RuntimeError as exc:
            print(f"[run_benchmark] {label}: {exc}", file=sys.stderr, flush=True)
        finally:
            _stop(server)
    return docs


def _stage_pristine(ws: Path, dest: Path) -> Path | None:
    """A pristine copy of the pinned tree, from repo/'s own `inference_base` tag."""
    repo = ws / "repo"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    proc = subprocess.run(["git", "-C", str(repo), "archive", "inference_base"], capture_output=True)
    if proc.returncode != 0:
        print(f"[run_benchmark] cannot stage pristine: {proc.stderr.decode()[:300]}", file=sys.stderr)
        return None
    if subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout).returncode != 0:
        return None
    # The compiled extensions live in the venv and are symlinked into repo/; mirror them, or the
    # pristine arm imports nothing and the comparison is against a corpse.
    for pat in ("*.so", "_version.py"):
        for src in (repo / "vllm").glob(pat):
            tgt = dest / "vllm" / src.name
            if not tgt.exists():
                tgt.symlink_to(src.resolve())
    # EVERYTHING under vllm_flash_attn, not just `*.so`. At this pin git holds a single `.gitkeep`
    # there: the package's Python AND its `layers/` and `ops/` subpackages are generated at build
    # time and reach `repo/` as overlay symlinks. Mirroring only the `.so` leaves the pristine arm
    # importing an empty namespace package, which dies inside the model registry with
    #     ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'
    # and reports it as a bench failure rather than as a staging failure.
    fa_src, fa_dst = repo / "vllm" / "vllm_flash_attn", dest / "vllm" / "vllm_flash_attn"
    if fa_src.is_dir() and fa_dst.is_dir():
        for src in fa_src.iterdir():
            if src.name == "__pycache__":
                continue
            tgt = fa_dst / src.name
            if not tgt.exists():
                tgt.symlink_to(src.resolve())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=1, help="alternating A/B rounds")
    ap.add_argument("--with-baseline", action="store_true",
                    help="also bench a pristine copy staged from the inference_base tag")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--workloads", default="all",
                    help="`all` (every scored shape of public/workloads.json on a server of its own, the guard "
                         "shapes after the anchor on its server, per arm and round), or a comma-separated list; "
                         "the anchor is always benched")
    ap.add_argument("--seed", type=int, default=None,
                    help="prompt seed shared by both arms and every shape (default: INFB_BENCH_SEED, else time-based)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    ws = _workspace()
    iters = ws / "iterations"
    iters.mkdir(exist_ok=True)
    if args.output is None:
        used = [int(m.group(1)) for p in iters.glob("iter_*.json")
                if (m := re.match(r"iter_(\d+)\.json$", p.name))]
        args.output = iters / f"iter_{max(used, default=0) + 1}.json"

    raw = iters / ".raw"
    raw.mkdir(exist_ok=True)
    cand_tree = ws / "repo"
    base_tree = _stage_pristine(ws, ws / ".baseline_tree") if args.with_baseline else None
    if args.with_baseline and base_tree is None:
        print("[run_benchmark] baseline staging failed; benching the candidate only", file=sys.stderr)

    matrix = _matrix(ws)
    anchor = matrix["anchor"]
    benched = [n for n in matrix["benched"]
               if args.workloads == "all" or n == anchor or n in args.workloads.split(",")]
    seed = _seed(args.seed)
    print(f"[run_benchmark] prompt seed {seed} (pass --seed to reproduce); shapes {benched}", flush=True)

    if not args.no_warmup:
        # EVERY arm gets one, and warming only the candidate was the bug here. The first server a tree
        # boots in a fresh process pays one-time initialisation, so a single warm-up on the candidate
        # leaves the baseline paying it inside a scored rep -- a bias that flatters the candidate by
        for _label, _tree in [("candidate", cand_tree)] + ([("baseline", base_tree)] if base_tree else []):
            print(f"[run_benchmark] warm-up (discarded): {_label}", flush=True)
            _bench(_tree, ws, raw, f"warmup_{_label}", [anchor], seed, args.port)

    docs: dict[str, dict[str, list[dict]]] = {"candidate": {n: [] for n in benched},
                                              "baseline": {n: [] for n in benched}}
    for r in range(1, args.rounds + 1):
        order = [("candidate", cand_tree)]
        if base_tree is not None:
            order.append(("baseline", base_tree))
        if r % 2 == 0:
            order.reverse()
        for label, tree in order:
            got = _bench(tree, ws, raw, f"r{r}_{label}", benched, seed, args.port)
            for n, doc in got.items():
                if doc:
                    docs[label][n].append(doc)

    def mean_value(label: str, name: str) -> float | None:
        values = [d["total_token_throughput"] for d in docs[label].get(name, []) if d]
        return statistics.mean(values) if values else None

    result: dict[str, object] = {
        "workload": matrix["points"][anchor],
        "seed": seed,
        "rounds": args.rounds,
        "candidate_total_token_throughput": mean_value("candidate", anchor),
        "baseline_total_token_throughput": mean_value("baseline", anchor),
        "points": {n: {"role": "scored" if n in matrix["scored"] else "guard",
                       "candidate": {"total_token_throughput": mean_value("candidate", n)},
                       "baseline": {"total_token_throughput": mean_value("baseline", n)}}
                   for n in benched},
    }
    per_point: dict[str, float] = {}
    for n in benched:
        c, b = mean_value("candidate", n), mean_value("baseline", n)
        if c and b:
            per_point[n] = c / b
    if per_point:
        result["speedup_per_point"] = per_point
        scored = [per_point[n] for n in matrix["scored"] if n in per_point]
        if len(scored) == len([n for n in matrix["scored"] if n in benched]):
            result["speedup"] = _geomean(scored)
        result["guards"] = {n: per_point.get(n) for n in matrix["guards"]}
    cand_anchor = [d["total_token_throughput"] for d in docs["candidate"][anchor]]
    base_anchor = [d["total_token_throughput"] for d in docs["baseline"][anchor]]
    if len(cand_anchor) > 1:
        result["candidate_sd"] = statistics.stdev(cand_anchor)
    if len(base_anchor) > 1:
        result["baseline_sd"] = statistics.stdev(base_anchor)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n[run_benchmark] wrote {args.output}")
    for n, s in (result.get("speedup_per_point") or {}).items():
        print(f"[run_benchmark] {n:34s} {s:.4f}x  ({'scored' if n in matrix['scored'] else 'guard'})")
    if result.get("speedup"):
        print(f"[run_benchmark] speedup = {result['speedup']:.4f}  (geometric mean over the scored shapes; "
              f"the score is the measured speedup itself, and a guard shape that reads slower on the "
              f"candidate scores zero)")
    elif not args.with_baseline:
        print("[run_benchmark] no ratio: pass --with-baseline to bench the pinned tree too")
    return 0 if any(docs["candidate"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
