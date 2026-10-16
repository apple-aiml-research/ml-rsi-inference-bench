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

TWO CAUTIONS:

  * A single unrotated sample lies. Benching A then B once can give a difference in the WRONG
    direction, because whichever arm runs second inherits a warmer machine. `--rounds N`
    alternates the arms so that bias cancels instead of accumulating.
  * The first bench in a fresh process pays one-time initialisation. A warm-up is discarded by
    default (`--no-warmup` if you know why you want to skip it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path(__file__).resolve().parents[1]


_OVERLAY_PATTERNS = (
    "*.so",
    "*.abi3.so",
    "_version.py",
    "vllm_flash_attn/*.so",
    "vllm_flash_attn/*.py",
    "third_party/flashmla/*.py",
    "third_party/triton_kernels/*.py",
)


def _engine_python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def _matrix(ws: Path) -> dict:
    """Every point of public/workloads.json: the anchor, the other scored points, the guard points."""
    doc = json.loads((ws / "public" / "workloads.json").read_text())
    points = {w["name"]: w for w in doc.get("workloads", []) if type(w) is dict and "name" in w}
    anchor = doc.get("primary_workload") or next(iter(points))
    scored = [str(n) for n in doc.get("scored_workloads") or [anchor]]
    guards = [str(n) for n in doc.get("guard_workloads") or []]
    order = [anchor] + [n for n in scored if n != anchor] + [n for n in guards
                                                            if type(points[n].get("from")) is not str]
    return {"anchor": anchor, "scored": scored, "guards": guards, "points": points, "order": order}


def _workload(ws: Path) -> dict:
    m = _matrix(ws)
    return m["points"][m["anchor"]]


def _geomean(values: list[float]) -> float | None:
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


def _bench(tree: Path, ws: Path, out_json: Path, log: Path,
           points: list[str] | None = None) -> dict[str, float | None]:
    """Boot the tree as a server and bench it the way the verifier does. Returns mean TPOT (ms) per point."""
    matrix = _matrix(ws)
    names = [n for n in matrix["order"] if points is None or n in points]
    wl = matrix["points"][matrix["anchor"]]
    in_len, out_len = int(wl["input_len"]), int(wl["output_len"])
    num_prompts = int(wl["num_prompts"])
    concurrency = int(wl.get("max_concurrency") or num_prompts)
    max_len = int(wl.get("max_model_len") or 0) or max(
        4096, in_len + out_len + 256
    )
    model = _model(ws)
    port = int(os.environ.get("INFB_BENCH_PORT", "8793"))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["VLLM_USE_FLASHINFER_MOE_FP4"] = "1"
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "3600")

    serve_cmd = [
        _engine_python(), "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--port", str(port),
        "--data-parallel-size", "8",
        "--enable-expert-parallel",
        "--max-num-seqs", "2048",
        "--max-num-batched-tokens", "16384",
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", "0.90",
        "--max-cudagraph-capture-size", "2048",
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--stream-interval", "100",
        "--load-format", "auto",
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    def bench_cmd_for(point: dict, result: Path) -> list[str]:
        return [
            _engine_python(), "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "vllm", "--model", model, "--tokenizer", model,
            "--port", str(port), "--endpoint", "/v1/completions",
            "--dataset-name", "random",
            "--random-input-len", str(int(point["input_len"])),
            "--random-output-len", str(int(point["output_len"])),
            "--max-concurrency", str(int(point.get("max_concurrency") or point["num_prompts"])),
            "--num-prompts", str(int(point["num_prompts"])),
            "--ignore-eos", "--trust-remote-code",
            "--save-result", "--result-filename", str(result),
            "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0",
        ]

    results: dict[str, Path] = {
        n: (out_json if n == matrix["anchor"] else out_json.with_name(f"{out_json.stem}.{n}.json"))
        for n in names
    }
    with log.open("w") as fh:
        server = subprocess.Popen(serve_cmd, env=env, cwd="/tmp", stdout=fh,
                                  stderr=subprocess.STDOUT)
        try:
            if not _wait_ready(port, server, log):
                return {n: None for n in names}
            for n in names:
                results[n].unlink(missing_ok=True)
                subprocess.run(bench_cmd_for(matrix["points"][n], results[n]), env=env, cwd="/tmp",
                               stdout=fh, stderr=subprocess.STDOUT)
        finally:
            server.terminate()
            try:
                server.wait(timeout=180)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=60)

    out: dict[str, float | None] = {}
    for n in names:
        path = results[n]
        if not path.is_file():
            out[n] = None
            continue
        doc = json.loads(path.read_text())
        tpot = doc.get("mean_tpot_ms")
        if tpot is None:
            tpot = (doc.get("summary", {}).get("latency_ms", {}).get("tpot", {}) or {}).get("mean")
        out[n] = float(tpot) if tpot else None
    return out


def _wait_ready(port: int, server: subprocess.Popen, log: Path, timeout: int = 3600) -> bool:
    """Poll /health until the server answers, and give up early if it has already died."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            print(f"[run_benchmark] server exited rc={server.returncode}; see {log}",
                  file=sys.stderr)
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    print(f"[run_benchmark] server not ready within {timeout}s; see {log}", file=sys.stderr)
    return False


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

    for pat in _OVERLAY_PATTERNS:
        for src in sorted((repo / "vllm").glob(pat)):
            tgt = dest / "vllm" / pat.rsplit("/", 1)[0] / src.name if "/" in pat \
                else dest / "vllm" / src.name
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if not tgt.exists():
                tgt.symlink_to(src.resolve())

    missing = [p for p in ("vllm/vllm_flash_attn/__init__.py", "vllm/vllm_flash_attn/flash_attn_interface.py")
               if (repo / p).exists() and not (dest / p).exists()]
    if missing:
        print(f"[run_benchmark] pristine staging incomplete, refusing to bench it: {missing}",
              file=sys.stderr)
        return None
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=1, help="alternating A/B rounds")
    ap.add_argument("--with-baseline", action="store_true",
                    help="also bench a pristine copy staged from the inference_base tag")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--workloads", default="all",
                    help="`all` (every scored and guard point of public/workloads.json against one "
                         "server per tree), or a comma-separated list; the anchor is always benched")
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
    points = matrix["order"] if args.workloads == "all" else [
        n for n in matrix["order"] if n == matrix["anchor"] or n in args.workloads.split(",")]
    if not args.no_warmup:
        warm = [("candidate", cand_tree)]
        if base_tree is not None:
            warm.append(("baseline", base_tree))
        for label, tree in warm:
            print(f"[run_benchmark] warm-up (discarded): {label}", flush=True)
            _bench(tree, ws, raw / f"warmup_{label}.json", raw / f"warmup_{label}.log",
                   points=[matrix["anchor"]])

    cand: dict[str, list[float]] = {n: [] for n in points}
    base: dict[str, list[float]] = {n: [] for n in points}
    for r in range(1, args.rounds + 1):
        order = [("candidate", cand_tree, cand)]
        if base_tree is not None:
            order.append(("baseline", base_tree, base))
        if r % 2 == 0:
            order.reverse()
        for label, tree, sink in order:
            t0 = time.time()
            readings = _bench(tree, ws, raw / f"r{r}_{label}.json", raw / f"r{r}_{label}.log", points=points)
            anchor_tpot = readings.get(matrix["anchor"])
            print(
                f"[run_benchmark] r{r} {label:9s} "
                + (f"{anchor_tpot:9.2f} ms TPOT (anchor)" if anchor_tpot else "FAILED (see iterations/.raw/)")
                + f"   [{time.time() - t0:.0f}s]",
                flush=True,
            )
            for n, tpot in readings.items():
                if tpot:
                    sink[n].append(tpot)
                    if n != matrix["anchor"]:
                        print(f"[run_benchmark]    {n:24s} {tpot:9.2f} ms TPOT", flush=True)

    def mean_or_none(values):
        return statistics.mean(values) if values else None

    anchor = matrix["anchor"]
    result: dict[str, object] = {
        "workload": matrix["points"][anchor],
        "rounds": args.rounds,
        "metric": "mean_tpot_ms (lower is faster)",
        "candidate_mean_tpot_ms": mean_or_none(cand[anchor]),
        "baseline_mean_tpot_ms": mean_or_none(base[anchor]),
        "points": {n: {"role": "scored" if n in matrix["scored"] else "guard",
                       "candidate_mean_tpot_ms": mean_or_none(cand[n]),
                       "baseline_mean_tpot_ms": mean_or_none(base[n])} for n in points},
    }
    per_point: dict[str, float] = {}
    for n in points:
        if cand[n] and base[n]:
            # INVERTED against the throughput form: TPOT is a cost, so the faster tree has the SMALLER
            # number and the ratio is baseline/candidate, the direction the verifier uses.
            per_point[n] = statistics.mean(base[n]) / statistics.mean(cand[n])
    if per_point:
        result["speedup_per_point"] = per_point
        scored = [per_point[n] for n in matrix["scored"] if n in per_point]
        if len(scored) == len([n for n in matrix["scored"] if n in points]):
            result["speedup"] = _geomean(scored)
        result["guards"] = {n: per_point.get(n) for n in matrix["guards"] if n in points}
    if len(cand[anchor]) > 1:
        result["candidate_sd"] = statistics.stdev(cand[anchor])
    if len(base[anchor]) > 1:
        result["baseline_sd"] = statistics.stdev(base[anchor])
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n[run_benchmark] wrote {args.output}")
    if result.get("speedup_per_point"):
        for n, s in result["speedup_per_point"].items():
            print(f"[run_benchmark] {n:24s} {s:.4f}x  ({'scored' if n in matrix['scored'] else 'guard'})")
    if result.get("speedup"):
        print(f"[run_benchmark] speedup = {result['speedup']:.4f}  "
              f"(geometric mean over the scored points of baseline/candidate mean TPOT; the score is "
              f"the measured speedup itself, and a guard point that reads slower than the pristine tree "
              f"scores zero)")
    elif not args.with_baseline:
        print("[run_benchmark] no ratio: pass --with-baseline to bench the pinned tree too")
    return 0 if any(cand.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
