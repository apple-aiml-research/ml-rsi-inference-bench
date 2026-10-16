#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-test probe: bench your edited tree on every shape in public/workloads.json.

Starts a server from `repo/`, drives each shape against it with public/_mm_ttft_client.py -- the
client the scored run uses -- tears the server down and writes iterations/iter_N.json. With
--with-baseline the unmodified tree is staged from the `inference_base` tag and benched the same
way, so the file carries every shape's ratio, the geometric mean over the scored shapes and the
guard readings.

Two cautions: bench each tree more than once and alternate them (`--rounds 2`), and keep the
discarded warm-up -- the first server in a fresh process pays one-time initialisation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 30118
READY_TIMEOUT_SEC = 900.0
REQUEST_TIMEOUT_SEC = 300
POINT_BUDGET_SEC = 900


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
    if Path("/opt/venv/bin/python").is_file():
        return "/opt/venv/bin/python"
    return sys.executable


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link.resolve())
    raise SystemExit("no model: set INFB_MODEL_PATH or stage <workspace>/model")


def _matrix(ws: Path) -> dict:
    """Every shape of public/workloads.json: the anchor, the other scored shapes, the guards."""
    doc = json.loads((ws / "public" / "workloads.json").read_text())
    points = {w["name"]: w for w in doc.get("workloads", []) if isinstance(w, dict) and "name" in w}
    anchor = doc.get("primary_workload") or next(iter(points))
    scored = [str(n) for n in doc.get("scored_workloads") or [anchor]]
    guards = [str(n) for n in doc.get("guard_workloads") or []]
    order = [anchor] + [n for n in scored if n != anchor] + [n for n in guards
                                                            if not isinstance(points[n].get("from"), str)]
    return {"anchor": anchor, "scored": scored, "guards": guards, "points": points, "order": order}


def _geomean(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _serve_argv(python: str, model: str, port: int) -> list[str]:
    """The server the scored run boots, flag for flag."""
    return [
        python, "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp-size", "1",
        "--mem-fraction-static", "0.85",
        "--context-length", "8192",
        "--chunked-prefill-size", "8192",
        "--cuda-graph-max-bs-decode", "4",
        "--disable-radix-cache",
        "--random-seed", "1234",
        "--trust-remote-code",
        "--log-level", "warning",
    ]


def _client_argv(python: str, client_py: Path, base_url: str, tokenizer: str, wl: dict, seed: int,
                 out_json: Path, budget_sec: float) -> list[str]:
    """One shape's client invocation, the same list the scored run builds."""
    return [
        python, str(client_py),
        "--base-url", base_url,
        "--tokenizer", tokenizer,
        "--input-len", str(int(wl["input_len"])),
        "--output-len", str(int(wl["output_len"])),
        "--num-prompts", str(int(wl["num_prompts"])),
        "--image-count", str(int(wl.get("image_count", 0))),
        "--image-format", str(wl.get("image_format", "jpeg")),
        "--image-min-hw", f"{int(wl.get('image_min_height', 512))}x{int(wl.get('image_min_width', 512))}",
        "--image-max-hw", f"{int(wl.get('image_max_height', 768))}x{int(wl.get('image_max_width', 1024))}",
        "--seed", str(int(seed)),
        "--request-timeout-sec", str(int(REQUEST_TIMEOUT_SEC)),
        "--time-budget-sec", str(int(budget_sec)),
        "--out-json", str(out_json),
    ]


def _fresh_home() -> Path:
    """A per-boot HOME for the server, so no boot inherits on-disk state from an earlier one.
    `~/.cache` (read-only, expensive to rebuild) stays shared."""
    home = Path(tempfile.mkdtemp(prefix="run_benchmark_home_")) / "a"
    home.mkdir()
    shared = Path(os.environ.get("HOME", "/root")) / ".cache"
    if shared.is_dir():
        (home / ".cache").symlink_to(shared, target_is_directory=True)
    return home


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3.0) as r:  # noqa: S310 -- fixed localhost
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _stop(proc: subprocess.Popen, port: int) -> None:
    """Signal the server's whole process group; scheduler and detokenizer are separate processes."""
    if proc.poll() is None:
        for sig, grace in ((signal.SIGTERM, 60.0), (signal.SIGKILL, 30.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
    deadline = time.time() + 60
    while _port_busy(port) and time.time() < deadline:
        time.sleep(1.0)


def _gpu_used_mb() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return max(int(v) for v in out.split() if v.strip().isdigit())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _stage_pristine(ws: Path, dest: Path) -> Path | None:
    """The unmodified tree, from repo/'s own `inference_base` tag, plus the build outputs git ignores."""
    repo = ws / "repo"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    proc = subprocess.run(["git", "-C", str(repo), "archive", "inference_base"], capture_output=True)
    if proc.returncode != 0:
        print(f"[run_benchmark] cannot stage the unmodified tree: {proc.stderr.decode()[:300]}",
              file=sys.stderr)
        return None
    if subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout).returncode != 0:
        return None
    ignored = subprocess.run(["git", "-C", str(repo), "ls-files", "-o", "-i", "--exclude-standard"],
                             capture_output=True, text=True)
    for rel in ignored.stdout.splitlines():
        src = repo / rel
        if src.is_file() and "__pycache__" not in src.parts:
            tgt = dest / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
    return dest


def _session(tree: Path, ws: Path, python: str, model: str, port: int, points: list[str],
             matrix: dict, seed: int, out_dir: Path, label: str) -> dict[str, dict]:
    """One server from `tree`, every requested shape against it; returns each shape's record."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}:{env.get('PYTHONPATH', '')}".rstrip(":")
    for k in list(env):
        # The hub's offline switches break engine init; a cache override by name would outrank the
        # fresh HOME below and point every boot at one shared place.
        if k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") or k.endswith(("_CACHE_DIR", "CACHE_PATH")):
            env.pop(k, None)
    client_env = dict(os.environ)
    client_env["PYTHONPATH"] = env["PYTHONPATH"]
    home = _fresh_home()
    env["HOME"] = str(home)
    if _port_busy(port):
        raise SystemExit(f"port {port} is already in use; stop that server or pass --port")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{label}.server.log"
    records: dict[str, dict] = {}
    with log.open("w") as slog:
        server = subprocess.Popen(_serve_argv(python, model, port), env=env, stdout=slog,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < READY_TIMEOUT_SEC and server.poll() is None:
            if _http_ok(f"http://127.0.0.1:{port}/health"):
                break
            time.sleep(1.0)
        else:
            print(f"[run_benchmark] {label}: server did not come up; see {log}", file=sys.stderr)
            return records
        print(f"[run_benchmark] {label}: server ready in {time.time() - t0:.0f}s", flush=True)
        for name in points:
            wl = matrix["points"][name]
            out_json = out_dir / f"{label}.{name}.json"
            out_json.unlink(missing_ok=True)
            argv = _client_argv(python, Path(__file__).resolve().parent / "_mm_ttft_client.py",
                                f"http://127.0.0.1:{port}", model, wl, seed, out_json, POINT_BUDGET_SEC)
            t1 = time.time()
            with (out_dir / f"{label}.{name}.log").open("w") as clog:
                rc = subprocess.run(argv, env=client_env, stdout=clog, stderr=subprocess.STDOUT,
                                    timeout=POINT_BUDGET_SEC + 2 * REQUEST_TIMEOUT_SEC).returncode
            doc = json.loads(out_json.read_text()) if out_json.is_file() else {}
            ok = (rc == 0 and doc.get("completed") == wl["num_prompts"] and not doc.get("failed")
                  and doc.get("mean_ttft_ms"))
            print(f"[run_benchmark] {label:9s} {name:20s} "
                  + (f"mean TTFT {doc['mean_ttft_ms']:8.1f} ms" if ok else f"FAILED (rc={rc}, see {out_dir})")
                  + f"   [{time.time() - t1:.0f}s]", flush=True)
            if ok:
                records[name] = doc
    finally:
        _stop(server, port)
        shutil.rmtree(home.parent, ignore_errors=True)
        for _ in range(60):
            if _gpu_used_mb() < 4000:
                break
            time.sleep(5)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None, help="iterations/iter_N.json by default")
    ap.add_argument("--rounds", type=int, default=1, help="alternating rounds over the two trees")
    ap.add_argument("--with-baseline", action="store_true",
                    help="also stage and bench the unmodified tree, so the probe reports real ratios")
    ap.add_argument("--no-warmup", action="store_true", help="skip the discarded warm-up boot per tree")
    ap.add_argument("--workloads", default="all",
                    help="`all` (every scored and guard shape of public/workloads.json against one "
                         "server per tree), `anchor`, or a comma-separated list; the anchor is always run")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    ws = _workspace()
    python, model = _engine_python(), _model(ws)
    matrix = _matrix(ws)
    anchor = matrix["anchor"]
    wanted = set(args.workloads.split(",")) if args.workloads not in ("all", "anchor") else set()
    points = [n for n in matrix["order"]
              if args.workloads == "all" or n == anchor or n in wanted]
    # One seed for the whole call: both trees, every shape and every round see identical requests.
    seed = int(os.environ.get("INFB_BENCH_SEED") or int(time.time()) % 2_000_000_000)

    iters = ws / "iterations"
    iters.mkdir(exist_ok=True)
    if args.output is None:
        used = [int(m.group(1)) for p in iters.glob("iter_*.json")
                if (m := re.match(r"iter_(\d+)\.json$", p.name))]
        args.output = iters / f"iter_{max(used, default=0) + 1}.json"
    raw = iters / ".raw" / args.output.stem
    raw.mkdir(parents=True, exist_ok=True)

    trees: list[tuple[str, Path]] = [("candidate", ws / "repo")]
    if args.with_baseline:
        base = _stage_pristine(ws, ws / ".baseline_tree")
        if base is None:
            print("[run_benchmark] staging failed; benching the candidate only", file=sys.stderr)
        else:
            trees.append(("baseline", base))
    print(f"[run_benchmark] seed {seed}; shapes {points}; trees {[t for t, _ in trees]}", flush=True)

    if not args.no_warmup:
        for label, tree in trees:
            print(f"[run_benchmark] warm-up ({label}, discarded): one server, the anchor only", flush=True)
            _session(tree, ws, python, model, args.port, [anchor], matrix, seed, raw, f"warmup_{label}")

    docs: dict[str, dict[str, list[dict]]] = {label: {n: [] for n in points} for label, _ in trees}
    for r in range(1, args.rounds + 1):
        order = list(trees) if r % 2 else list(reversed(trees))
        for label, tree in order:
            for name, doc in _session(tree, ws, python, model, args.port, points, matrix, seed, raw,
                                      f"r{r}_{label}").items():
                docs[label][name].append(doc)

    def mean_ttft(label: str, name: str) -> float | None:
        values = [d["mean_ttft_ms"] for d in docs.get(label, {}).get(name, []) if d.get("mean_ttft_ms")]
        return statistics.fmean(values) if values else None

    result: dict[str, object] = {
        "seed": seed, "rounds": args.rounds, "shapes": points,
        "scored_workloads": matrix["scored"], "guard_workloads": matrix["guards"],
        "metric": "mean_ttft_ms", "lower_is_better": True,
        "points": {n: {"role": "scored" if n in matrix["scored"] else "guard",
                       "candidate_mean_ttft_ms": mean_ttft("candidate", n),
                       "baseline_mean_ttft_ms": mean_ttft("baseline", n),
                       "mean_ttft_ms": mean_ttft("candidate", n)}
                   for n in points},
    }
    ratios: dict[str, float] = {}
    for n in points:
        c, b = mean_ttft("candidate", n), mean_ttft("baseline", n)
        if c and b:
            ratios[n] = b / c  # TTFT: lower is better, so baseline over candidate
    if ratios:
        result["speedup_per_point"] = ratios
        scored = [ratios[n] for n in matrix["scored"] if n in ratios]
        if len(scored) == len([n for n in matrix["scored"] if n in points]):
            result["speedup"] = _geomean(scored)
        result["guards"] = {n: {"ratio": ratios.get(n)} for n in matrix["guards"] if n in points}
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n[run_benchmark] wrote {args.output}")
    for n in points:
        role = "scored" if n in matrix["scored"] else "guard"
        c = mean_ttft("candidate", n)
        line = f"[run_benchmark] {n:20s} {role:6s} candidate {c:8.1f} ms" if c else f"[run_benchmark] {n:20s} {role:6s} candidate FAILED"
        if n in ratios:
            line += f"   baseline {mean_ttft('baseline', n):8.1f} ms   ratio {ratios[n]:.4f}x"
        print(line)
    if result.get("speedup"):
        print(f"[run_benchmark] speedup = {result['speedup']:.4f}  (geometric mean of baseline/candidate "
              f"mean TTFT over the scored shapes)")
        for n, g in (result.get("guards") or {}).items():
            if g["ratio"] is not None:
                print(f"[run_benchmark] guard {n}: {g['ratio']:.4f}x -> "
                      + "(a guard shape that reads slower than the pristine tree scores zero)")
    elif not args.with_baseline:
        print("[run_benchmark] no ratio: pass --with-baseline to bench the unmodified tree too")
    return 0 if any(docs["candidate"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
