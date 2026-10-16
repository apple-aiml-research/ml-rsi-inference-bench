#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-test decode-throughput probe for the agent.

NOT the sealed verifier -- this is the cheap measurement you use to decide whether your edits in
`repo/` recovered the regression. It starts a server, runs the same load the verifier runs, tears the
server down, and writes one `iter_N.json` to `iterations/`.

WHAT IT MEASURES. The scored quantity is SUSTAINED OUTPUT TOKEN THROUGHPUT, tokens/s. Higher is
better, so the reported ratio is `candidate / baseline`: above 1 means you improved it.

THE LOAD SHAPE IS FIXED, and the concurrency is the one parameter you should not change while
comparing runs. This script drives the same server flags and the same load shape as the scored runner;
change the shape here and your number stops predicting your score. In particular, re-running at a
higher concurrency to "make the benchmark more realistic" gives a comparison that no longer
distinguishes the two trees, and reads as "my change did nothing".

`output_throughput` is the scored quantity. `mean_tpot_ms` is printed beside it as a second reading of
the same run.

No absolute baseline ships with the task (the verifier benches the pristine tree live, in the same
run, on the same GPU), so either pass `--baseline-throughput`, or bench the unmodified tree yourself:

    git -C repo worktree add /tmp/pristine inference_base
    # A BARE WORKTREE CANNOT IMPORT THE ENGINE. The compiled half -- the `.so` files, `_version.py`
    # -- is linked into repo/ at preparation time and is
    # gitignored, so `worktree add` gives you the Python and none of the extensions. This line copies
    # exactly the ignored-but-present files across, which is the compiled overlay and nothing else:
    (cd repo && git ls-files -o -i --exclude-standard) | tar -C repo -cf - -T - | tar -C /tmp/pristine -xf -
    python3 public/run_benchmark.py --repo /tmp/pristine --output iterations/baseline.json
    python3 public/run_benchmark.py --baseline-throughput <the tok/s it printed>

Cost on 1xH100 TP=1: roughly 3-4 min, most of it model load.

Usage:
  python3 public/run_benchmark.py                                  # auto iter number
  python3 public/run_benchmark.py --output iterations/iter_1.json
  python3 public/run_benchmark.py --baseline-throughput <tok/s>    # report the ratio
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 31137
SERVER_READY_TIMEOUT_SEC = 900.0
WARMUP_PROMPTS = 8


def _app() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path("/app")


def _python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def _model() -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = _app() / "model"
    if link.exists():
        return str(link)
    raise SystemExit("model not resolved (set INFB_MODEL_PATH or <workspace>/model)")


def _matrix() -> dict:
    """Every point of public/workloads.json: the anchor, the other scored points, the guard points."""
    doc = json.loads((_app() / "public" / "workloads.json").read_text())
    points = {w["name"]: w for w in doc.get("workloads", []) if isinstance(w, dict) and "name" in w}
    anchor = doc.get("primary_workload") or next(iter(points))
    scored = [str(n) for n in doc.get("scored_workloads") or [anchor]]
    guards = [str(n) for n in doc.get("guard_workloads") or []]
    return {"anchor": anchor, "scored": scored, "guards": guards, "points": points}


def _benched_points(matrix: dict, selection: str) -> list[str]:
    """The points benched after the anchor: scored ones first, then guards; a guard that is a second
    reading of another point (`from`) is derived, not benched."""
    if selection == "anchor":
        return []
    names = [n for n in matrix["scored"] if n != matrix["anchor"]] + list(matrix["guards"])
    if selection != "all":
        wanted = {s.strip() for s in selection.split(",") if s.strip()}
        names = [n for n in names if n in wanted]
    return [n for n in names if not isinstance(matrix["points"][n].get("from"), str)]


def _geomean(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _point_value(rec: dict, point: dict) -> float | None:
    """The reading a point is compared on: output throughput unless the point names another key."""
    key = str(point.get("metric_key") or "throughput.output_tokens_per_sec")
    if key == "latency_ms.ttft.mean":
        value = rec.get("mean_ttft_ms")
    elif key == "latency_ms.tpot.mean":
        value = rec.get("mean_tpot_ms")
    else:
        value = rec.get("output_throughput")
    return float(value) if isinstance(value, (int, float)) else None


def _workload() -> dict:
    doc = json.loads((_app() / "public" / "workloads.json").read_text())
    wls = doc["workloads"]
    name = doc.get("primary_workload")
    for w in wls:
        if w.get("name") == name:
            return w
    return wls[0]


def _next_output(iters: Path) -> Path:
    used = []
    for p in glob.glob(str(iters / "iter_*.json")):
        m = re.search(r"iter_(\d+)\.json$", p)
        if m:
            used.append(int(m.group(1)))
    return iters / f"iter_{max(used, default=0) + 1}.json"


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


def _free_gpu_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=30,
        )
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _wait_for_a_free_gpu(need_mib: int, timeout_sec: float) -> None:
    """Block until the device is actually free."""
    free = _free_gpu_mib()
    if free is None:
        print("[probe] nvidia-smi unavailable; not waiting for a free GPU", flush=True)
        return
    deadline = time.time() + timeout_sec
    while free is not None and free < need_mib and time.time() < deadline:
        print(f"[probe] waiting for the GPU: {free} MiB free, want >= {need_mib}", flush=True)
        time.sleep(10.0)
        free = _free_gpu_mib()
    print(f"[probe] GPU free memory {free} MiB", flush=True)


def _pdeathsig() -> None:
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
    except Exception:  # noqa: BLE001 -- best effort; explicit teardown still runs
        pass
    os.setpgrp()


def _tail(p: Path, n: int = 30) -> str:
    try:
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def _serve_argv(python: str, model: str, port: int) -> list[str]:
    return [
        python, "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp-size", "1",
        "--log-level", "info",
    ]


def _bench_argv(python: str, port: int, wl: dict, n_prompts: int, out_file: Path) -> list[str]:
    declared_ratio = float(wl.get("range_ratio", 1.0))
    if declared_ratio != 1.0:
        raise SystemExit(f"workload declares range_ratio={declared_ratio}; this runner pins 1.0 (exact lengths)")
    return [
        python, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dataset-name", "random",
        "--dataset-path", "/opt/infb/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
        "--random-input-len", str(int(wl["input_len"])),
        "--random-output-len", str(int(wl["output_len"])),
        "--random-range-ratio", "1.0",
        "--num-prompts", str(n_prompts),
        "--max-concurrency", str(int(wl["max_concurrency"])),
        "--output-file", str(out_file),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="engine tree to bench (default: <workspace>/repo)")
    ap.add_argument("--output", default=None, help="where to write the iteration JSON")
    ap.add_argument("--baseline-throughput", type=float, default=None,
                    help="the unmodified tree's output tokens/s on the anchor shape, to print that ratio")
    ap.add_argument("--baseline-json", default=None,
                    help="an iteration JSON this script wrote for the unmodified tree: prints every "
                         "point's ratio, the geometric mean over the scored points and the guard readings")
    ap.add_argument("--workloads", default="all",
                    help="`all` (every scored and guard point, one server), `anchor`, or a comma list")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--free-gpu-mib", type=int, default=40000,
                    help="wait until at least this much device memory is free before starting")
    args = ap.parse_args()

    app = _app()
    repo = Path(args.repo).resolve() if args.repo else (app / "repo")
    if not (repo / "python" / "sglang").is_dir():
        raise SystemExit(f"{repo} is not an SGLang tree (no python/sglang)")

    iters = app / "iterations"
    iters.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else _next_output(iters)
    raw = iters / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    server_log = raw / "server.log"
    bench_jsonl = raw / "bench_scored.jsonl"
    warm_jsonl = raw / "bench_warmup.jsonl"
    for stale in (bench_jsonl, warm_jsonl):
        stale.unlink(missing_ok=True)

    python = _python()
    model = _model()
    wl = _workload()
    matrix = _matrix()
    extra_points = _benched_points(matrix, args.workloads)
    port = int(args.port)
    base_url = f"http://127.0.0.1:{port}"

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}/python:{env.get('PYTHONPATH', '')}".rstrip(":")

    if _port_busy(port):
        raise SystemExit(f"port {port} is already in use; a previous probe may still be running")
    _wait_for_a_free_gpu(int(args.free_gpu_mib), 600.0)

    serve_cmd = _serve_argv(python, model, port)
    print(f"[probe] serve: {shlex.join(serve_cmd)}", flush=True)

    t0 = time.time()
    record: dict = {}
    point_records: dict[str, dict | None] = {}
    with server_log.open("w") as slog:
        proc = subprocess.Popen(serve_cmd, env=env, stdout=slog, stderr=subprocess.STDOUT,
                                preexec_fn=_pdeathsig)
        try:
            deadline = time.time() + SERVER_READY_TIMEOUT_SEC
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise SystemExit(f"server exited rc={proc.returncode} before becoming ready\n"
                                     f"--- server log ---\n{_tail(server_log)}")
                if _http_ok(f"{base_url}/health"):
                    break
                time.sleep(1.0)
            else:
                raise SystemExit(f"server not ready after {SERVER_READY_TIMEOUT_SEC:.0f}s\n"
                                 f"--- server log ---\n{_tail(server_log)}")
            boot = time.time() - t0
            print(f"[probe] server ready in {boot:.0f}s", flush=True)

            print(f"[probe] warmup: {WARMUP_PROMPTS} prompts (discarded)", flush=True)
            wrc = subprocess.run(_bench_argv(python, port, wl, WARMUP_PROMPTS, warm_jsonl),
                                 env=env, check=False).returncode
            if wrc != 0:
                raise SystemExit(f"warmup bench exited rc={wrc}")

            bench_cmd = _bench_argv(python, port, wl, int(wl["num_prompts"]), bench_jsonl)
            print(f"[probe] scored: {shlex.join(bench_cmd)}", flush=True)
            brc = subprocess.run(bench_cmd, env=env, check=False).returncode
            if brc != 0:
                raise SystemExit(f"scored bench exited rc={brc}\n"
                                 f"--- server log ---\n{_tail(server_log)}")

            lines = [l for l in bench_jsonl.read_text().splitlines() if l.strip()]
            if not lines:
                raise SystemExit("bench_serving wrote no record")
            record = json.loads(lines[-1])

            for name in extra_points:
                point = matrix["points"][name]
                pj = raw / f"bench_{name}.jsonl"
                pj.unlink(missing_ok=True)
                cmd = _bench_argv(python, port, point, int(point["num_prompts"]), pj)
                print(f"[probe] point {name}: {shlex.join(cmd)}", flush=True)
                prc = subprocess.run(cmd, env=env, check=False).returncode
                plines = [l for l in pj.read_text().splitlines() if l.strip()] if pj.exists() else []
                if prc != 0 or not plines:
                    print(f"[probe] point {name}: bench failed (rc={prc}); recorded as unmeasured", flush=True)
                    point_records[name] = None
                    continue
                point_records[name] = json.loads(plines[-1])
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
            deadline = time.time() + 60
            while _port_busy(port) and time.time() < deadline:
                time.sleep(1.0)

    tps = float(record.get("output_throughput") or 0.0)
    tpot = record.get("mean_tpot_ms")
    payload = {
        "repo": str(repo),
        "model": model,
        "workload": wl,
        "output_throughput": tps,
        "mean_tpot_ms": tpot,
        "mean_ttft_ms": record.get("mean_ttft_ms"),
        "request_throughput": record.get("request_throughput"),
        "completed": record.get("completed"),
        "elapsed_sec": round(time.time() - t0, 2),
        "raw": record,
    }
    if args.baseline_throughput:
        payload["baseline_throughput"] = args.baseline_throughput
        payload["ratio_vs_baseline"] = (
            tps / args.baseline_throughput if args.baseline_throughput else None
        )
    points_out: dict[str, dict] = {}
    for name, rec in [(matrix["anchor"], record)] + list(point_records.items()):
        point = matrix["points"].get(name, {})
        points_out[name] = {
            "role": "scored" if name in matrix["scored"] else "guard",
            "metric_key": str(point.get("metric_key") or "throughput.output_tokens_per_sec"),
            "lower_is_better": bool(point.get("lower_is_better", False)),
            "value": _point_value(rec, point) if rec else None,
            "output_throughput": (rec or {}).get("output_throughput"),
            "mean_ttft_ms": (rec or {}).get("mean_ttft_ms"),
            "mean_tpot_ms": (rec or {}).get("mean_tpot_ms"),
            "completed": (rec or {}).get("completed"),
        }
    payload["points"] = points_out
    if args.baseline_json:
        base = json.loads(Path(args.baseline_json).read_text()).get("points", {})
        ratios: dict[str, float | None] = {}
        for name, mine in points_out.items():
            theirs = base.get(name, {})
            a, b = mine.get("value"), theirs.get("value")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
                ratios[name] = (b / a) if mine["lower_is_better"] else (a / b)
            else:
                ratios[name] = None
        payload["ratio_per_point_vs_baseline"] = ratios
        payload["scored_geomean_vs_baseline"] = _geomean([ratios[n] for n in matrix["scored"] if ratios.get(n)])
        payload["guards_vs_baseline"] = {n: ratios.get(n) for n in matrix["guards"]}
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"[probe] output throughput {tps:.2f} tok/s"
          + (f"   mean TPOT {tpot:.3f} ms" if isinstance(tpot, (int, float)) else ""), flush=True)
    for name, pt in points_out.items():
        if name == matrix["anchor"]:
            continue
        v = pt["value"]
        print(f"[probe] point {name} ({pt['role']}): "
              + (f"{v:.2f} {'ms' if pt['lower_is_better'] else 'tok/s'}" if isinstance(v, (int, float))
                 else "unmeasured"), flush=True)
    if args.baseline_json:
        for name, r in payload["ratio_per_point_vs_baseline"].items():
            print(f"[probe] ratio vs baseline, {name}: " + (f"{r:.4f}x" if r else "n/a"), flush=True)
        g = payload["scored_geomean_vs_baseline"]
        print("[probe] scored points, geometric mean of the ratios: " + (f"{g:.4f}x" if g else "n/a"),
              flush=True)
        for name, r in payload["guards_vs_baseline"].items():
            print(f"[probe] guard {name}: " + (f"{r:.4f}x (above 1 is faster than the baseline)" if r else "n/a"),
                  flush=True)
    elif args.baseline_throughput:
        print(f"[probe] ratio vs baseline: {payload['ratio_vs_baseline']:.4f}x", flush=True)
    else:
        print("[probe] no baseline given, so no ratio. The score is the RATIO of the two trees on these "
              "shapes, not an absolute number; run this once with --repo <pristine tree> and pass that "
              "iteration file back as --baseline-json.", flush=True)
    print(f"[probe] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
