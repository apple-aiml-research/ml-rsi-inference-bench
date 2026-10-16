#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run the scored configuration once against a tree and file the result under `iterations/`.

This is YOUR copy of the driver. The verifier runs the same two steps against a freshly staged
pristine tree and against yours and divides the two, so what is fixed is the shape, not this file --
read it, change the flags, or drive the server yourself.

Two steps, because this task is scored through a SERVER rather than an offline engine:

  1. launch `sglang.launch_server` with `PYTHONPATH` pointing at the tree under measurement, and
     wait for `/health_generate`. On this model that takes several minutes before a token is served.
  2. drive it with `aiperf`. The flags below are the scored ones; the reading is aiperf's
     `output_token_throughput`.

Usage:
    python3 public/run_benchmark.py                                   # bench repo/, file iter_N.json
    python3 public/run_benchmark.py --repo /tmp/pristine --output iterations/baseline.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from pathlib import Path as pathlib_Path

PORT = int(os.environ.get("INFB_BENCH_PORT", "31418"))
READY_TIMEOUT = int(os.environ.get("INFB_SERVER_READY_SEC", "1800"))
BENCH_TIMEOUT = int(os.environ.get("INFB_BENCH_TIMEOUT_SEC", "1800"))


def _root() -> Path:
    return Path(os.environ.get("INFB_WORKSPACE", "/app"))


def _python() -> str:
    for c in (os.environ.get("INFB_VLLM_PYTHON"), "/opt/venv/bin/python3", sys.executable):
        if c and Path(c).exists():
            return c
    return sys.executable


def _matrix() -> dict:
    """Every point of public/workloads.json: the anchor, the other scored points, the guard points."""
    doc = json.loads((_root() / "public" / "workloads.json").read_text())
    points = {w["name"]: w for w in doc.get("workloads", []) if isinstance(w, dict) and "name" in w}
    anchor = doc.get("primary_workload") or next(iter(points))
    scored = [str(n) for n in doc.get("scored_workloads") or [anchor]]
    guards = [str(n) for n in doc.get("guard_workloads") or []]
    order = [anchor] + [n for n in scored if n != anchor] + [n for n in guards
                                                            if not isinstance(points[n].get("from"), str)]
    return {"anchor": anchor, "scored": scored, "guards": guards, "points": points, "order": order}


def _geomean(values):
    values = [v for v in values if isinstance(v, (int, float)) and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _workload() -> dict:
    wl = json.loads((_root() / "public" / "workloads.json").read_text())
    entries = wl["workloads"]
    primary = wl.get("primary_workload")
    for spec in entries:
        if primary is None or spec.get("name") == primary:
            return dict(spec)
    raise SystemExit(f"[run_benchmark] primary_workload {primary!r} not in workloads.json")


def _next_iter(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.stem.split("_")[-1]) for p in d.glob("iter_*.json")
                 if p.stem.split("_")[-1].isdigit()] or [0])
    return d / f"iter_{n}.json"


def _gpu_used_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
        return sum(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(_root() / "repo"))
    ap.add_argument("--output", default="")
    ap.add_argument("--model", default=str(_root() / "model"))
    ap.add_argument("--workloads", default="all",
                    help="`all` (every scored and guard point of public/workloads.json against one "
                         "server), `anchor`, or a comma-separated list; the anchor is always benched")
    ap.add_argument("--baseline-json", default="",
                    help="an iteration JSON this script wrote for the unmodified tree: prints every "
                         "point's ratio, the geometric mean over the scored points and the guard readings")
    args = ap.parse_args()

    wl = _workload()
    matrix = _matrix()
    points = [n for n in matrix["order"]
              if args.workloads == "all" or n == matrix["anchor"] or n in args.workloads.split(",")]
    py = _python()
    out = Path(args.output) if args.output else _next_iter(_root() / "iterations")
    out.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{Path(args.repo).resolve()}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["TOKENIZERS_PARALLELISM"] = "false"
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(k, None)

    server = subprocess.Popen(
        [py, "-m", "sglang.launch_server", "--model-path", args.model,
         "--served-model-name", "scored-model", "--tp", str(int(wl.get("tp_size", 4))),
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        env=env, stdout=open(out.with_suffix(".server.log"), "w"),
        stderr=subprocess.STDOUT, start_new_session=True)
    rc = 1
    try:
        print(f"[run_benchmark] server starting (this model takes minutes) ...", flush=True)
        deadline = time.time() + READY_TIMEOUT
        ready = False
        while time.time() < deadline and server.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate", timeout=10):
                    ready = True
                    break
            except Exception:
                time.sleep(5)
        if not ready:
            print(f"[run_benchmark] server did not come up; see {out.with_suffix('.server.log')}",
                  file=sys.stderr)
            return 2
        print("[run_benchmark] server ready; benching", flush=True)
        aiperf_dir = out.with_suffix(".aiperf")
        point_rcs: dict[str, int] = {}
        for name in points:
            pt = matrix["points"][name]
            pdir = aiperf_dir if name == matrix["anchor"] else out.with_suffix(f".{name}.aiperf")
            print(f"[run_benchmark] point {name}: c={pt['concurrency']} out={pt['output_len']} "
                  f"n={pt['num_prompts']}", flush=True)
            point_rcs[name] = subprocess.run(
                [os.environ.get("INFB_AIPERF_BIN", "/opt/aiperf-venv/bin/aiperf"), "profile",
                 "--url", f"http://127.0.0.1:{PORT}", "--model", "scored-model",
                 "--tokenizer", args.model, "--tokenizer-trust-remote-code",
                 "--endpoint-type", "completions", "--streaming", "--ui", "simple",
                 "--extra-inputs", f"min_tokens:{int(pt['output_len'])}",
                 "--extra-inputs", f"max_tokens:{int(pt['output_len'])}",
                 "--extra-inputs", "ignore_eos:true",
                 "--concurrency", str(int(pt["concurrency"])),
                 "--synthetic-input-tokens-mean", str(int(pt["input_len"])),
                 "--synthetic-input-tokens-stddev", "0",
                 "--output-tokens-mean", str(int(pt["output_len"])),
                 "--output-tokens-stddev", "0",
                 "--request-count", str(int(pt["num_prompts"])), "--warmup-request-count", "1",
                 "--artifact-dir", str(pdir)],
                env=env, timeout=BENCH_TIMEOUT).returncode
        rc = point_rcs.get(matrix["anchor"], 1)
    finally:
        # The server's scheduler and detokenizer are separate processes; signal the GROUP or they
        # keep the GPUs and the next run boots alongside them.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(server.pid), sig)
                server.wait(timeout=60)
                break
            except Exception:
                continue
        for _ in range(60):
            if _gpu_used_mb() < 4000:
                break
            time.sleep(5)

    # The scored readings come out of aiperf's exports, one per point.
    def find(o, k="output_token_throughput"):
        if isinstance(o, dict):
            v = o.get(k)
            if isinstance(v, dict) and "avg" in v:
                return v["avg"]
            if isinstance(v, (int, float)):
                return v
            for x in o.values():
                r = find(x, k)
                if r is not None:
                    return r
        return None

    readings: dict[str, dict] = {}
    for name in points:
        pdir = out.with_suffix(".aiperf") if name == matrix["anchor"] else out.with_suffix(f".{name}.aiperf")
        exp = next(iter(sorted(pdir.rglob("profile_export_aiperf.json"))), None)
        if exp is None:
            readings[name] = {"output_throughput": None, "inter_token_latency_ms": None, "time_to_first_token_ms": None}
            continue
        doc = json.loads(exp.read_text())
        readings[name] = {"output_throughput": find(doc), "inter_token_latency_ms": find(doc, "inter_token_latency"),
                          "time_to_first_token_ms": find(doc, "time_to_first_token")}
    # A derived guard reads another point's record with its own metric.
    for name in matrix["guards"]:
        pt = matrix["points"][name]
        if isinstance(pt.get("from"), str) and pt["from"] in readings:
            readings[name] = dict(readings[pt["from"]])

    def value_of(name: str):
        pt = matrix["points"][name]
        key = str(pt.get("metric_key") or "throughput.output_tokens_per_sec")
        rec = readings.get(name) or {}
        if key == "latency_ms.ttft.mean":
            return rec.get("time_to_first_token_ms")
        if key == "latency_ms.itl.mean":
            return rec.get("inter_token_latency_ms")
        return rec.get("output_throughput")

    scored = readings.get(matrix["anchor"], {}).get("output_throughput")
    payload = {"output_tokens_per_second": scored, "output_throughput": scored,
               "inter_token_latency_ms": readings.get(matrix["anchor"], {}).get("inter_token_latency_ms"),
               "source": "aiperf",
               "points": {n: {"role": "scored" if n in matrix["scored"] else "guard", "value": value_of(n),
                              "lower_is_better": bool(matrix["points"][n].get("lower_is_better", False)),
                              **readings.get(n, {})} for n in list(points) + [g for g in matrix["guards"] if g not in points]}}
    if args.baseline_json:
        base = json.loads(Path(args.baseline_json).read_text()).get("points", {})
        ratios = {}
        for n, mine in payload["points"].items():
            a, b = mine.get("value"), (base.get(n) or {}).get("value")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
                ratios[n] = (b / a) if mine["lower_is_better"] else (a / b)
            else:
                ratios[n] = None
        payload["ratio_per_point_vs_baseline"] = ratios
        payload["scored_geomean_vs_baseline"] = _geomean([ratios[n] for n in matrix["scored"] if ratios.get(n)])
        payload["guards_vs_baseline"] = {n: ratios.get(n) for n in matrix["guards"]}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if scored is not None:
        print(f"[run_benchmark] SCORED (aiperf), anchor: {scored:.2f} generated tok/s", flush=True)
    for n, pt in payload["points"].items():
        if n != matrix["anchor"] and pt.get("value") is not None:
            print(f"[run_benchmark] point {n} ({pt['role']}): {pt['value']:.2f} "
                  f"{'ms' if pt['lower_is_better'] else 'tok/s'}", flush=True)
    if args.baseline_json:
        for n, r in payload["ratio_per_point_vs_baseline"].items():
            print(f"[run_benchmark] ratio vs baseline, {n}: " + (f"{r:.4f}x" if r else "n/a"), flush=True)
        g = payload["scored_geomean_vs_baseline"]
        print("[run_benchmark] scored points, geometric mean of the ratios: " + (f"{g:.4f}x" if g else "n/a"), flush=True)
        for n, r in payload["guards_vs_baseline"].items():
            print(f"[run_benchmark] guard {n}: " + (f"{r:.4f}x (above 1 is better than the baseline)" if r else "n/a"), flush=True)
    else:
        print("[run_benchmark] no baseline given: run this once with --repo <a checkout of inference_base> "
              "and pass that iteration file back as --baseline-json for the ratios you are scored on.", flush=True)
    print(f"[run_benchmark] wrote {out}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
