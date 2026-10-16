#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The host's load during every efficiency draw, so a slow draw can be put on the host or on the tree.

WHY. On one recorded oracle run (2026-09-25) two of the candidate's three throughput draws ran at about
half speed -- prefill and decode together, the speculative acceptance unchanged -- in windows 60 s apart, while the
reference's draws on the same host did not. A verifier that cannot say what else the host was doing during a draw
cannot tell a slow tree from a busy host. This records it; it scores nothing.

    host_load.py sample --out FILE [--interval S]
        every S seconds until killed: wall and monotonic time, the 1-minute load average, the CPU time split since the
        last sample from /proc/stat (busy, iowait, steal, as fractions), and each GPU's utilization, SM clock, power
        and temperature (nvidia-smi); one JSON line per sample
    host_load.py summarize --samples FILE --sidecar requests.jsonl.gz --out FILE
        each efficiency draw's window from the efficiency client's own sidecar rows (their monotonic send and done times; a draw is
        a warmup block and the timed block after it), and the samples inside it: mean and worst of each quantity;
        printed as INFB_HOST_LOAD and written to --out

Task-local. Standard library only; never fails the verifier.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

_GPU_QUERY = "index,utilization.gpu,clocks.sm,power.draw,temperature.gpu"


def _cpu_times() -> list[int] | None:
    try:
        first = Path("/proc/stat").read_text().splitlines()[0].split()
    except OSError:
        return None
    return [int(v) for v in first[1:]]  # user nice system idle iowait irq softirq steal guest guest_nice


def _gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        def num(v: str) -> float | None:
            try:
                return float(v)
            except ValueError:
                return None
        rows.append({"index": int(parts[0]) if parts[0].isdigit() else parts[0], "util": num(parts[1]),
                     "sm_mhz": num(parts[2]), "power_w": num(parts[3]), "temp_c": num(parts[4])})
    return rows


def sample(out: Path, interval: float) -> None:
    prev = _cpu_times()
    with out.open("a", encoding="utf-8") as fh:
        while True:
            time.sleep(interval)
            cur = _cpu_times()
            split = {}
            if prev and cur and len(cur) >= 8:
                d = [c - p for c, p in zip(cur, prev)]
                total = sum(d[:8]) or 1
                split = {"busy": 1 - (d[3] + d[4]) / total, "iowait": d[4] / total, "steal": d[7] / total}
            prev = cur
            try:
                load1 = float(Path("/proc/loadavg").read_text().split()[0])
            except (OSError, ValueError, IndexError):
                load1 = None
            fh.write(json.dumps({"t": time.time(), "mono": time.monotonic(), "load1": load1, "cpu": split,
                                 "gpus": _gpus()}) + "\n")
            fh.flush()


def _draw_windows(sidecar: Path) -> list[dict]:
    rows = [json.loads(line) for line in gzip.open(sidecar, "rt", encoding="utf-8")]
    windows: list[dict] = []
    for r in rows:
        side, axis, phase = r.get("side"), r.get("axis"), str(r.get("phase") or "")
        t0, t1 = r.get("t_send"), r.get("done_s")
        if t0 is None:
            continue
        end = t0 + t1 if isinstance(t1, (int, float)) and t1 < t0 else (t1 if isinstance(t1, (int, float)) else t0)
        new = (not windows or windows[-1]["side"] != side or windows[-1]["axis"] != axis
               or (phase.endswith("warmup") and windows[-1]["phase"].endswith("timed")))
        if new:
            windows.append({"side": side, "axis": axis, "phase": phase, "start": t0, "end": end, "timed_start": None})
        w = windows[-1]
        w["phase"] = phase
        w["start"], w["end"] = min(w["start"], t0), max(w["end"], end)
        if phase.endswith("timed") and w["timed_start"] is None:
            w["timed_start"] = t0
    for i, w in enumerate(windows):
        w["draw"] = sum(1 for v in windows[: i + 1] if v["side"] == w["side"] and v["axis"] == w["axis"])
    return windows


def summarize(samples: Path, sidecar: Path, out: Path) -> None:
    pts = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    for w in _draw_windows(sidecar):
        lo = w["timed_start"] if w["timed_start"] is not None else w["start"]
        inside = [p for p in pts if lo <= p.get("mono", -1) <= w["end"]]
        entry = {"side": w["side"], "axis": w["axis"], "draw": w["draw"], "seconds": round(w["end"] - lo, 1),
                 "samples": len(inside)}
        if inside:
            load = [p["load1"] for p in inside if p.get("load1") is not None]
            steal = [p["cpu"].get("steal") for p in inside if p.get("cpu")]
            busy = [p["cpu"].get("busy") for p in inside if p.get("cpu")]
            util = [g["util"] for p in inside for g in p.get("gpus", []) if g.get("util") is not None]
            sm = [g["sm_mhz"] for p in inside for g in p.get("gpus", []) if g.get("sm_mhz") is not None]
            entry.update({
                "load1_mean": round(statistics.fmean(load), 2) if load else None,
                "load1_max": max(load) if load else None,
                "cpu_busy_mean": round(statistics.fmean(busy), 3) if busy else None,
                "cpu_steal_max": round(max(steal), 4) if steal else None,
                "gpu_util_mean": round(statistics.fmean(util), 1) if util else None,
                "gpu_util_min": min(util) if util else None,
                "gpu_sm_mhz_min": min(sm) if sm else None,
            })
        result.append(entry)
        print("INFB_HOST_LOAD " + json.dumps(entry), flush=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--interval", type=float, default=1.0)
    m = sub.add_parser("summarize")
    m.add_argument("--samples", type=Path, required=True)
    m.add_argument("--sidecar", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        if args.cmd == "sample":
            sample(args.out, args.interval)
        else:
            summarize(args.samples, args.sidecar, args.out)
    except Exception as exc:  # a diagnostic never fails the verifier
        print(f"[host_load] {args.cmd} failed: {exc!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
