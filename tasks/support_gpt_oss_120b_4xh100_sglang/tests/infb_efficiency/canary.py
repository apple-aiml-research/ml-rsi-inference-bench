#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The environment canary. A fixed GPU workload and a fixed CPU workload are read
before and after every efficiency draw, on both sides. A draw measured while the host itself ran slow is then
re-drawn, not scored as the tree's speed.

WHY. A server can run at about half its usual pace, in prefill and in decode, for reasons that have nothing to do
with the code being measured: a power or clock cap, a starved CPU, load outside the container. A tree at the bar
then reads below it, and the same bytes measured again read normal. The servers' own logs can show that a draw was
slow. They cannot show whether the host or the tree made it so. Two fixed workloads can, because they are the same
work every time.

What one reading is (`canary.py read`, run with the engine's interpreter between draws, while the server is up and
idle):
  * GPU: on every visible GPU at once, a fixed loop of bf16 batched matmuls (4 x 8192 x 8192, `--gpu-iters`, about 1 s
    on a current datacentre GPU), timed on each device by CUDA events. The reading is TFLOP/s per GPU. Beside it, taken
    while the loop runs, nvidia-smi's SM clock, power draw, temperature and active clock-event (throttle) reasons for
    every GPU.
  * CPU: a fixed pure-Python integer loop (`--cpu-iters`, about 0.13 s on one server core), run five times in the
    same process. The reading is the best repetition's millions of iterations per second. The engine's scheduler loop
    is single-threaded Python, and this loop stands for it.

How a draw is judged (nodes.Bracket). A draw is environment-invalid when either bracketing reading reads more than
`tolerance` below the trial's first reading, on any GPU or on the CPU. The bracketing readings are the one taken just
before the draw and the one just after it; the default tolerance is 0.10, so a ratio under 0.90 is invalid.
  * An invalid draw is re-drawn once, whichever side it is on and whatever its ratio.
  * If the re-draw is invalid too, the axis is recorded as environment-invalid. The verifier then asks for a
    verifier-only re-run of the trial on the same tree (`rerun-request.json`), and the re-run is appended beside this
    trial.
  * The canary never decides a verdict. It decides only whether a draw is re-drawn, the same way for a pass and a
    fail, which keeps it from biasing toward either. A re-drawn draw is replaced, so it can change which draw decides.
  * A reading that could not be taken (no GPU, no torch, a timeout) judges nothing and is recorded as unavailable.

Where it ran. Every reading also names its host from inside the container: the kernel's boot_id (one per host boot),
each GPU's PCI bus id, uuid and NUMA node, the CPUs this process may run on, and the container's own CPU, memory and
I/O pressure (its cgroup's PSI, throttling and memory against its limit). And who was running: the five visible
processes that used the most CPU while the reading ran, each as CPUs' worth over the reading. Nothing else in the
trial's records says where a reading was taken, so these are what tell two readings apart.

What it cannot see. A competitor that comes and goes inside one draw can fall between two brackets: a slow window
narrower than the gap between two readings reads normal at both ends. The per-draw host-load record samples inside
the draw.

Standard library only at import. The GPU part imports torch inside `read`, in the engine's interpreter.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOLERANCE_DEFAULT = 0.10
GPU_N = 8192
GPU_BATCH = 4
GPU_ITERS_DEFAULT = 400
GPU_WARMUP = 5
CPU_ITERS_DEFAULT = 2_000_000
CPU_REPS = 5
TIMEOUT_S = 180.0
_SMI_BASE = "index,pci.bus_id,uuid,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu"
_SMI_REASONS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cpu(iters: int, reps: int = CPU_REPS) -> dict:
    """The loop `reps` times; the reading is the best repetition. A transient (one of the server's own threads waking
    beside it) slows one repetition, while a starved CPU slows every one, which is the difference the best of `reps`
    reads out."""
    rates = []
    total = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        acc = 0
        for i in range(iters):
            acc = (acc + i * i) & 0xFFFFFFFF
        s = time.perf_counter() - t0
        total += s
        rates.append(round(iters / s / 1e6, 4))
    return {"iters": iters, "reps": reps, "seconds": round(total, 4), "rate": max(rates), "rates": rates}


def _ranges(cpus: set[int]) -> str:
    out, run = [], []
    for c in sorted(cpus):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _pressure(path: Path) -> dict | None:
    """One PSI file's `some`/`full` avg10 (the share of the last 10 s in which a task stalled on the resource)."""
    try:
        out = {}
        for line in path.read_text().splitlines():
            kind, *fields = line.split()
            out[kind] = float(dict(f.split("=") for f in fields)["avg10"])
        return out
    except (OSError, ValueError, KeyError):
        return None


def host_identity() -> dict:
    """Where this reading ran, from inside the container: the kernel's boot_id (one per host boot), the CPUs this
    process may run on, the hostname, and the container's own resource pressure: its cgroup's CPU,
    memory and I/O stall shares (PSI avg10), its CPU throttling count and its memory use against its limit. Starving a
    server's CPUs halves its pace, and these read that directly. A part that cannot
    be read is left out."""
    out: dict = {"hostname": os.uname().nodename}
    try:
        out["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        pass
    try:
        out["cpu_affinity"] = _ranges(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    cg = Path("/sys/fs/cgroup")
    pressure = {r: _pressure(cg / f"{r}.pressure") for r in ("cpu", "memory", "io")}
    if any(pressure.values()):
        out["cgroup_pressure_avg10"] = {r: v for r, v in pressure.items() if v}
    try:
        stat = dict(line.split() for line in (cg / "cpu.stat").read_text().splitlines())
        out["cgroup_cpu"] = {k: int(stat[k]) for k in ("nr_periods", "nr_throttled", "throttled_usec") if k in stat}
    except (OSError, ValueError):
        pass
    try:
        out["cgroup_memory"] = {"current": int((cg / "memory.current").read_text()),
                                "max": (cg / "memory.max").read_text().strip()}
    except (OSError, ValueError):
        pass
    return out


def _proc_cpu() -> dict[int, tuple[str, int]]:
    """Every visible process's cumulative CPU ticks (utime + stime) and name, from /proc; the canary's own excluded."""
    out: dict[int, tuple[str, int]] = {}
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            raw = (d / "stat").read_text()
            comm = raw[raw.index("(") + 1:raw.rindex(")")]
            fields = raw[raw.rindex(")") + 2:].split()
            out[int(d.name)] = (comm, int(fields[11]) + int(fields[12]))
        except (OSError, ValueError, IndexError):
            continue
    return out


def top_consumers(before: dict[int, tuple[str, int]], after: dict[int, tuple[str, int]], seconds: float,
                  n: int = 5) -> list[dict]:
    """The processes that used the most CPU between two /proc snapshots, as CPUs' worth over the interval: who, beside
    the canary, was running while it read (the pressure fields say that something stalled; this says who)."""
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    rows = []
    for pid, (comm, t1) in after.items():
        t0 = before.get(pid, (comm, t1))[1] if pid in before else 0
        if t1 > t0:
            rows.append({"pid": pid, "comm": comm, "cpus": round((t1 - t0) / tick / max(seconds, 1e-3), 2)})
    return sorted(rows, key=lambda r: -r["cpus"])[:n]


def _numa_node(bus_id: str) -> int | None:
    """A GPU's NUMA node from sysfs; nvidia-smi's bus id has an 8-digit domain, sysfs a 4-digit one."""
    try:
        dom, rest = bus_id.lower().split(":", 1)
        return int(Path(f"/sys/bus/pci/devices/{dom[-4:]}:{rest}/numa_node").read_text().strip())
    except (OSError, ValueError):
        return None


def _smi_start() -> tuple[subprocess.Popen | None, str, str | None]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None, "", "nvidia-smi is not on PATH"
    fields = f"{_SMI_BASE},{_SMI_REASONS[0]}"
    p = subprocess.Popen([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    return p, fields, None


def _smi_collect(p: subprocess.Popen | None, fields: str, why: str | None) -> list[dict] | str:
    """nvidia-smi's rows as dicts, or why there are none (a failing nvidia-smi prints its complaint on stdout)."""
    if p is None:
        return why or "no nvidia-smi"
    try:
        out, _ = p.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        return "nvidia-smi gave no answer in 30 s"
    if p.returncode != 0 and fields.endswith(_SMI_REASONS[0]):
        smi = shutil.which("nvidia-smi") or "nvidia-smi"
        fields = f"{_SMI_BASE},{_SMI_REASONS[1]}"
        q = subprocess.run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=30)
        out, p = q.stdout, q
    keys = [k.replace("clocks_event_reasons.active", "reasons").replace("clocks_throttle_reasons.active", "reasons")
            for k in fields.split(",")]
    rows = [[c.strip() for c in row.split(",")] for row in (out or "").strip().splitlines() if row.strip()]
    good = [dict(zip(keys, r)) for r in rows if len(r) == len(keys)]
    if p.returncode != 0 or not good or len(good) != len(rows):
        return (f"nvidia-smi exited {p.returncode}: " + ((out or "").strip() or "no output"))[:300]
    for row in good:
        row["numa_node"] = _numa_node(row.get("pci.bus_id", ""))
    return good


def _gpus(iters: int) -> tuple[list[dict], list[dict] | str]:
    import torch  # noqa: PLC0415 - the engine's interpreter has it; importing this module does not need it

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA device is visible")
    work = []
    for d in range(n):
        with torch.cuda.device(d):
            a = torch.randn(GPU_BATCH, GPU_N, GPU_N, device=f"cuda:{d}", dtype=torch.bfloat16)
            b = torch.randn(GPU_BATCH, GPU_N, GPU_N, device=f"cuda:{d}", dtype=torch.bfloat16)
            c = torch.empty_like(a)
            for _ in range(GPU_WARMUP):
                torch.bmm(a, b, out=c)
            work.append((d, a, b, c))
    for d, *_ in work:
        torch.cuda.synchronize(d)
    timed = []
    for d, a, b, c in work:
        with torch.cuda.device(d):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                torch.bmm(a, b, out=c)
            end.record()
            timed.append((d, start, end))
    smi = _smi_collect(*_smi_start())
    flop = 2.0 * GPU_BATCH * GPU_N ** 3 * iters
    out = []
    for d, start, end in timed:
        end.synchronize()
        s = start.elapsed_time(end) / 1000.0
        out.append({"device": d, "seconds": round(s, 4), "tflops": round(flop / s / 1e12, 2)})
    del work
    return out, smi


def read(gpu_iters: int = GPU_ITERS_DEFAULT, cpu_iters: int = CPU_ITERS_DEFAULT) -> dict:
    """One reading: {"at", "ok", "cpu", "gpus", "smi"} and "error" when a part could not be read."""
    out: dict = {"at": _utcnow(), "ok": True, "gpu_iters": gpu_iters, "host": host_identity()}
    procs0, t0 = _proc_cpu(), time.monotonic()
    try:
        out["cpu"] = _cpu(cpu_iters)
    except Exception as exc:  # noqa: BLE001 - a reading that fails judges nothing and says why
        out["ok"] = False
        out["error"] = f"cpu: {type(exc).__name__}: {exc}"[:300]
    try:
        out["gpus"], out["smi"] = _gpus(gpu_iters)
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = (out.get("error", "") + f" gpu: {type(exc).__name__}: {exc}")[:300].strip()
    try:
        out["top_cpu"] = top_consumers(procs0, _proc_cpu(), time.monotonic() - t0)
    except Exception:  # noqa: BLE001 - a diagnostic
        pass
    return out


def take(argv: list[str], env: dict, timeout_s: float = TIMEOUT_S) -> dict:
    """Run one reading in its own process (the engine's interpreter, `python -I`) and return what it printed."""
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout_s)
        lines = [line for line in p.stdout.splitlines() if line.startswith("{")]
        reading = json.loads(lines[-1]) if lines else {
            "ok": False, "error": f"no reading (exit {p.returncode}): " + (p.stderr or p.stdout).strip()[-300:]}
    except subprocess.TimeoutExpired:
        reading = {"ok": False, "error": f"no reading in {timeout_s:g} s"}
    except Exception as exc:  # noqa: BLE001
        reading = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    reading["took_s"] = round(time.monotonic() - t0, 2)
    return reading


def compare(first: dict | None, reading: dict | None, tolerance: float) -> dict:
    """One reading against the trial's first: {"valid": True | False | None, "worst", "ratio", "ratios"}.
    False when any GPU's TFLOP/s or the CPU's rate is below (1 - tolerance) of its first reading; None when there is
    nothing to judge by (either reading unavailable, or no quantity in common)."""
    if not first or not first.get("ok"):
        return {"valid": None, "why": "no first reading"}
    if not reading or not reading.get("ok"):
        return {"valid": None, "why": "this reading is unavailable"}
    ratios: dict[str, float] = {}
    base = {g.get("device"): g.get("tflops") for g in first.get("gpus") or []}
    for g in reading.get("gpus") or []:
        b = base.get(g.get("device"))
        if isinstance(b, (int, float)) and b > 0 and isinstance(g.get("tflops"), (int, float)):
            ratios[f"gpu{g['device']}"] = g["tflops"] / b
    b, r = (first.get("cpu") or {}).get("rate"), (reading.get("cpu") or {}).get("rate")
    if isinstance(b, (int, float)) and b > 0 and isinstance(r, (int, float)):
        ratios["cpu"] = r / b
    if not ratios:
        return {"valid": None, "why": "no quantity in common with the first reading"}
    worst = min(ratios, key=ratios.__getitem__)
    return {"valid": ratios[worst] >= 1.0 - tolerance, "worst": worst, "ratio": round(ratios[worst], 4),
            "ratios": {k: round(v, 4) for k, v in sorted(ratios.items())}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="take one reading and print it as one JSON line")
    r.add_argument("--gpu-iters", type=int, default=GPU_ITERS_DEFAULT)
    r.add_argument("--cpu-iters", type=int, default=CPU_ITERS_DEFAULT)
    args = ap.parse_args(argv)
    print(json.dumps(read(args.gpu_iters, args.cpu_iters)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
