#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The environment canary. A fixed GPU workload and a fixed CPU workload are read before and after every efficiency
draw, on both sides. A draw measured while the host itself ran slow is then re-drawn, not scored as the tree's speed.

WHY. A server can run at about half its usual pace, in prefill and in decode, for reasons that have nothing to do
with the code being measured: a power or clock cap, a starved CPU, load outside the container. A tree at the bar
then reads below it, and the same bytes measured again read normal. The servers' own logs can show that a draw was
slow. They cannot show whether the host or the tree made it so. Two fixed workloads can, because they are the same
work every time.

What one reading is (`canary.py read`, run with the engine's interpreter between draws, while the server is up and
idle):
  * GPU: on every visible GPU at once, a fixed loop of bf16 batched matmuls (4 x 8192 x 8192, `--gpu-iters`, about 1 s
    on a B200), timed on each device by CUDA events. The reading is TFLOP/s per GPU. Beside it, taken while the loop
    runs, nvidia-smi's SM clock, power draw, temperature and active clock-event (throttle) reasons for every GPU.
  * CPU: a fixed pure-Python integer loop (`--cpu-iters`, about 0.13 s on one server core), run five times in the
    same process. The reading is the best repetition's millions of iterations per second. The engine's scheduler loop
    is single-threaded Python, and this loop stands for it.

How a draw is judged (nodes.Bracket). A draw is environment-invalid when either bracketing reading reads more than
`tolerance` below the baseline, on any GPU or on the CPU. The bracketing readings are the one taken just
before the draw and the one just after it; the default tolerance is 0.10, so a ratio under 0.90 is invalid.
  * The baseline is the running upper median of the trial's readings up to and including the one judged
    (`baseline`), and the CPU has its own `cpu_tolerance`, default 0.25; the GPUs keep `tolerance`. A whole CPU
    reading can come back somewhat slow while the draws beside it are normal, including a trial's first, which is
    why the baseline is a median and why the CPU's tolerance is the wider of the two; a process hogging the serving
    processes' own CPUs is far below either. The judgement against the trial's first reading stays in the record
    beside the median one.
  * A reading is judged on the quantities it read. A part that could not be read leaves the others judged, where
    judging the reading as a whole would discard every part of it. A GPU whose free memory is below the probe's need
    (`GPU_NEED_MIB`) is not probed at all: the reading records it under `gpus_skipped`, with its free MiB and what
    holds it, and judges the rest. Without that, a candidate serving at a high concurrency can leave too little free
    memory for the probe, which costs the reading its GPU part and with it the CPU and the other GPUs, so the draw
    beside it goes unjudged. The node counts judged and unjudged draws per axis and side.
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
import contextlib
import datetime as _dt
import json
import math
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
#: The probe's need on one GPU once this process's CUDA context is on it: its three bf16 operands
#: (3 x GPU_BATCH x GPU_N^2 x 2 B = 1,536 MiB), cuBLAS's workspace and the allocator's rounding, with room to spare.
#: A GPU with less free than this is skipped, not probed.
GPU_NEED_MIB = 2560
#: nvidia-smi renamed clocks_throttle_reasons to clocks_event_reasons; either spelling answers, so both are tried.
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
    server's CPUs halves its pace, and these read that directly. A part that cannot be read is left out."""
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


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def quantities(reading: dict | None) -> dict[str, float]:
    """What a reading read, by quantity: `gpu<device>` in TFLOP/s and `cpu` in
    millions of iterations per second. A reading is judged on these, whether or not every part of it was read."""
    out: dict[str, float] = {}
    for g in (reading or {}).get("gpus") or []:
        if _finite(g.get("tflops")):
            out[f"gpu{g.get('device')}"] = float(g["tflops"])
    rate = ((reading or {}).get("cpu") or {}).get("rate")
    if _finite(rate):
        out["cpu"] = float(rate)
    return out


def gpu_holders() -> list[dict] | str:
    """The compute processes nvidia-smi lists, as {gpu_uuid, pid, name, used_mib, here}, `here` when this
    container sees the pid; or why there are none. No command line is read."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return "nvidia-smi is not on PATH"
    try:
        p = subprocess.run([smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi --query-compute-apps: {type(exc).__name__}: {exc}"[:300]
    if p.returncode != 0:
        return (f"nvidia-smi --query-compute-apps exited {p.returncode}: " + (p.stdout or p.stderr).strip())[:300]
    out = []
    for line in p.stdout.splitlines():
        f = [c.strip() for c in line.split(",")]
        if len(f) == 4:
            out.append({"gpu_uuid": f[0], "pid": int(f[1]) if f[1].isdigit() else f[1], "name": f[2],
                        "used_mib": int(f[3]) if f[3].isdigit() else None,
                        "here": f[1].isdigit() and Path(f"/proc/{f[1]}").is_dir()})
    return out[:50]


def _uuid(torch, d: int) -> str | None:
    """GPU d's uuid as nvidia-smi writes it (`GPU-...`), to name its holders; None where this torch does not say."""
    try:
        u = getattr(torch.cuda.get_device_properties(d), "uuid", None)
    except Exception:  # noqa: BLE001 - only a name for the record
        return None
    return None if u is None else (str(u) if str(u).startswith("GPU-") else f"GPU-{u}")


def _smi_free() -> dict[str, int]:
    """Each GPU's free memory in MiB as nvidia-smi reads it, by lower-case uuid; empty where nvidia-smi cannot say.
    nvidia-smi needs no CUDA context, so it answers for a GPU on which this process could not create one."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    try:
        p = subprocess.run([smi, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = [[c.strip() for c in line.split(",")] for line in p.stdout.splitlines()] if p.returncode == 0 else []
    return {r[0].lower(): int(r[1]) for r in rows if len(r) == 2 and r[1].isdigit()}


def _skipped(torch, entries: list[dict]) -> list[dict]:
    """The skipped GPUs, each with nvidia-smi's free MiB (`smi_free_mib`) and its holders: the processes nvidia-smi
    lists on its uuid, or every process nvidia-smi lists (`holders_of`: "every GPU") where its uuid is not known, or why
    nvidia-smi could not say."""
    holders, free = gpu_holders(), _smi_free()
    for e in entries:
        e["uuid"] = _uuid(torch, e["device"])
        if e["uuid"] and e["uuid"].lower() in free:
            e["smi_free_mib"] = free[e["uuid"].lower()]
        if isinstance(holders, str):
            e["holders"] = holders
        elif e["uuid"]:
            e["holders"] = [h for h in holders if h["gpu_uuid"].lower() == e["uuid"].lower()]
        else:
            e["holders"], e["holders_of"] = holders, "every GPU"
    return entries


def _gpus(iters: int) -> tuple[list[dict], list[dict] | str, list[dict]]:
    """The GPU part: each visible GPU's TFLOP/s, nvidia-smi's rows, and the GPUs skipped.

    Each GPU is prepared on its own. A GPU with less free memory than `GPU_NEED_MIB`, read once this process's CUDA
    context is on it, is skipped without an allocation; so is one whose context or operands could not be had, with the
    error. Both the allocator's `OutOfMemoryError` and a bare `CUDA error: out of memory` raised by a CUDA call
    outside the allocator are caught here. The GPUs left are read as before, all at once."""
    import torch  # noqa: PLC0415 - the engine's interpreter has it; importing this package does not need it

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA device is visible")
    work, skipped = [], []
    for d in range(n):
        entry: dict = {"device": d, "need_mib": GPU_NEED_MIB}
        a = b = c = None
        try:
            entry["free_mib"] = int(torch.cuda.mem_get_info(d)[0] >> 20)
            if entry["free_mib"] < GPU_NEED_MIB:
                skipped.append(entry)
                continue
            with torch.cuda.device(d):
                a = torch.randn(GPU_BATCH, GPU_N, GPU_N, device=f"cuda:{d}", dtype=torch.bfloat16)
                b = torch.randn(GPU_BATCH, GPU_N, GPU_N, device=f"cuda:{d}", dtype=torch.bfloat16)
                c = torch.empty_like(a)
                for _ in range(GPU_WARMUP):
                    torch.bmm(a, b, out=c)
            work.append((d, a, b, c))
        except Exception as exc:  # noqa: BLE001 - this GPU is skipped and says why; the others are still read
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
            skipped.append(entry)
            a = b = c = None
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
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
    return out, smi, _skipped(torch, skipped) if skipped else []


def read(gpu_iters: int = GPU_ITERS_DEFAULT, cpu_iters: int = CPU_ITERS_DEFAULT) -> dict:
    """One reading: {"at", "ok", "cpu", "gpus", "smi"}, "gpus_skipped" when a GPU was not probed, and
    "error" when a part could not be read. `ok` when every part was read and no GPU skipped. A reading that is not ok
    is judged on what it did read (`quantities`)."""
    out: dict = {"at": _utcnow(), "ok": True, "gpu_iters": gpu_iters, "host": host_identity()}
    procs0, t0 = _proc_cpu(), time.monotonic()
    try:
        out["cpu"] = _cpu(cpu_iters)
    except Exception as exc:  # noqa: BLE001 - a part that fails is not judged and says why
        out["ok"] = False
        out["error"] = f"cpu: {type(exc).__name__}: {exc}"[:300]
    try:
        out["gpus"], out["smi"], skipped = _gpus(gpu_iters)
        if skipped:
            out["ok"] = False
            out["gpus_skipped"] = skipped
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = (out.get("error", "") + f" gpu: {type(exc).__name__}: {exc}")[:300].strip()
    try:
        out["top_cpu"] = top_consumers(procs0, _proc_cpu(), time.monotonic() - t0)
    except Exception:  # noqa: BLE001 - a diagnostic
        pass
    return out


def describe(reading: dict) -> str:
    """What a reading read, in a few words for the node's log."""
    if reading.get("ok"):
        return "ok"
    skipped = [f"gpu{g.get('device')} ({g.get('free_mib', g.get('smi_free_mib', '?'))} MiB free"
               f"{', ' + g['error'] if g.get('error') else ''})" for g in reading.get("gpus_skipped") or []]
    why = "; ".join(x for x in (reading.get("error"), "skipped " + ", ".join(skipped) if skipped else "") if x)
    q = quantities(reading)
    return f"read in part, {', '.join(sorted(q))} ({why})"[:400] if q else f"unavailable ({why})"[:400]


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


def compare(first: dict | None, reading: dict | None, tolerance: float, cpu_tolerance: float | None = None) -> dict:
    """One reading against a baseline reading -- the trial's first, or the running median `baseline()` returns -- as
    {"valid": True | False | None, "worst", "ratio", "ratios"}. False when any GPU's TFLOP/s is below (1 - tolerance) of
    its baseline, or the CPU's rate below (1 - cpu_tolerance) of its own (cpu_tolerance defaults to tolerance); None when
    there is nothing to judge by (either reading unavailable, or no quantity in common). `worst` and `ratio` name the
    quantity furthest below its own threshold.

    Each quantity either reading has is judged where both have it, whether or not
    the reading was read whole; `unjudged` names the quantities either has that were not judged."""
    base, now = quantities(first), quantities(reading)
    if not base:
        return {"valid": None, "why": "no first reading"}
    if not now:
        return {"valid": None, "why": "this reading is unavailable"}
    cpu_tolerance = tolerance if cpu_tolerance is None else cpu_tolerance
    ratios = {k: v / base[k] for k, v in now.items() if base.get(k, 0) > 0}
    unjudged = sorted((now.keys() | base.keys()) - ratios.keys())
    if not ratios:
        return {"valid": None, "why": "no quantity in common with the first reading", "unjudged": unjudged}
    floor = {k: 1.0 - (cpu_tolerance if k == "cpu" else tolerance) for k in ratios}
    worst = min(ratios, key=lambda k: ratios[k] - floor[k])
    out = {"valid": all(ratios[k] >= floor[k] for k in ratios), "worst": worst, "ratio": round(ratios[worst], 4),
           "ratios": {k: round(v, 4) for k, v in sorted(ratios.items())}}
    if unjudged:
        out["unjudged"] = unjudged
    return out


def baseline(readings: list[dict]) -> dict | None:
    """The median, per quantity, of the readings given -- each GPU's TFLOP/s by
    device and the CPU's rate -- as one reading `compare` can judge against. None when none read anything.
    A whole reading can run slow with the draws beside it normal, and that reading can be a trial's first; a
    median of the readings so far is not moved by one such reading. It is the upper median: of
    an even count, the higher middle reading, so a slow second reading is judged against the first
    and not against the mean of the two.

    Each quantity's median is over the readings that read it, a reading read in
    part included; `of` is how many."""
    import statistics  # noqa: PLC0415

    per_gpu: dict = {}
    cpu: list[float] = []
    of = 0
    for r in readings:
        if not quantities(r):
            continue
        of += 1
        for g in r.get("gpus") or []:
            if _finite(g.get("tflops")):
                per_gpu.setdefault(g.get("device"), []).append(float(g["tflops"]))
        if _finite((r.get("cpu") or {}).get("rate")):
            cpu.append(float(r["cpu"]["rate"]))
    if not of:
        return None
    out: dict = {"ok": True, "baseline_of": of,
                 "gpus": [{"device": d, "tflops": statistics.median_high(v), "of": len(v)} for d, v in per_gpu.items()]}
    if cpu:
        out["cpu"] = {"rate": statistics.median_high(cpu), "of": len(cpu)}
    return out


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
