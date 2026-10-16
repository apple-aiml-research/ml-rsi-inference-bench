#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""This task's process and GPU hygiene for the reference launch and the deadline runner.

Six functions, carried unchanged from the benchmark's shared `_paired_helper.py`:

  cleanup_stale_ipc           reap leaked shared-memory segments and semaphores before a launch
  spawn_with_retry            the reference launch, retried once after an IPC reap if it crashes on start
  _gpu_pids_via_proc          the processes holding the GPUs, read from /proc
  kill_stray_gpu_processes    stop what still holds the GPUs after a teardown
  wait_for_gpu_free_memory    wait until the devices report their memory free
  teardown_and_confirm_free   stop a server's process tree and confirm the GPUs are free

The rest of that module -- worktrees, build artefacts, rebuild and binary-swap policies, benchmark
statistics -- is outside this task's launch and cleanup calls and is not carried.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable


# ── IPC cleanup ─────────────────────────────────────────────────────────────

def cleanup_stale_ipc(*, verbose: bool = True) -> dict[str, int]:
    """Reap leaked vllm IPC resources from prior spawns.

    Returns a dict with counts of {shm_files, sem_ids, killed_pids} for
    diagnostic logging. All operations are best-effort: missing tools or
    permission errors are caught and logged, never raised.

    Cleans:
      * `/dev/shm/vllm_*` files (POSIX shared memory backing files)
      * Orphan POSIX semaphores under `/dev/shm/sem.*` matching vllm
      * Stray vllm worker processes via `pkill -9 -f 'vllm.*serve'`
      * SysV IPC orphans owned by current user (ipcrm best-effort)

    Diagnostic env vars:
      * INFB_DIAG_SKIP_CLEANUP=1 — no-op (skip the pre-spawn cleanup entirely,
        to test whether the cleanup is itself the trigger of a spawn failure)
      * INFB_DIAG_CLEANUP_SIGTERM=1 — SIGTERM + 5s grace + SIGKILL
        stragglers (give the CUDA driver a chance to release device state on
        a normal exit before a forced kill)
      * INFB_DIAG_CLEANUP_GRACE_SEC=N — after the kill, sleep N seconds
        and poll nvidia-smi until all compute apps are gone, so the next vllm
        spawn does not race with driver state teardown
    """
    counts = {"shm_files": 0, "sem_ids": 0, "killed_pids": 0}

    # ── diagnostic: skip cleanup entirely ────────────────────────────
    if os.environ.get("INFB_DIAG_SKIP_CLEANUP") == "1":
        if verbose:
            print("[paired_helper] cleanup_stale_ipc: SKIPPED "
                  "(INFB_DIAG_SKIP_CLEANUP=1)", flush=True)
        return counts

    # /dev/shm/vllm_* (POSIX shm backing) + sem.* (POSIX named sems)
    shm_dir = Path("/dev/shm")
    if shm_dir.exists():
        for child in shm_dir.iterdir():
            try:
                name = child.name
                if (name.startswith("vllm_") or
                        name.startswith("psm_") or  # PyTorch shm
                        name.startswith("nccl-")):
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                        counts["shm_files"] += 1
                    elif child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                        counts["shm_files"] += 1
                elif name.startswith("sem.") and (
                        "vllm" in name.lower() or "torch" in name.lower()
                        or "nccl" in name.lower()):
                    child.unlink(missing_ok=True)
                    counts["sem_ids"] += 1
            except OSError:
                pass

    # Stray vllm worker processes (zombies still holding GPU mem).
    use_sigterm = os.environ.get("INFB_DIAG_CLEANUP_SIGTERM") == "1"
    try:
        r = subprocess.run(
            ["pgrep", "-f", "vllm.*serve"],
            capture_output=True, text=True, timeout=10,
        )
        pids = [p for p in r.stdout.split() if p.isdigit()]
        if pids:
            # ── diagnostic: graceful SIGTERM with grace then SIGKILL ─────────
            if use_sigterm:
                subprocess.run(
                    ["kill", "-TERM", *pids],
                    capture_output=True, timeout=10, check=False,
                )
                time.sleep(5.0)
                subprocess.run(
                    ["kill", "-9", *pids],
                    capture_output=True, timeout=10, check=False,
                )
            else:
                subprocess.run(
                    ["kill", "-9", *pids],
                    capture_output=True, timeout=10, check=False,
                )
            counts["killed_pids"] = len(pids)
    except (OSError, subprocess.SubprocessError):
        pass

    # ── diagnostic: post-cleanup driver settle ───────────────────────
    grace_str = os.environ.get("INFB_DIAG_CLEANUP_GRACE_SEC", "")
    if grace_str:
        try:
            grace_sec = float(grace_str)
            time.sleep(grace_sec)
            # Best-effort: poll nvidia-smi for compute apps still
            # holding GPU. Once empty (or after 30s cap), proceed.
            nvsmi = shutil.which("nvidia-smi")
            if nvsmi:
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    r = subprocess.run(
                        [nvsmi, "--query-compute-apps=pid",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10,
                        check=False,
                    )
                    out = (r.stdout or "").strip()
                    if not out or all(not ln.strip() for ln in out.splitlines()):
                        if verbose:
                            print(f"[paired_helper] cleanup_stale_ipc: "
                                  f"GPU compute apps drained after "
                                  f"{grace_sec:.1f}+ s grace", flush=True)
                        break
                    time.sleep(2.0)
        except (ValueError, OSError, subprocess.SubprocessError):
            pass

    # SysV IPC owned by current uid (ipcs -s; ipcrm -s id). Best-effort.
    try:
        uid = os.getuid()
        r = subprocess.run(
            ["ipcs", "-s"], capture_output=True, text=True, timeout=10,
            check=False,
        )
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0].startswith("0x"):
                try:
                    semid = int(parts[1])
                    owner = parts[2]
                except (ValueError, IndexError):
                    continue
                # Match by uid (numeric) or username; cheap heuristic.
                if owner == str(uid) or owner == os.environ.get("USER", ""):
                    subprocess.run(
                        ["ipcrm", "-s", str(semid)],
                        capture_output=True, timeout=5, check=False,
                    )
                    counts["sem_ids"] += 1
    except (OSError, subprocess.SubprocessError):
        pass

    if verbose and any(counts.values()):
        print(f"[paired_helper] cleanup_stale_ipc: "
              f"shm={counts['shm_files']} sems={counts['sem_ids']} "
              f"killed_pids={counts['killed_pids']}", flush=True)
    return counts


# ── Spawn-with-retry ────────────────────────────────────────────────────────

def spawn_with_retry(
    spawn_fn: Callable[[int], int],
    *,
    max_retries: int = 1,
    cleanup_between: bool = True,
    grace_sec: float = 5.0,
    label: str = "spawn",
) -> int:
    """Call `spawn_fn(attempt_idx)` up to `max_retries+1` times.

    `spawn_fn` should return an rc-style int (0 = success, nonzero =
    failure). Between attempts, optionally run `cleanup_stale_ipc()` and
    sleep `grace_sec` seconds to let the kernel reap fds before the next
    fork.

    Returns the rc of the first successful attempt, or the last failed rc.

    Designed for the verifier's first-spawn-fail recovery path: a long run
    leaks a few semaphores and one shm segment per vllm spawn, so after many
    spawns the first verifier spawn often dies in `WorkerProc.wait_for_ready`
    because fork cannot allocate IPC. After a cleanup pass + 5s grace, the
    second spawn typically succeeds.
    """
    last_rc = -1
    for attempt in range(max_retries + 1):
        if attempt > 0:
            if cleanup_between:
                cleanup_stale_ipc(verbose=True)
            print(f"[paired_helper] {label}: attempt {attempt+1}/"
                  f"{max_retries+1} after {grace_sec}s grace",
                  flush=True)
            time.sleep(grace_sec)
        rc = spawn_fn(attempt)
        if rc == 0:
            if attempt > 0:
                print(f"[paired_helper] {label}: succeeded on retry "
                      f"{attempt}", flush=True)
            return rc
        last_rc = rc
        print(f"[paired_helper] {label}: attempt {attempt+1} failed "
              f"rc={rc}", flush=True)
    return last_rc


# ── GPU process cleanup helpers ─────────────────────────────────────────────
# Centralized here so a fix to GPU-cleanup logic propagates with one edit.

def _gpu_pids_via_proc() -> list[int]:
    """Fallback when nvidia-smi is missing: scan /proc/<pid>/fd for handles
    on /dev/nvidia*. Returns the list of pids holding any nvidia device.
    """
    pids: list[int] = []
    proc = Path("/proc")
    if not proc.exists():
        return pids
    self_pid = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        fd_dir = entry / "fd"
        if not fd_dir.exists():
            continue
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("/dev/nvidia"):
                    pids.append(pid)
                    break
        except (PermissionError, FileNotFoundError):
            continue
    return pids


def kill_stray_gpu_processes(cuda_device: str = "0", grace_sec: float = 5.0,
                             **kwargs) -> None:
    """Kill any processes still holding GPU memory on the given CUDA device.

    Accepts either `cuda_device` (singular, single-GPU tasks) or
    `cuda_devices` (plural, multi-GPU tasks; comma-separated string) as
    kwarg. When `cuda_devices` is given, the first device in the list is
    used for the nvidia-smi query (the /proc fallback being device-agnostic
    anyway).

    CALLERS ON MULTI-GPU TASKS MUST THEREFORE LOOP, one call per device: this
    function queries a single device, so a straggler on device 1..N-1 survives
    and OOMs the next boot exactly like one on device 0. Passing the whole list
    is not a substitute either way — via `cuda_devices` it is truncated to its
    first element three lines below, and via `cuda_device` it reaches
    `nvidia-smi --id=0,1,...` which does accept a list (exit 0, the union of
    the listed devices' compute apps) but returns ZERO rows at exit 0 if any
    one entry is stale or out of range — which reads here as "no stray pids"
    and skips the /proc fallback entirely.

    vllm V1 spawns EngineCore in a new session (different process group from
    the APIServer). A SIGKILL on the APIServer's pgid misses the EngineCore,
    leaving it alive and holding GPU memory. This prevents the next vllm serve
    from starting.

    Primary mechanism: nvidia-smi --query-compute-apps=pid (per-device).
    Fallback when nvidia-smi is missing (e.g. verifier container without
    nvidia-container-runtime injection): scan /proc/<pid>/fd for /dev/nvidia*
    handles. The fallback is device-agnostic — it kills ANY process holding a
    GPU device — but in a single-GPU-per-trial setup that's correct.
    """
    import signal as _signal

    # Accept plural-form kwarg too (multi-GPU tasks call as
    # `kill_stray_gpu_processes(cuda_devices="0,1,2,3")`). Passing the singular name raises
    # TypeError at the first call site that uses the plural one.
    if "cuda_devices" in kwargs:
        cd = kwargs.pop("cuda_devices")
        if cd:
            cuda_device = str(cd).split(",")[0].strip() or cuda_device
    if kwargs:
        raise TypeError(
            f"kill_stray_gpu_processes: unexpected kwargs {sorted(kwargs.keys())}"
        )

    nvidia_smi = shutil.which("nvidia-smi") or (
        "/usr/bin/nvidia-smi" if Path("/usr/bin/nvidia-smi").exists() else None
    )

    pids: list[int] = []
    method = ""
    self_pid = os.getpid()
    if nvidia_smi is not None:
        try:
            r = subprocess.run(
                [nvidia_smi, f"--id={cuda_device}",
                 "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                pids = [
                    int(p.strip()) for p in r.stdout.splitlines()
                    if p.strip().isdigit() and int(p.strip()) != self_pid
                ]
                method = "nvidia-smi"
        except Exception as e:
            print(f"  [gpu-cleanup] nvidia-smi failed ({e}); falling back to /proc",
                  flush=True)

    if not method:
        try:
            pids = [p for p in _gpu_pids_via_proc() if p != self_pid]
            method = "/proc fd-scan"
        except Exception as e:
            print(f"  [gpu-cleanup] /proc fallback failed: {e}", flush=True)
            return

    if not pids:
        return
    print(f"  [gpu-cleanup] killing {len(pids)} stray GPU pid(s) via {method}: {pids}",
          flush=True)
    for pid in pids:
        try:
            os.kill(pid, _signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(grace_sec)
    if nvidia_smi is not None and method == "nvidia-smi":
        try:
            r2 = subprocess.run(
                [nvidia_smi, f"--id={cuda_device}",
                 "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            still_alive = [p.strip() for p in r2.stdout.splitlines() if p.strip().isdigit()]
            if still_alive:
                print(f"  [gpu-cleanup] WARNING: {len(still_alive)} pid(s) still alive "
                      f"after cleanup", flush=True)
            else:
                print(f"  [gpu-cleanup] GPU memory freed", flush=True)
        except Exception:
            pass
    else:
        try:
            still = _gpu_pids_via_proc()
            if still:
                print(f"  [gpu-cleanup] WARNING: {len(still)} pid(s) still hold "
                      f"/dev/nvidia* after cleanup", flush=True)
            else:
                print(f"  [gpu-cleanup] GPU memory freed", flush=True)
        except Exception:
            pass


def wait_for_gpu_free_memory(cuda_device: str = "0",
                             required_gib: float = 60.0,
                             timeout_sec: float = 90.0,
                             poll_sec: float = 3.0,
                             gpu_memory_utilization: float | None = None,
                             **kwargs) -> float:
    """Block until EVERY target GPU has enough free memory to spawn a serve.

    Accepts both `cuda_device` (singular) and `cuda_devices` (plural,
    comma-separated string), polls ALL listed devices and only returns once
    the WORST device clears the threshold (a TP>1 serve OOMs if ANY rank's
    GPU is still occupied by a not-yet-reaped prior worker).

    Threshold is per-device: if `gpu_memory_utilization` is given (the value
    the serve will launch with, e.g. 0.9), the requirement is derived as
    `gpu_memory_utilization * memory.total` for that device — so the guard
    gates on what the serve ACTUALLY needs (0.9*79≈71 GiB) instead of a
    hardcoded 60 GiB that silently passes a GPU which vLLM's own precheck
    will then reject. Falls back to the flat `required_gib` when util is None.

    When the agent container / a prior serve is hard-killed (SIGKILL), its
    vLLM workers take 10-30s to actually die and release VRAM. Spawning in
    that window crashes with `ValueError: Free memory ... < desired`. This
    poll closes that race. Returns the worst-device free GiB once all clear;
    raises TimeoutError (with the offending device + deficit) otherwise.
    """
    devs: list[str] | None = None
    if "cuda_devices" in kwargs:
        cd = kwargs.pop("cuda_devices")
        if cd:
            devs = [d.strip() for d in str(cd).split(",") if d.strip()]
    if kwargs:
        raise TypeError(
            f"wait_for_gpu_free_memory: unexpected kwargs {sorted(kwargs.keys())}"
        )
    if not devs:
        devs = [str(cuda_device)]
    nvidia_smi = shutil.which("nvidia-smi") or (
        "/usr/bin/nvidia-smi" if Path("/usr/bin/nvidia-smi").exists() else None
    )
    if nvidia_smi is None:
        print("[gpu-wait] nvidia-smi unavailable; skipping pre-flight wait", flush=True)
        return float("inf")

    def _free_total(dev: str) -> tuple[float, float]:
        r = subprocess.run(
            [nvidia_smi, f"--id={dev}",
             "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return (0.0, 0.0)
        try:
            free_mb, total_mb = (int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(","))
            return (free_mb / 1024.0, total_mb / 1024.0)
        except (TypeError, ValueError, IndexError):
            return (0.0, 0.0)

    deadline = time.monotonic() + timeout_sec
    polls = 0
    while True:
        polls += 1
        worst_free = float("inf")
        worst_dev = devs[0]
        worst_need = required_gib
        all_ok = True
        for dev in devs:
            free, total = _free_total(dev)
            need = (gpu_memory_utilization * total) if (gpu_memory_utilization and total) else required_gib
            if free - need < worst_free - worst_need:
                worst_free, worst_dev, worst_need = free, dev, need
            if free < need:
                all_ok = False
        if all_ok:
            if polls > 1:
                print(f"[gpu-wait] all {len(devs)} dev(s) ready (worst gpu={worst_dev} "
                      f"{worst_free:.1f} GiB free >= {worst_need:.1f}); spawning serve", flush=True)
            return worst_free
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"[gpu-wait] gpu={worst_dev} only {worst_free:.1f} / {worst_need:.1f} GiB free "
                f"after {timeout_sec}s (devices={devs}); spawn would OOM"
            )
        print(f"[gpu-wait] gpu={worst_dev} free={worst_free:.1f} GiB (need {worst_need:.1f}); "
              f"poll {polls}, sleeping {poll_sec}s ...", flush=True)
        time.sleep(poll_sec)


def teardown_and_confirm_free(proc,
                              cuda_devices: str = "0",
                              gpu_memory_utilization: float | None = None,
                              required_gib: float = 60.0,
                              timeout_sec: float = 240.0,
                              term_grace_sec: float = 8.0) -> bool:
    """Synchronous serve teardown that does NOT return until the GPU(s) are
    actually free — the fix for the "next serve OOMs because the prior
    serve's VRAM wasn't reclaimed yet" race.

    Steps: (1) graceful SIGTERM the serve's process GROUP so vLLM tears down
    its CUDA context cleanly (faster reclaim than a raw SIGKILL); (2) after a
    grace window, SIGKILL any survivor; (3) SIGKILL stray GPU-resident workers
    on EVERY device (vLLM V1 EngineCore workers run in a SEPARATE session and
    escape killpg — reaping by GPU residency is the only reliable path);
    (4) POLL nvidia-smi until every device has util-derived free memory back
    (or `timeout_sec`). Returns True if all GPUs confirmed free, False on
    timeout (caller should then treat the next serve as infra-risky / retry).
    """
    import signal as _signal
    devs = [d.strip() for d in str(cuda_devices).split(",") if d.strip()] or ["0"]
    if proc is not None and proc.poll() is None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, _signal.SIGTERM)          # (1) graceful
        except (ProcessLookupError, PermissionError):
            pgid = None
        try:
            proc.wait(timeout=term_grace_sec)
        except subprocess.TimeoutExpired:
            try:                                       # (2) force
                if pgid is not None:
                    os.killpg(pgid, _signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            try:
                getattr(proc, "_log_fp", None) and proc._log_fp.close()
            except Exception:
                pass
    # (3) reap stray GPU-resident workers (EngineCore escapes killpg)
    for dev in devs:
        try:
            kill_stray_gpu_processes(cuda_device=dev, grace_sec=3.0)
        except Exception:
            pass
    # (4) block until the driver has actually reclaimed VRAM on every device
    try:
        wait_for_gpu_free_memory(
            cuda_devices=",".join(devs),
            gpu_memory_utilization=gpu_memory_utilization,
            required_gib=required_gib,
            timeout_sec=timeout_sec,
            poll_sec=3.0,
        )
        return True
    except TimeoutError as e:
        print(f"[teardown] WARN: GPU not confirmed free after teardown: {e}", flush=True)
        return False
