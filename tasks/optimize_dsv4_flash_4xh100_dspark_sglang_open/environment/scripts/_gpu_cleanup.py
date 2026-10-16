#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared GPU cleanup helper for agent-side scripts."""
from __future__ import annotations
import os
import signal
import subprocess
import time
from pathlib import Path


def _gpu_pids_via_proc() -> list[int]:
    """Fallback when nvidia-smi is missing: scan /proc/<pid>/fd for handles
    on /dev/nvidia*.  Returns the list of pids holding any nvidia device.
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
                             quiet: bool = False) -> int:
    """Kill any processes still holding GPU memory.  Returns count killed."""
    import shutil
    nvidia_smi = shutil.which("nvidia-smi") or (
        "/usr/bin/nvidia-smi" if Path("/usr/bin/nvidia-smi").exists() else None
    )

    pids: list[int] = []
    method = ""
    if nvidia_smi is not None:
        try:
            r = subprocess.run(
                [nvidia_smi, f"--id={cuda_device}",
                 "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                pids = [int(p.strip()) for p in r.stdout.splitlines() if p.strip().isdigit()]
                method = "nvidia-smi"
        except Exception:
            pass

    if not method:
        try:
            pids = _gpu_pids_via_proc()
            method = "/proc fd-scan"
        except Exception:
            return 0

    if not pids:
        return 0
    if not quiet:
        print(f"[gpu-cleanup] killing {len(pids)} stray GPU pid(s) via {method}: {pids}",
              flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(grace_sec)
    return len(pids)
