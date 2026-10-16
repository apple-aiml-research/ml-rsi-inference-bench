#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared helper for PAIRED-mode benchmark robustness."""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator



_BUILD_ARTIFACT_GLOBS: tuple[str, ...] = (
    "vllm/**/*.so",
    "vllm/**/*.pyd",
    "vllm/_C*.so",
    "vllm/_moe_C*.so",
    "vllm/_custom_ops*.so",
)

_BUILD_FILE_OVERLAYS: tuple[str, ...] = (
    "vllm/_version.py",
    "vllm/vllm_flash_attn/__init__.py",
    "vllm/vllm_flash_attn/flash_attn_interface.py",
)

# Directory overlays from prepare_target_env.sh — they sit in the venv's vllm
# only; the source tree under inference_base does not ship them. A fresh
# worktree would lack them, so vllm's import time falls back to different code
# paths than the candidate side, which produces a SYSTEMATIC speedup difference
# between the two sides on identical code.
#
# Mirroring strategy: in the source repo these are SYMLINKS into the venv
# (`ln -sfn $venv_vllm/<sub> $repo/vllm/<sub>`). On the worktree we
# replicate the same symlinks — it's faster than copying tens of MB of
# CUTLASS / DeepGEMM Python files, and matches the byte-for-byte semantic
# of the candidate side (both sides resolve to identical venv files).
_BUILD_DIR_OVERLAYS: tuple[str, ...] = (
    "vllm/third_party/deep_gemm",         # DeepGEMM Python — MoE/GEMM kernels
    "vllm/third_party/flashmla",          # FlashMLA — sparse MLA backend (FP8 lazy-import chain)
    "vllm/third_party/fmha_sm100",        # SM100 (B200) fused-MHA — block-sparse attention / DSA import at serve init
    "vllm/vllm_flash_attn/cute",          # CUTLASS Python DSL — FA4 backend
    "vllm/vllm_flash_attn/layers",        # multimodal rotary impls
    "vllm/vllm_flash_attn/ops",           # auxiliary ops
)

_SURFACE_TO_OVERLAY: dict[str, str] = {
    "cute": "vllm/vllm_flash_attn/cute",
    "deep_gemm": "vllm/third_party/deep_gemm",
}


def _git(repo: Path, *args: str, check: bool = True,
         capture: bool = True) -> subprocess.CompletedProcess:
    """Run `git -C repo args...` with sensible defaults."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=capture, text=True, check=check,
    )


def _path_is_outside(p: Path, root: Path) -> bool:
    """True when `p` does not live under `root`."""
    try:
        p.resolve().relative_to(root.resolve())
        return False
    except (ValueError, OSError):
        return True


def rsync_build_artifacts(src_repo: Path, dst_worktree: Path,
                          skip_overlays: frozenset[str] = frozenset()) -> int:
    """Mirror prepare_target_env.sh's overlays from src_repo to dst_worktree."""
    if not src_repo.exists() or not dst_worktree.exists():
        return 0
    n = 0

    _csrc_open = False
    try:
        _csrc_open = bool(get_edit_policy(src_repo).get("csrc"))
    except Exception as e:
        print(f"[paired_helper] WARN: edit-policy read failed; leaving the .so mirror "
              f"untouched: {e}", flush=True)
    rebuilt_rel: set[str] = set()
    install_dir = src_repo / ".csredit" / "install"
    if install_dir.is_dir() and not _csrc_open:
        print("[paired_helper] a csrc rebuild exists but the csrc surface is CLOSED on "
              "this task -> mirroring it to pristine unchanged, so the out-of-policy "
              "edit cancels instead of being scored", flush=True)
    if _csrc_open and install_dir.is_dir():
        for p in install_dir.rglob("*.so"):
            rebuilt_rel.add(str(p.relative_to(install_dir)))

    baseline_so: dict[str, str] = {}
    try:
        _snap = src_repo / ".csredit" / "venv_symlinks.json"
        if _snap.is_file():
            baseline_so = {f"vllm/{k}": v for k, v in json.loads(_snap.read_text()).items()}
    except Exception as e:
        print(f"[paired_helper] WARN: could not read venv_symlinks.json: {e}", flush=True)

    venv_vllm: Path | None = None
    if rebuilt_rel:
        for probe in (src_repo / "vllm").glob("*.so"):
            if probe.is_symlink():
                venv_vllm = Path(os.readlink(probe)).parent
                break
        # Deliberately NOT falling back to a path out of venv_symlinks.json here. That
        # file is agent-writable (swap_so_to_built writes it only when ABSENT, so a
        # planted one survives), and deriving the TRUST ROOT from it would defeat the
        # point of having one.

    def _baseline_for(rel: Path) -> Path | None:
        """The venv copy of `rel`, or None when this target was not rebuilt.

        `rebuilt_rel` is relative to `.csredit/install`, whose layout mirrors the
        repo's (`install/vllm/_moe_C.abi3.so` <-> `repo/vllm/_moe_C.abi3.so`), so the
        two keys compare directly. Prefixing again would silently match nothing.

        Two sources, and the ORDER is a trust decision that matches
        `swap_so_to_venv`: the surviving sibling symlink is read from the live
        filesystem, `venv_symlinks.json` is a file the agent can write, so the derived
        path wins and the record is only a fallback. A disagreement is announced —
        silently preferring the right one would hide an attempt.

        Neither source is a real trust root (a symlink can be re-pointed too), so the
        clearly-wrong case is refused outright: a baseline inside `src_repo` would hand
        the pristine leg something from the candidate's own tree, which is the exact
        thing this substitution exists to prevent.
        """
        if str(rel) not in rebuilt_rel:
            return None
        recorded = baseline_so.get(str(rel))
        derived = (venv_vllm / rel.relative_to("vllm")) if venv_vllm is not None else None
        if recorded and derived is not None and Path(recorded) != derived:
            where = ("INSIDE the candidate repo"
                     if not _path_is_outside(Path(recorded), src_repo) else "elsewhere")
            print(f"[paired_helper] NOTE: venv_symlinks.json records {rel} at "
                  f"{recorded} ({where}); using the derived {derived}", flush=True)
        for cand, src in ((derived, "sibling"), (Path(recorded) if recorded else None,
                                                 "recorded")):
            if cand is None or not cand.is_file():
                continue
            if not _path_is_outside(cand, src_repo):
                print(f"[paired_helper] REFUSED: the {src} baseline for {rel} is {cand}, "
                      f"INSIDE the candidate repo -- the pristine leg would load the "
                      f"candidate's own kernel", flush=True)
                continue
            print(f"[paired_helper] pristine baseline for {rel}: {cand} ({src})",
                  flush=True)
            return cand
        print(f"[paired_helper] WARN: {rel} was rebuilt but its venv baseline could not "
              f"be located outside the candidate repo; the pristine will carry the "
              f"CANDIDATE's kernel and any csrc-only change cancels to ~1.0", flush=True)
        return None

    substituted: list[str] = []
    for pattern in _BUILD_ARTIFACT_GLOBS:
        for src in src_repo.glob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(src_repo)
            dst = dst_worktree / rel
            base = _baseline_for(rel)
            real_src = base if base is not None else src
            if base is not None:
                substituted.append(str(rel))
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(real_src, dst)
                n += 1
            except OSError as e:
                print(f"[paired_helper] WARN: failed to copy {rel}: {e}",
                      flush=True)
    if rebuilt_rel:
        print(f"[paired_helper] candidate carries a rebuild of {sorted(rebuilt_rel)}; "
              f"pristine took the venv baseline for {sorted(set(substituted))}",
              flush=True)

    for rel in _BUILD_FILE_OVERLAYS:
        src = src_repo / rel
        if not src.exists():
            continue
        dst = dst_worktree / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink():
                target = os.readlink(src)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(target, dst)
            else:
                shutil.copy2(src, dst)
            n += 1
        except OSError as e:
            print(f"[paired_helper] WARN: failed to mirror {rel}: {e}",
                  flush=True)

    missing_overlays: list[str] = []
    for rel in _BUILD_DIR_OVERLAYS:
        if rel in skip_overlays:
            continue
        src = src_repo / rel
        if not src.exists():
            missing_overlays.append(rel)
            continue
        dst = dst_worktree / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst, ignore_errors=True)
            if src.is_symlink():
                target = os.readlink(src)
                os.symlink(target, dst)
            else:
                os.symlink(src.resolve(), dst)
            n += 1
        except OSError as e:
            print(f"[paired_helper] WARN: failed to mirror {rel}: {e}",
                  flush=True)

    if missing_overlays:
        print(f"[paired_helper] WARN: {len(missing_overlays)} declared build-dir overlay(s) are ABSENT from "
              f"the source repo, so the pristine worktree will not have them either: "
              f"{missing_overlays}. If the engine imports one of these, its workers die with "
              f"`No module named ...` rather than saying the staging was incomplete.", flush=True)

    return n


def reap_gpus_before_pristine_engine(grace_sec: float = 2.0,
                                     required_gib: float = 60.0,
                                     timeout_sec: float = 120.0) -> None:
    """Kill GPU-resident strays on every visible device, then wait for VRAM.

    `teardown_and_confirm_free` covers the case where you own the process being
    torn down. This covers the other one: something else left VRAM held and an
    engine is about to be started on top of it.

    Kills on EVERY device before waiting on any of them. `kill_stray_gpu_processes`
    queries a single device -- its own docstring insists callers loop -- and
    interleaving kill/wait stalls on device 0 while device 7 is still held.

    Warn-and-continue on timeout: a slow drain is not worth failing a trial that
    may still fit.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
    devices = [d.strip() for d in visible.split(",") if d.strip()] or ["0"]
    print(f"[paired_helper] reaping GPUs {','.join(devices)} before the pristine engine "
          f"(require {required_gib:.0f} GiB, timeout {timeout_sec:.0f}s)", flush=True)
    for device in devices:
        kill_stray_gpu_processes(cuda_device=device, grace_sec=grace_sec)
    try:
        wait_for_gpu_free_memory(cuda_devices=",".join(devices),
                                 required_gib=required_gib,
                                 timeout_sec=timeout_sec)
    except TimeoutError as exc:
        print(f"[paired_helper] WARN: {exc}; starting the pristine engine anyway",
              flush=True)


@contextlib.contextmanager
def disposable_pristine_worktree(
    repo: Path,
    *,
    base_ref: str = "inference_base",
    workdir: Path,
    label: str = "_paired_pristine_worktree",
    copy_build_artifacts: bool = True,
    skip_overlays: frozenset[str] | None = None,
    reap_gpus: bool = True,
) -> Iterator[Path]:
    """Yield a path to a freshly-created git worktree at `base_ref`.

    The main repo is NEVER mutated. On exit (clean or exceptional) the
    worktree is removed via `git worktree remove --force`, with a
    belt-and-suspenders `shutil.rmtree` in case git left anything behind.

    `copy_build_artifacts=True` mirrors `vllm/**/*.so` from `repo` into the
    worktree post-create, which is what stops the pristine vllm from failing
    to import.

    `reap_gpus=True` (default) kills GPU-resident strays and waits for VRAM
    BEFORE handing the tree back. Every caller of this helper starts an engine
    from what it returns, and the first of them -- the correctness live
    reference -- is the first engine the verifier runs at all, after the agent
    has had the machine to itself: a `vllm serve` the agent left resident holds
    the memory, the agent's own per-benchmark cleanup has no next round to fire
    in, and nothing between the agent phase and the verifier reclaims a GPU. The
    reference then fails to boot and a patch that was never graded scores zero on
    correctness.

    Reaping HERE rather than at each call site is deliberate: this is the one
    place both pristine-engine paths funnel through, so the invariant holds
    without every task-local copy having to remember it. It runs before the
    worktree is even created, which also gives the cards the whole
    create-plus-mirror window to drain. On an already-free GPU it is a no-op.
    """
    if reap_gpus:
        reap_gpus_before_pristine_engine()

    if os.environ.get("INFB_DIAG_NO_WORKTREE") == "1":
        saved_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        dirty = _git(repo, "status", "--porcelain", check=False).stdout.strip()
        if dirty:
            raise RuntimeError(
                "[paired_helper] INFB_DIAG_NO_WORKTREE=1 refuses dirty tree: "
                f"{repo} (commit/stash first)"
            )
        print(f"[paired_helper] NO_WORKTREE: git reset --hard {base_ref}",
              flush=True)
        _git(repo, "reset", "--hard", base_ref)
        try:
            yield repo
        finally:
            print(f"[paired_helper] NO_WORKTREE: git reset --hard {saved_head[:12]}",
                  flush=True)
            try:
                _git(repo, "reset", "--hard", saved_head, check=False)
            except Exception as e:
                print(f"[paired_helper] WARN: git reset back failed: {e}",
                      flush=True)
        return

    worktree_dir = workdir / label

    _git(repo, "worktree", "remove", "--force", str(worktree_dir), check=False)
    _git(repo, "worktree", "prune", check=False)
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir, ignore_errors=True)

    _git(repo, "worktree", "add", "--detach", str(worktree_dir), base_ref)

    if copy_build_artifacts:
        if skip_overlays is None:
            try:
                _pol = get_edit_policy(repo)
                skip_overlays = frozenset(
                    _SURFACE_TO_OVERLAY[s] for s in _pol.get("editable_pkgs", [])
                    if s in _SURFACE_TO_OVERLAY
                )
            except Exception as e:
                print(f"[paired_helper] WARN: skip_overlays auto-derive failed: {e}",
                      flush=True)
                skip_overlays = frozenset()
        try:
            _openable = frozenset(_SURFACE_TO_OVERLAY.values())
            _tree_opened = frozenset(
                rel for rel in _BUILD_DIR_OVERLAYS
                if rel in _openable
                and (worktree_dir / rel).is_dir()
                and not (worktree_dir / rel).is_symlink()
            )
        except Exception as e:
            print(f"[paired_helper] WARN: tree-derived skip_overlays failed: {e}",
                  flush=True)
            _tree_opened = frozenset()
        if skip_overlays is None:
            skip_overlays = frozenset()
        _extra = _tree_opened - skip_overlays
        if _extra:
            print(f"[paired_helper] ground-truth: {sorted(_extra)} opened in base "
                  f"-> added to skip_overlays (drift-proof cancel-safety)", flush=True)
        skip_overlays = skip_overlays | _tree_opened
        n = rsync_build_artifacts(repo, worktree_dir, skip_overlays=skip_overlays)
        if n:
            print(f"[paired_helper] copied {n} build artifact(s) into "
                  f"{worktree_dir}", flush=True)

    try:
        yield worktree_dir
    finally:
        try:
            _git(repo, "worktree", "remove", "--force", str(worktree_dir),
                 check=False)
            _git(repo, "worktree", "prune", check=False)
        except Exception as e:
            print(f"[paired_helper] WARN: worktree cleanup failed: {e}",
                  flush=True)
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir, ignore_errors=True)



def cleanup_stale_ipc(*, verbose: bool = True) -> dict[str, int]:
    """Reap leaked vllm IPC resources from prior spawns."""
    counts = {"shm_files": 0, "sem_ids": 0, "killed_pids": 0}

    if os.environ.get("INFB_DIAG_SKIP_CLEANUP") == "1":
        if verbose:
            print("[paired_helper] cleanup_stale_ipc: SKIPPED "
                  "(INFB_DIAG_SKIP_CLEANUP=1)", flush=True)
        return counts

    shm_dir = Path("/dev/shm")
    if shm_dir.exists():
        for child in shm_dir.iterdir():
            try:
                name = child.name
                if (name.startswith("vllm_") or
                        name.startswith("psm_") or
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

    use_sigterm = os.environ.get("INFB_DIAG_CLEANUP_SIGTERM") == "1"
    try:
        r = subprocess.run(
            ["pgrep", "-f", "vllm.*serve"],
            capture_output=True, text=True, timeout=10,
        )
        pids = [p for p in r.stdout.split() if p.isdigit()]
        if pids:
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

    grace_str = os.environ.get("INFB_DIAG_CLEANUP_GRACE_SEC", "")
    if grace_str:
        try:
            grace_sec = float(grace_str)
            time.sleep(grace_sec)
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



def spawn_with_retry(
    spawn_fn: Callable[[int], int],
    *,
    max_retries: int = 1,
    cleanup_between: bool = True,
    grace_sec: float = 5.0,
    label: str = "spawn",
) -> int:
    """Call `spawn_fn(attempt_idx)` up to `max_retries+1` times."""
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
    used for the nvidia-smi query; the /proc fallback is device-agnostic
    anyway.

    CALLERS ON MULTI-GPU TASKS MUST THEREFORE LOOP, one call per device: this
    function queries a single device, so a straggler on device 1..N-1 survives
    and OOMs the next boot exactly like one on device 0. Passing the whole list
    is not a substitute either way — via `cuda_devices` it is truncated to its
    first element three lines below, and via `cuda_device` it reaches
    `nvidia-smi --id=0,1,...` which does accept a list (measured: exit 0, the
    union of the listed devices' compute apps) but returns ZERO rows at exit 0
    if any one entry is stale or out of range — which reads here as "no stray
    pids" and skips the /proc fallback entirely.

    vllm V1 spawns EngineCore in a new session (a different process group from
    the APIServer), so a SIGKILL on the APIServer's pgid misses the EngineCore,
    leaving it alive and holding GPU memory. That prevents the next `vllm serve`
    from starting.

    Primary mechanism: nvidia-smi --query-compute-apps=pid (per-device).
    Fallback when nvidia-smi is missing (e.g. a container without the NVIDIA
    container runtime's device injection): scan /proc/<pid>/fd for /dev/nvidia*
    handles. The fallback is device-agnostic — it kills ANY process holding a
    GPU device — but with one GPU per trial that is correct.
    """
    import signal as _signal

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
    comma-separated string). It polls ALL listed devices and only returns once
    the WORST device clears the threshold: a TP>1 serve OOMs if ANY rank's GPU
    is still occupied by a not-yet-reaped prior worker.

    The threshold is per-device. If `gpu_memory_utilization` is given (the value
    the serve will launch with, e.g. 0.9), the requirement is derived as
    `gpu_memory_utilization * memory.total` for that device — so the guard gates
    on what the serve ACTUALLY needs instead of a hardcoded 60 GiB that silently
    passes a GPU vLLM's own precheck will then reject. Falls back to the flat
    `required_gib` when util is None.

    When a prior serve is hard-killed (SIGKILL), its vLLM workers take tens of
    seconds to die and release VRAM; spawning in that window crashes with
    `ValueError: Free memory ... < desired`. This poll closes that race. Returns
    the worst-device free GiB once all clear; raises TimeoutError (with the
    offending device and its deficit) otherwise.
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

    Steps: (1) graceful SIGTERM to the serve's process GROUP so vLLM tears down
    its CUDA context cleanly (faster reclaim than a raw SIGKILL); (2) after a
    grace window, SIGKILL any survivor; (3) SIGKILL stray GPU-resident workers
    on EVERY device — vLLM V1 EngineCore workers run in a SEPARATE session and
    escape `killpg`, so reaping by GPU residency is the only reliable path;
    (4) POLL nvidia-smi until every device has its util-derived free memory back
    (or `timeout_sec`). Returns True if all GPUs are confirmed free, False on
    timeout, in which case the caller should treat the next serve as risky.
    """
    import signal as _signal
    devs = [d.strip() for d in str(cuda_devices).split(",") if d.strip()] or ["0"]
    if proc is not None and proc.poll() is None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, _signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pgid = None
        try:
            proc.wait(timeout=term_grace_sec)
        except subprocess.TimeoutExpired:
            try:
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
    for dev in devs:
        try:
            kill_stray_gpu_processes(cuda_device=dev, grace_sec=3.0)
        except Exception:
            pass
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




def trimmed_mean(values: list[float]) -> float:
    """Drop min+max before averaging when n>=3; plain mean below that.

    Short serving workloads carry a few percent of run-to-run spread with
    occasional freak outliers. A plain mean propagates an outlier and makes a
    1.05x speedup indistinguishable from noise; trimming min+max handles both
    tails without a stats library.

    AT n=3, "DROP MIN AND MAX" IS THE MEDIAN -- one value survives. That is the
    point and not an accident: the median of three is the cheapest estimator one
    outlier cannot move, and it costs no GPU time, which is why it shipped
    instead of raising the repetition count. At n>=5 this is the same estimator
    as before.

    WHAT IT DOES NOT FIX, stated so nobody reads a trimmed ratio as exactly 1.0
    on identical code: the candidate arm runs FIRST and COLD, so its bias is
    systematic rather than random, and trimming does not remove it. Fixing that
    needs a warm-up inside the measured pools, or alternating the arm order
    across repetitions.
    """
    if not values:
        return 0.0
    if len(values) >= 3:
        trimmed = sorted(values)[1:-1]
    else:
        trimmed = values
    return statistics.mean(trimmed)


def stats(values: list[float]) -> dict:
    """Return n, mean, trimmed_mean, stddev, ci95_halfwidth, min, max."""
    if not values:
        return {"n": 0, "mean": 0.0, "trimmed_mean": 0.0, "stddev": 0.0,
                "ci95_halfwidth": 0.0, "min": 0.0, "max": 0.0}
    # ONE IMPLEMENTATION, and this used to be two. `stats()` carried its own inline
    # copy of the trim rule with its own threshold, so changing the threshold in
    # `trimmed_mean()` alone would have made `stats()["trimmed_mean"]` disagree with
    # `trimmed_mean()` -- and `stats()` is the one whose value reaches metrics.json
    # and the agent's own bench readout. Delegating removes the possibility of them
    # diverging again rather than fixing one instance of it.
    mean = statistics.mean(values)
    tmean = trimmed_mean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * stddev / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "trimmed_mean": tmean,
        "stddev": stddev,
        "ci95_halfwidth": ci95,
        "min": min(values),
        "max": max(values),
    }


def geomean(values: list[float]) -> float:
    """Geometric mean of positive values; 0.0 if empty."""
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))



def extract_tps(bench_result: dict) -> float | None:
    """Extract a tokens-per-second scalar from a vllm bench JSON result."""
    for k in ("total_token_throughput", "tokens_per_second",
              "throughput_tokens_per_second"):
        if k in bench_result and isinstance(bench_result[k], (int, float)):
            return float(bench_result[k])
    return None


# ── Per-workload primary metric (InferenceBench A/B/C alignment) ────────────
# - prefill_heavy (Scenario A, 8192-token inputs) → TTFT, lower is better
# - decode_heavy  (Scenario B, 1024/8192)         → TPOT, lower is better
# - high_load     (Scenario C, 1024/1024)         → request throughput, higher is better
# - any other                                     → token throughput (legacy behavior)
PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "prefill_heavy": ("ttft_ms", "lower_is_better"),
    "decode_heavy":  ("tpot_ms", "lower_is_better"),
    "high_load":     ("request_throughput", "higher_is_better"),
    "dllm_concurrency_sweep": ("request_throughput", "higher_is_better"),
    "pd_multiturn":  ("pd_latency_itl_geomean_ms", "lower_is_better"),
    "mtp_longctx_8k": ("output_throughput", "higher_is_better"),
}


def extract_metrics_dict(bench_result: dict) -> dict[str, float]:
    """Extract every metric we care about from a vllm bench serve JSON."""
    out: dict[str, float] = {}
    pairs = (
        ("median_ttft_ms",       "ttft_ms"),
        ("median_tpot_ms",       "tpot_ms"),
        ("median_itl_ms",        "itl_ms"),
        ("total_token_throughput", "token_throughput"),
        ("output_throughput",    "output_throughput"),
        ("request_throughput",   "request_throughput"),
    )
    for src, dst in pairs:
        v = bench_result.get(src)
        if isinstance(v, (int, float)):
            out[dst] = float(v)
    if "token_throughput" not in out:
        tps = extract_tps(bench_result)
        if tps is not None:
            out["token_throughput"] = tps
    return out


def compute_primary_score(bench_result: dict, workload_name: str) -> float | None:
    """Compute a single 'higher-is-better' score for the workload's primary
    metric. PAIRED speedup = candidate_score / pristine_score works for all
    workloads regardless of metric direction (we invert lower-is-better)."""
    metrics = extract_metrics_dict(bench_result)
    metric_name, direction = PRIMARY_METRIC.get(
        workload_name, ("token_throughput", "higher_is_better")
    )
    v = metrics.get(metric_name)
    if v is None or v <= 0:
        return None
    if direction == "lower_is_better":
        return 1000.0 / v
    return v


def measure_workload_multi_profile(
    workload: dict,
    run_bench: "Callable[[dict, str, int | None], dict | None]",
    *,
    num_runs: int,
    warmup_runs: int,
    side_label: str,
    warmup_num_prompts: int | None = None,
    warm_all_profiles: bool = True,
) -> list[float]:
    """Scenario C (multi-profile, e.g. high_load) per-workload MEASURE against an
    ALREADY-RUNNING server — the agent self-bench counterpart of the verifier's
    ``_measure_one_workload_serve_multi_profile``. Kept in the SHARED helper so the
    agent's high_load self-signal is computed by the SAME logic that scores it:
    per-measurement GEOMEAN across the profiles of ``compute_primary_score`` (=
    ``request_throughput`` for high_load, per PRIMARY_METRIC), with a measurement
    index dropped if ANY profile failed to score it (paired-noise-canceling). The
    returned per-measurement list is then reduced by the SAME ratio-of-trimmed-means
    the verifier uses, so agent and verifier compute the high_load speedup identically."""
    profiles = workload.get("profiles") or []
    if not profiles:
        raise ValueError(
            f"{workload.get('name')!r}: measure_workload_multi_profile called "
            "without a 'profiles' field"
        )
    name = workload["name"]
    _warm = profiles if warm_all_profiles else profiles[:1]
    _wnp = warmup_num_prompts if warmup_num_prompts is not None else workload.get("num_prompts")
    for prof in _warm:
        for i in range(max(warmup_runs, 0)):
            run_bench(prof, f"{name}_{side_label}_{prof['name']}_warmup{i}", _wnp)
    per_profile: dict[str, list[float]] = {p["name"]: [] for p in profiles}
    for i in range(num_runs):
        for prof in profiles:
            result = run_bench(prof, f"{name}_{side_label}_{prof['name']}_meas{i}", None)
            if result:
                score = compute_primary_score(result, name)
                if score:
                    per_profile[prof["name"]].append(score)
    scored = [p for p in profiles if p.get("scored", True)]
    if not scored:
        raise ValueError(
            f"{name!r}: every profile is scored=false, so the workload has no score"
        )
    aggregated: list[float] = []
    n_meas = min((len(per_profile[p["name"]]) for p in scored), default=0)
    for i in range(n_meas):
        scores_at_i = [per_profile[p["name"]][i] for p in scored]
        if all(s > 0 for s in scores_at_i):
            aggregated.append(geomean(scores_at_i))
    return aggregated


DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_NUM_PROMPTS = 30


def resolve_workload_runtime(workload: dict) -> tuple[int, int]:
    """Return (num_prompts, max_concurrency) for a workload — from the FILE, only."""
    num_prompts = int(workload.get("num_prompts", DEFAULT_NUM_PROMPTS))
    max_concurrency = int(workload.get("max_concurrency", DEFAULT_MAX_CONCURRENCY))

    return num_prompts, max_concurrency


_kill_stray_gpu_processes = kill_stray_gpu_processes
_wait_for_gpu_free_memory = wait_for_gpu_free_memory
_trimmed_mean = trimmed_mean
_stats = stats
_geomean = geomean
_extract_tps = extract_tps
_extract_metrics_dict = extract_metrics_dict
_compute_primary_score = compute_primary_score
_resolve_workload_runtime = resolve_workload_runtime



class BuildFailed(Exception):
    """Raised by run_csrc_rebuild on cmake/nvcc failure or timeout."""


_CSRC_REBUILD_SURFACES: tuple[tuple[str, str], ...] = (
    ("csrc", "csrc/"),
    ("csrc_deps", ".csrc_src/"),
)


def csrc_surfaces_touched(repo: Path) -> set[str]:
    """Which rebuild-triggering source surfaces the agent edited.

    Returns a subset of {"csrc", "csrc_deps"} -- the keys of
    ``_CSRC_REBUILD_SURFACES``, which are also the ``get_edit_policy()`` keys
    that authorise each, so the caller can check policy per surface instead of
    letting one flag speak for a surface it does not govern.

    One `git diff` for both pathspecs: two calls would let the tree change
    between them, and the result classifies by path prefix anyway.

    The form `git diff <base>` (vs `<base>..HEAD`) compares the working
    tree against base, so it picks up both committed changes AND any
    uncommitted edits. That is what is wanted here: the reference `solve.sh`
    applies patches via `git apply --3way` without committing, and real agents
    may also leave edits uncommitted — both must trigger the rebuild.
    """
    pathspecs = [spec for _, spec in _CSRC_REBUILD_SURFACES]
    r = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only",
         "inference_base", "--", *pathspecs],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"git diff -- {' '.join(pathspecs)} failed rc={r.returncode}: "
            f"{r.stderr.strip()}"
        )
    touched: set[str] = set()
    for line in r.stdout.splitlines():
        name = line.strip()
        if not name:
            continue
        for key, spec in _CSRC_REBUILD_SURFACES:
            if name.startswith(spec):
                touched.add(key)
                break
    return touched


def csrc_touched_in_diff(repo: Path) -> bool:
    """Return True iff any rebuild-triggering C/CUDA source surface changed."""
    return bool(csrc_surfaces_touched(repo))


def run_csrc_rebuild(workspace: Path, repo: Path, venv: Path, raw_dir: Path,
                    build_timeout_sec: int, *, targets: tuple[str, ...] = (),
                    extra_env: dict | None = None,
                    log_name: str = "csrc_build.log") -> Path:
    """Invoke build_vllm_csrc.sh; return the install_prefix containing fresh .so."""
    script = workspace / "scripts" / "build_vllm_csrc.sh"
    if not script.exists():
        env_override = os.environ.get("INFB_BUILD_VLLM_CSRC_SH")
        if env_override and Path(env_override).exists():
            script = Path(env_override)
        else:
            shared_runtime = Path("/app/harness/scripts/build_vllm_csrc.sh")
            shared_devtree = Path(__file__).resolve().parent / "scripts" / "build_vllm_csrc.sh"
            alt = Path(__file__).resolve().parent.parent / "environment" / "scripts" / "build_vllm_csrc.sh"
            for cand in (shared_runtime, shared_devtree, alt):
                if cand.exists():
                    script = cand
                    break
            else:
                raise BuildFailed(
                    f"build_vllm_csrc.sh not found (tried {workspace/'scripts'}, "
                    f"{shared_runtime}, {shared_devtree}, and {alt})"
                )

    log_path = raw_dir / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()}
    env["INFB_WORKSPACE"] = str(workspace)
    env["INFB_VLLM_PYTHON"] = env.get("INFB_VLLM_PYTHON", str(venv / "bin" / "python3"))
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print(f"[csrc-rebuild] invoking {script} (timeout={build_timeout_sec}s); "
          f"log -> {log_path}", flush=True)
    t0 = time.time()
    try:
        with log_path.open("w") as logf:
            r = subprocess.run(
                ["bash", str(script), *targets],
                stdout=logf, stderr=subprocess.STDOUT,
                env=env, timeout=build_timeout_sec,
            )
    except subprocess.TimeoutExpired as e:
        raise BuildFailed(
            f"csrc rebuild timed out after {build_timeout_sec}s; log: {log_path}"
        ) from e
    elapsed = time.time() - t0
    print(f"[csrc-rebuild] rc={r.returncode} elapsed={elapsed:.1f}s", flush=True)
    if r.returncode != 0:
        try:
            tail = log_path.read_text().splitlines()[-50:]
            print("[csrc-rebuild] last 50 log lines:", flush=True)
            for ln in tail:
                print(f"  {ln}", flush=True)
        except Exception:
            pass
        raise BuildFailed(
            f"csrc rebuild failed rc={r.returncode}; log: {log_path}"
        )
    install_prefix = repo / ".csredit" / "install"
    if not (install_prefix / "vllm").is_dir():
        raise BuildFailed(
            f"csrc rebuild reported success but {install_prefix}/vllm/ missing"
        )
    return install_prefix


_DEFAULT_SO_OVERLAY_NAMES: tuple[str, ...] = (
    "_C.abi3.so",
    "_moe_C.abi3.so",
    "_C_stable_libtorch.abi3.so",
    "_moe_C_stable_libtorch.abi3.so",
)


def so_overlay_targets(repo: Path,
                       extra_targets: tuple[str, ...] = ()) -> list[Path]:
    """Per-side swap targets: the .so files prepare_target_env.sh symlinked."""
    base = repo / "vllm"
    names = list(_DEFAULT_SO_OVERLAY_NAMES) + list(extra_targets)
    return [base / n for n in names if (base / n).exists() or (base / n).is_symlink()]


def swap_so_to_built(repo: Path, install_prefix: Path,
                     extra_targets: tuple[str, ...] = ()) -> None:
    """Candidate-side swap: replace venv symlinks under repo/vllm/ with the
    freshly-built .so files from install_prefix/vllm/."""
    state_dir = repo / ".csredit"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "venv_symlinks.json"
    base = repo / "vllm"
    if not state_path.exists():
        snap = {}
        for p in so_overlay_targets(repo, extra_targets):
            if p.is_symlink():
                snap[str(p.relative_to(base))] = os.readlink(p)
        state_path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
        print(f"[so-swap] captured {len(snap)} venv symlink targets -> {state_path.name}", flush=True)

    for _t in extra_targets:
        if not (install_prefix / "vllm" / _t).exists():
            raise BuildFailed(
                f"swap_so_to_built: requested extra target '{_t}' missing from "
                f"{install_prefix}/vllm/ — the rebuild reported success but the "
                f".so is absent (csrc+FA combined-build clobber or build-script "
                f"regression). Refusing to silently benchmark the baseline .so."
            )

    swapped = 0
    for p in so_overlay_targets(repo, extra_targets):
        src = install_prefix / "vllm" / p.relative_to(base)
        if not src.exists():
            continue
        if p.is_symlink() or p.is_file():
            p.unlink()
        with src.open("rb") as fi, p.open("wb") as fo:
            while True:
                chunk = fi.read(1 << 20)
                if not chunk:
                    break
                fo.write(chunk)
        swapped += 1
    print(f"[so-swap] cand: swapped {swapped} rebuilt .so into repo/vllm/", flush=True)
    if swapped == 0:
        so_names = [p.name for p in so_overlay_targets(repo, extra_targets)]
        built = sorted(q.name for q in (install_prefix / "vllm").glob("*.abi3.so"))
        print(
            "[so-swap] *** CSREDIT-SWAP-ZERO *** rebuild produced .so but 0 were "
            "overlaid onto the candidate — the compiled kernel edit is NOT active "
            "and will be benchmarked as the BASELINE. overlay-names="
            f"{so_names} built-in-prefix={built}. Check that "
            "_DEFAULT_SO_OVERLAY_NAMES / extra_targets match the built target "
            "names for this vLLM version (e.g. _C_stable_libtorch on >=0.24.0).",
            flush=True,
        )


def venv_vllm_dir(venv: Path) -> Path | None:
    """`<venv>/lib/python*/site-packages/vllm`, or None if it is not there.

    The trust root for the pristine side's `.so`. Derived from the venv the caller
    already knows, so it does not depend on anything inside the agent's repo.
    """
    try:
        for sp in sorted(Path(venv).glob("lib/python*/site-packages/vllm")):
            if sp.is_dir():
                return sp
    except OSError:
        pass
    return None


def swap_so_to_venv(repo: Path, venv: Path | None = None) -> None:
    """Pristine-side swap: restore the original venv symlinks under repo/vllm/.

    `.csredit/venv_symlinks.json` says WHICH names were swapped. It used to also
    say WHERE each one came from, and those paths were followed with no validation
    at all — while `swap_so_to_built` writes the file only when it is ABSENT
    (under `if not state_path.exists()`), so an agent can pre-plant it and the
    verifier will not overwrite it. That let the agent choose what the SCORED
    pristine leg loads: point a record at a deliberately slow `.so` and the
    reference side gets slower, which inflates the measured speedup. The candidate
    side is unaffected (the verifier rebuilds it from the agent's own `csrc/`), so
    the channel is asymmetric and biases upward.

    Passing `venv` removes the dependency instead of policing it: the target is
    recomputed as `<venv>/lib/python*/site-packages/vllm/<name>`, and the recorded
    path is used only to cross-check (a mismatch is reported, not followed). The
    caller already has `venv` — the verifiers hand it to
    `prepare_csredit_rebuild_if_enabled` a few lines earlier.

    With `venv=None` the old behaviour is kept for compatibility, minus the clearly
    wrong case: a recorded target that resolves INSIDE `repo` would hand the pristine
    leg something from the candidate's own tree, and is refused.

    No-op if no swap has happened yet (e.g. csrc/ wasn't touched, no rebuild ran).
    """
    state_path = repo / ".csredit" / "venv_symlinks.json"
    if not state_path.exists():
        return
    snap = json.loads(state_path.read_text())
    base = repo / "vllm"
    vdir = venv_vllm_dir(venv) if venv is not None else None
    if venv is not None and vdir is None:
        print(f"[so-swap] WARN: no site-packages/vllm under {venv}; falling back to "
              f"the recorded paths for this restore", flush=True)

    def _resolved(name: str, recorded: str) -> Path | None:
        if vdir is not None:
            derived = vdir / name
            if derived.is_file():
                if recorded and Path(recorded) != derived:
                    print(f"[so-swap] NOTE: venv_symlinks.json recorded {name} at "
                          f"{recorded}, using the derived {derived}", flush=True)
                return derived
            print(f"[so-swap] WARN: {derived} is not a file; cannot derive the "
                  f"baseline for {name}", flush=True)
        if not recorded:
            return None
        rp = Path(recorded)
        if not _path_is_outside(rp, repo):
            print(f"[so-swap] REFUSED: venv_symlinks.json points {name} at {rp}, "
                  f"inside the candidate repo — the pristine leg would load the "
                  f"candidate's own kernel. Leaving {name} as-is.", flush=True)
            return None
        return rp

    restored = 0
    for name, target in snap.items():
        chosen = _resolved(name, target)
        if chosen is None:
            continue
        p = base / name
        if p.exists() or p.is_symlink():
            p.unlink()
        os.symlink(chosen, p)
        restored += 1
    src = "derived from the venv" if vdir is not None else "recorded (no venv passed)"
    print(f"[so-swap] pris: restored {restored}/{len(snap)} venv .so symlinks ({src})",
          flush=True)


_csrc_touched_in_diff = csrc_touched_in_diff
_run_csrc_rebuild = run_csrc_rebuild
_so_overlay_targets = so_overlay_targets
_swap_so_to_built = swap_so_to_built
_swap_so_to_venv = swap_so_to_venv



def get_edit_policy(task_dir: Path) -> dict:
    """Read the edit policy from the verifier's environment.

    Returns a dict with keys:
      python: bool — always True (Python is always editable)
      csrc: bool — gated by INFB_EDIT_POLICY_CSRC; default False
      editable_pkgs: list[str] — gated by INFB_EDIT_POLICY_PKGS (comma-separated);
        default []
      fa_rebuild: bool — gated by INFB_EDIT_POLICY_FA_REBUILD; default False
      csrc_deps: bool — gated by INFB_EDIT_POLICY_CSRC_DEPS; default False

    The variables are read from the verifier's environment, where the task sets
    them (`[verifier.env]` in `task.toml`, and `[environment.env]` for the agent
    phase):

      INFB_EDIT_POLICY_CSRC=1|true|yes        → csrc=True
      INFB_EDIT_POLICY_CSRC=0|false|no        → csrc=False (also the default)
      INFB_EDIT_POLICY_PKGS=pkg1,pkg2         → editable_pkgs=["pkg1", "pkg2"]
      INFB_EDIT_POLICY_FA_REBUILD=1|true|yes  → fa_rebuild=True
      INFB_EDIT_POLICY_CSRC_DEPS=1|true|yes   → csrc_deps=True

    `task_dir` is kept in the signature for backward compatibility with
    callers but no longer used — `task.toml`'s
    `[metadata.inference_bench.edit_policy]` is intentionally ignored.
    """
    del task_dir
    csrc_env = os.environ.get("INFB_EDIT_POLICY_CSRC", "").strip().lower()
    csrc = csrc_env in ("1", "true", "yes")
    pkgs_env = os.environ.get("INFB_EDIT_POLICY_PKGS", "").strip()
    editable_pkgs = [p.strip() for p in pkgs_env.split(",") if p.strip()] if pkgs_env else []
    fa_env = os.environ.get("INFB_EDIT_POLICY_FA_REBUILD", "").strip().lower()
    fa_rebuild = fa_env in ("1", "true", "yes")
    deps_env = os.environ.get("INFB_EDIT_POLICY_CSRC_DEPS", "").strip().lower()
    csrc_deps = deps_env in ("1", "true", "yes")
    return {"python": True, "csrc": csrc, "editable_pkgs": editable_pkgs,
            "fa_rebuild": fa_rebuild, "csrc_deps": csrc_deps}


def prepare_csredit_rebuild_if_enabled(
    *, task_dir: Path, workspace: Path, repo: Path, venv: Path,
    raw_dir: Path, build_timeout_sec: int = 1800,
) -> Path | None:
    """Idiomatic verifier-side gate for csrc rebuild."""
    del task_dir
    touched = csrc_surfaces_touched(repo)
    if not touched:
        return None
    policy = get_edit_policy(repo)
    _SPEC = dict(_CSRC_REBUILD_SURFACES)
    _FLAG = {"csrc": "INFB_EDIT_POLICY_CSRC", "csrc_deps": "INFB_EDIT_POLICY_CSRC_DEPS"}
    open_surfaces = {s for s in touched if policy.get(s)}
    for surface in sorted(touched - open_surfaces):
        print(
            f"[edit_policy] {_SPEC[surface]} touched but that surface is CLOSED "
            f"on this task ({_FLAG[surface]} unset) → NOT rebuilding it; the "
            "candidate side keeps the baseline .so, exactly as instruction.md "
            f"states. Recording edit_policy_violation={surface}.",
            flush=True,
        )
        record_edit_policy_violation(repo, surface)
    if not open_surfaces:
        return None
    print(
        "[edit_policy] "
        + ", ".join(_SPEC[s] for s in sorted(open_surfaces))
        + " touched in diff → triggering rebuild",
        flush=True,
    )
    return run_csrc_rebuild(
        workspace=workspace, repo=repo, venv=venv,
        raw_dir=raw_dir, build_timeout_sec=build_timeout_sec,
    )


_EDIT_POLICY_VIOLATIONS_REL = ".infb_edit_policy_violations"


def record_edit_policy_violation(repo: Path, surface: str) -> None:
    """Append ``surface`` to the violation log BESIDE the repo, best-effort.

    Beside, not inside. ``repo/`` is the tree that gets scored and turned into the
    candidate patch, and an untracked breadcrumb in it is not free:

      * ``disposable_pristine_worktree``'s ``INFB_DIAG_NO_WORKTREE=1`` path refuses
        to run on a dirty tree ("refuses dirty tree: … (commit/stash first)"), so on
        that diagnostic path recording a violation would turn a gradeable run into a
        crash — and only for runs that touched a closed surface, i.e. exactly the
        runs this breadcrumb exists to explain;
      * nothing gitignores it, so it is a candidate-diff contaminant waiting for
        someone to move the capture earlier in the sequence.

    ``repo.parent`` is the workspace (``/app/repo`` -> ``/app``): outside git, still
    adjacent enough for anything reading the trial's files to find it.
    """
    try:
        path = repo.parent / _EDIT_POLICY_VIOLATIONS_REL
        seen = set(path.read_text(encoding="utf-8").split()) if path.is_file() else set()
        if surface not in seen:
            seen.add(surface)
            path.write_text("\n".join(sorted(seen)) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[edit_policy] WARN: could not record violation: {exc}", flush=True)


def edit_policy_violations(repo: Path) -> list[str]:
    """Surfaces the agent edited that this task does not score. Never raises."""
    try:
        for path in (
            repo.parent / _EDIT_POLICY_VIOLATIONS_REL,
            repo / _EDIT_POLICY_VIOLATIONS_REL,
        ):
            if path.is_file():
                return sorted(set(path.read_text(encoding="utf-8").split()))
        return []
    except OSError:
        return []


_FA_SRC_REL = ".fa_src"


def fa_touched_in_diff(repo: Path) -> bool:
    """True iff `git diff inference_base -- .fa_src/` is non-empty (agent
    edited the staged flash-attention submodule C/CUDA source)."""
    r = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only",
         "inference_base", "--", f"{_FA_SRC_REL}/"],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"git diff -- {_FA_SRC_REL}/ failed rc={r.returncode}: {r.stderr.strip()}")
    return bool([ln for ln in r.stdout.splitlines() if ln.strip()])


def prepare_fa_rebuild_if_enabled(
    *, workspace: Path, repo: Path, venv: Path, raw_dir: Path,
    build_timeout_sec: int = 3600,
) -> Path | None:
    """Verifier-side gate for the flash-attention (_vllm_fa*_C) rebuild."""
    if not get_edit_policy(repo).get("fa_rebuild"):
        return None
    if not fa_touched_in_diff(repo):
        return None
    env_timeout = os.environ.get("INFB_FA_BUILD_TIMEOUT_SEC", "").strip()
    if env_timeout:
        try:
            build_timeout_sec = int(env_timeout)
        except ValueError:
            pass
    fa_src = os.environ.get("INFB_FA_SRC_DIR", str(repo / _FA_SRC_REL))
    print(f"[edit_policy] fa_rebuild on + .fa_src/ touched → rebuilding _vllm_fa3_C "
          f"(src={fa_src})", flush=True)
    return run_csrc_rebuild(
        workspace=workspace, repo=repo, venv=venv, raw_dir=raw_dir,
        build_timeout_sec=build_timeout_sec,
        targets=("_vllm_fa3_C",),
        extra_env={"INFB_FA_SRC_DIR": fa_src},
        log_name="fa_build.log",
    )

