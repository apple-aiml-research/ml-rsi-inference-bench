#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for optimize_dsv4_flash_pd_8xb200_sglang_open."""
from __future__ import annotations

import os as _infb_os
for _infb_np in ("no_proxy", "NO_PROXY"):
    _infb_cur = _infb_os.environ.get(_infb_np, "")
    if "127.0.0.1" not in _infb_cur:
        _infb_os.environ[_infb_np] = "127.0.0.1,localhost,::1" + (
            f",{_infb_cur}" if _infb_cur else ""
        )

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── tvm-ffi JIT cache isolation (csrc / jit_kernel editable surface) ─────
# SGLang JIT-compiles sglang/jit_kernel/csrc/** at serve time (tvm-ffi -> nvcc).
# The candidate subprocess (PYTHONPATH=<repo>) compiles the edited kernel; the
# pristine subprocess (worktree @ inference_base) compiles the unedited
# one. Both inherit the same HOME and tvm-ffi's module name doesn't encode
# source content, so a shared cache could serve an edited kernel to the
# pristine side (or vice-versa) -> silent false 1.0x. Give each side a private
# cache dir. No-op cost when no jit_kernel edit (each side just compiles once).
def _set_jit_cache_env(env: dict, base_dir, side: str) -> None:
    try:
        base = Path(base_dir) if base_dir is not None else Path("/tmp")
    except Exception:
        base = Path("/tmp")
    cache_dir = base / f"jit_cache_{side}"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    env["TVM_FFI_CACHE_DIR"] = str(cache_dir)
    env["HOME"] = str(cache_dir)


_THIS_VERIFIER_DIR = Path(__file__).resolve().parent
_HELPER_LIB_DIR = _THIS_VERIFIER_DIR / "_verifier_lib"
if _HELPER_LIB_DIR.is_dir() and str(_HELPER_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_LIB_DIR))
try:
    from _paired_helper import (  # type: ignore
        _compute_primary_score,
        _extract_metrics_dict,
        _extract_tps,
        _gpu_pids_via_proc,
        _kill_stray_gpu_processes,
        _resolve_workload_runtime,
        _trimmed_mean,
        _wait_for_gpu_free_memory,
        cleanup_stale_ipc,
        disposable_pristine_worktree,
        prepare_csredit_rebuild_if_enabled,
        PRIMARY_METRIC,
        swap_so_to_built,
        swap_so_to_venv,
    )
except ImportError:  # pragma: no cover — helper not overlaid (local dev fallback)
    cleanup_stale_ipc = None  # type: ignore
    disposable_pristine_worktree = None  # type: ignore
    _compute_primary_score = None  # type: ignore
    _extract_metrics_dict = None  # type: ignore
    _resolve_workload_runtime = None  # type: ignore
    _extract_tps = None  # type: ignore
    _gpu_pids_via_proc = None  # type: ignore
    _kill_stray_gpu_processes = None  # type: ignore
    _trimmed_mean = None  # type: ignore
    _wait_for_gpu_free_memory = None  # type: ignore
    prepare_csredit_rebuild_if_enabled = None  # type: ignore
    PRIMARY_METRIC = {}  # type: ignore
    swap_so_to_built = None  # type: ignore
    swap_so_to_venv = None  # type: ignore

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _verifier_lib import (  # noqa: E402
    classify_repo_diff,
    compute_reward,
    geomean,
    is_rejected_by_narrow,
)
from _verifier_lib.agent_notes import (
    collect_agent_notes,
)
from _verifier_lib.candidate_capture import (
    CandidateCaptureError,
    capture_untracked_candidate,
    capture_void_reason,
)
from _verifier_lib.acceptance import (
    gate as _acceptance_gate,
    read_from_raw_dir as _acceptance_read,
)
from _verifier_lib.equivalence import (
    read_from_raw_dir as _equivalence_read,
    summarise as _equivalence_summarise,
)
from _verifier_lib.output_shape import (
    read_from_raw_dir as _output_shape_read,
    summarise as _output_shape_summarise,
)
from _verifier_lib.deterministic_failures import (
    deterministic_pristine_signature,
    deterministic_signature as _deterministic_signature,
)
from _verifier_lib.preflight import (
    PREFLIGHT_VOID_EXIT_CODE,
    run_preflight,
    write_void_metrics,
)

_INFB_SPECULATIVE = False


MODEL_ID   = "deepseek-ai/DeepSeek-V4-Flash"
EXPECTED_GPU_TYPES = ['B200']
EXPECTED_GPU_COUNT = 8
# 4 -- the width of ONE pool, which is what a PD instance is launched at. pd_topology.plan() starts one
# tensor-parallel instance per pool over that pool's four devices, so the verifier scores at 4 even though
# the machine has 8.
TP_SIZE    = 4

ENGINE = os.environ.get("INFB_ENGINE", "sglang").lower()

LIVE_CORRECTNESS_SUPPORTED = True

# Per-task correctness calibration (baked in from task.toml
# [metadata.inference_bench.correctness]; the verifier container
# cannot reliably read task.toml at runtime). Keep these IN SYNC with task.toml.
CORRECTNESS_CALIBRATION = {
    "atol":          0.4,
    "rtol":          0.15,
    "min_pass_rate": 0.85,
}

# Per-task scoring calibration (baked in from task.toml
# [metadata.inference_bench.scoring]; the verifier container cannot
# reliably read task.toml at runtime). Keep IN SYNC with task.toml.
SCORING_CALIBRATION = {
    "num_runs":               5,
    "warmup_runs":            2,
    "num_runs_per_workload": {'pd_multiturn': 3},
}

TASK_SCORING = {
    "num_runs":              int(os.environ.get("INFB_NUM_RUNS_OVERRIDE", "5")),
    "warmup_runs":           int(os.environ.get("INFB_WARMUP_RUNS_OVERRIDE", "2")),
}


def _declared_runs_for(workload: str, fallback: int) -> int:
    """Per-workload rep count from the task's own declaration, with the global as fallback."""
    table = (globals().get("SCORING_CALIBRATION") or {}).get("num_runs_per_workload") or {}
    return int(table.get(workload, fallback))

# The verifier loads the real JSON if available; this dict is the final fallback.
_FALLBACK_PUBLIC_BASELINE: dict = {}

# Narrow variant: broad verifier does NOT reject any category.
NARROW_REJECTED_KINDS: set[str] = set()

_env_seed = os.environ.get("INFB_BENCH_SEED")
_RUN_SEED = int(_env_seed) if (_env_seed and _env_seed.strip()) else (int(time.time()) & 0x7FFFFFFF)
print(f"[verifier] prompt seed for this eval (cand+pris shared): {_RUN_SEED} "
      f"(set INFB_BENCH_SEED=<int> to pin/reproduce)", flush=True)


def _default_visible_devices() -> str:
    """CUDA_VISIBLE_DEVICES fallback for every engine this file spawns: TP_SIZE devices."""
    return ",".join(str(i) for i in range(TP_SIZE))



def _apply_nccl_compat(env, fallback_path):
    """Prefer the harness NCCL shim (INFB_NCCL_PRELOAD); legacy per-build path as
    fallback. Pins BOTH LD_PRELOAD (prepended, preserving any existing value) and
    VLLM_NCCL_SO_PATH so vLLM's own loader picks the compat nccl too (more robust
    than a bare LD_PRELOAD set)."""
    for _nccl in (env.get("INFB_NCCL_PRELOAD"), fallback_path):
        if not _nccl or not __import__("os").path.exists(_nccl):
            continue
        _parts = env.get("LD_PRELOAD", "").split()
        if _nccl not in _parts:
            env["LD_PRELOAD"] = " ".join([_nccl, *_parts])
        env["VLLM_NCCL_SO_PATH"] = _nccl
        return


def _force_spawn_workers(env):
    """Make vLLM start its TP workers with `spawn`, not `fork`."""
    method = os.environ.get("INFB_CORRECTNESS_MULTIPROC_METHOD", "spawn")
    env["VLLM_WORKER_MULTIPROC_METHOD"] = method
    print(f"[multiproc] VLLM_WORKER_MULTIPROC_METHOD={method} for the correctness engine "
          f"(fork into an initialized CUDA context kills every TP worker)",
          flush=True)


def _flashinfer_overlay_path() -> str | None:
    """Return the writable flashinfer_cubin overlay dir if it exists."""
    _cands = []
    _ws = os.environ.get("INFB_WORKSPACE")
    if _ws:
        _cands.append(Path(_ws) / ".flashinfer_overlay")
    _cands += [Path("/app/.flashinfer_overlay"), Path("/logs/artifacts/.flashinfer_overlay")]
    for _c in _cands:
        if (_c / "flashinfer_cubin").is_dir():
            return str(_c)
    return None


_SGLK_OVERLAY: str | None = None


def _sgl_kernel_overlay_path() -> str | None:
    """Return the rebuilt-sgl_kernel overlay dir (candidate-only), or None."""
    return _SGLK_OVERLAY


def _sgl_kernel_rebuild_enabled() -> bool:
    return os.environ.get("INFB_EDIT_POLICY_CSRC_SGLKERNEL", "").strip().lower() in ("1", "true", "yes")


def _sgl_kernel_touched(repo: Path) -> bool:
    """True iff `git diff --name-only inference_base -- sgl-kernel/` is non-empty."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "diff", "--name-only",
             "inference_base", "--", "sgl-kernel/"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return False
    if r.returncode != 0:
        return False
    return any(ln.strip() for ln in r.stdout.splitlines())


def _maybe_build_sgl_kernel(*, workspace: Path, repo: Path, venv: Path,
                            raw_dir: Path, metrics: dict) -> str | None:
    """Heavy sgl-kernel rebuild gate. Returns the candidate-side PYTHONPATH
    overlay dir (rebuilt sgl_kernel package), or None."""
    if not _sgl_kernel_rebuild_enabled():
        metrics["sglkernel_rebuild"] = "disabled (INFB_EDIT_POLICY_CSRC_SGLKERNEL unset in the VERIFIER env)"
        return None
    if not _sgl_kernel_touched(repo):
        metrics["sglkernel_rebuild"] = "skipped (sgl-kernel/ untouched)"
        return None

    _here = Path(__file__).resolve().parent
    script = None
    for cand in (
        Path(os.environ.get("INFB_BUILD_SGL_KERNEL_SH", "")) if os.environ.get("INFB_BUILD_SGL_KERNEL_SH") else None,
        _here.parent / "environment" / "scripts" / "build_sgl_kernel.sh",
        workspace / "scripts" / "build_sgl_kernel.sh",
        Path("/app/scripts/build_sgl_kernel.sh"),
        Path("/app/harness/scripts/build_sgl_kernel.sh"),
    ):
        if cand is not None and cand.exists():
            script = cand
            break
    if script is None:
        metrics["sglkernel_rebuild"] = "failed: build_sgl_kernel.sh not found"
        raise RuntimeError(
            "[sgl-kernel] INFB_EDIT_POLICY_CSRC_SGLKERNEL=1 + sgl-kernel/ touched, but "
            f"build_sgl_kernel.sh not found (tried {_here.parent/'environment'/'scripts'}, "
            f"{workspace/'scripts'}, /app/scripts, /app/harness/scripts). Either the verifier env "
            "lost INFB_BUILD_SGL_KERNEL_SH or the task image never got the helper.")
    print(f"[sgl-kernel] using build script: {script}", flush=True)

    install_prefix = repo / ".sglkredit" / "install"
    log_path = raw_dir / "sgl_kernel_build.log"
    build_timeout = int(os.environ.get("INFB_SGLKERNEL_BUILD_TIMEOUT_SEC", "5400"))
    env = {k: v for k, v in os.environ.items()}
    env["INFB_WORKSPACE"] = str(workspace)
    env["INFB_SGLKERNEL_INSTALL_PREFIX"] = str(install_prefix)
    env["INFB_SGLKERNEL_PYTHON"] = env.get("INFB_SGLKERNEL_PYTHON", str(venv / "bin" / "python3"))
    print(f"[sgl-kernel] rebuild gate: sgl-kernel/ touched + flag on → building "
          f"(script={script}, timeout={build_timeout}s, log→{log_path})", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with log_path.open("w") as logf:
            r = subprocess.run(["bash", str(script)], stdout=logf,
                               stderr=subprocess.STDOUT, env=env, timeout=build_timeout)
    except subprocess.TimeoutExpired:
        metrics["sglkernel_rebuild"] = f"failed: timed out after {build_timeout}s"
        raise RuntimeError(f"[sgl-kernel] rebuild timed out after {build_timeout}s; log: {log_path}")
    elapsed = time.time() - t0
    if r.returncode != 0:
        try:
            tail = log_path.read_text().splitlines()[-40:]
            print("[sgl-kernel] build FAILED — last 40 log lines:", flush=True)
            for ln in tail:
                print(f"  {ln}", flush=True)
        except Exception:
            pass
        metrics["sglkernel_rebuild"] = f"failed: rc={r.returncode} (elapsed={elapsed:.0f}s)"
        raise RuntimeError(f"[sgl-kernel] rebuild failed rc={r.returncode}; log: {log_path}")
    if not (install_prefix / "sgl_kernel").is_dir():
        metrics["sglkernel_rebuild"] = "failed: install_prefix/sgl_kernel missing after build"
        raise RuntimeError(f"[sgl-kernel] build reported success but {install_prefix}/sgl_kernel/ missing")
    metrics["sglkernel_rebuild"] = "ok"
    metrics["sglkernel_overlay"] = str(install_prefix)
    metrics["sglkernel_build_elapsed_sec"] = round(elapsed, 1)
    print(f"[sgl-kernel] rebuild OK ({elapsed:.0f}s); candidate PYTHONPATH overlay → {install_prefix}", flush=True)
    return str(install_prefix)


_SGLANG_PP_DIAG_DONE = False


def _sglang_venv_site_packages(venv: Path, *, label: str = "") -> list[str]:
    """Return the artifact-venv site-packages dir(s) holding the sglang deps
    (IPython/torch/sgl_kernel/flashinfer), for injection onto a subprocess
    PYTHONPATH."""
    import glob as _glob

    def _scan(roots):
        seen = []
        for r in roots:
            for sub in ("lib", "lib64"):
                for d in sorted(_glob.glob(str(Path(r) / sub / "python*" / "site-packages"))):
                    if os.path.isdir(d) and d not in seen:
                        seen.append(d)
        return seen

    def _valid(dirs):
        return [d for d in dirs
                if os.path.isdir(os.path.join(d, "IPython"))
                or os.path.isdir(os.path.join(d, "torch"))]

    artifact_roots = []
    for ev in ("INFB_VLLM_TARGET_ENV_ARTIFACT", "INFB_SGLANG_TARGET_ENV_ARTIFACT"):
        base = os.environ.get(ev)
        if base and (Path(base) / "venv") not in artifact_roots:
            artifact_roots.append(Path(base) / "venv")

    art_matched = _scan(artifact_roots)
    art_valid = _valid(art_matched)
    fb_matched = _scan([venv])
    fb_valid = _valid(fb_matched)

    if art_valid:
        result = art_valid
    elif art_matched:
        result = art_matched
    else:
        result = fb_valid or fb_matched

    global _SGLANG_PP_DIAG_DONE
    if not _SGLANG_PP_DIAG_DONE:
        _SGLANG_PP_DIAG_DONE = True
        try:
            vlist = os.listdir(venv) if os.path.isdir(venv) else "<venv-missing>"
        except Exception as e:
            vlist = f"<listdir-err:{e}>"
        print(f"[sglang-pp:{label}] resolved_venv={venv} exists={os.path.isdir(venv)} "
              f"contents={vlist}", flush=True)
        print(f"[sglang-pp:{label}] artifact_roots={[str(r) for r in artifact_roots]}", flush=True)
        print(f"[sglang-pp:{label}] artifact_validated={art_valid}", flush=True)
        print(f"[sglang-pp:{label}] fallback(resolved-venv)_validated={fb_valid}", flush=True)
        print(f"[sglang-pp:{label}] -> PYTHONPATH adds (artifact-exclusive when avail) {result}", flush=True)
    return result


def _sglang_syslib_dirs() -> list[str]:
    """Return <artifact>/syslibs dir(s) the target-env artifact bundles (libnuma.so.1, …)."""
    dirs: list[str] = []
    for ev in ("INFB_VLLM_TARGET_ENV_ARTIFACT", "INFB_SGLANG_TARGET_ENV_ARTIFACT"):
        base = os.environ.get(ev)
        if base:
            d = os.path.join(base, "syslibs")
            if os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    return dirs


def _set_offline_env(env: dict) -> None:
    """Apply (or remove) HF offline env vars on a subprocess env, engine-aware."""
    if ENGINE == "sglang":
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    else:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")


def _find_model(workspace: Path) -> str:
    """Resolve the model path."""
    if env := os.environ.get("INFB_MODEL_PATH"):
        return env
    ws_model = workspace / "model"
    if ws_model.exists():
        return str(ws_model)
    hf_home = Path(os.environ.get("HF_HOME",
                   Path(os.environ.get("HOME", "/root")) / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub" / "models--deepseek-ai--DeepSeek-V4-Flash" / "snapshots"
    if hub_dir.is_dir():
        snapshots = sorted(hub_dir.iterdir())
        if snapshots:
            return str(snapshots[-1])
    return MODEL_ID


def _find_venv(task_dir: Path) -> Path:
    """Resolve the path to the vLLM venv (root dir, .../venv)."""
    if env := os.environ.get("INFB_VENV"):
        return Path(env)
    try:
        import sys as _sys
        candidate_dirs = []
        ws = os.environ.get("INFB_WORKSPACE")
        if ws:
            candidate_dirs.append(Path(ws) / "scripts")
        candidate_dirs.append(Path("/app/scripts"))
        candidate_dirs.append(Path(__file__).resolve().parent.parent / "environment" / "scripts")
        for d in candidate_dirs:
            if (d / "_target_env.py").exists():
                _sys.path.insert(0, str(d))
                from _target_env import resolve_venv_dir
                return resolve_venv_dir()
    except Exception:
        pass
    if env := os.environ.get("INFB_ARTIFACT_CACHE"):
        candidate = Path(env) / "builds" / "vllm" / "dsv4_flash_pd_8xb200" / "venv"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Cannot find vllm venv. Set INFB_VENV, INFB_VLLM_PYTHON, "
        "INFB_VLLM_TARGET_ENV_ARTIFACT, or INFB_ARTIFACT_CACHE; or install "
        "to .cache/inference-bench/builds/vllm/dsv4_flash_pd_8xb200/venv"
    )




def _clear_l2_cache() -> None:
    """Best-effort L2 cache thrash before a throughput subprocess."""
    code = (
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    dummy = torch.empty(\n"
        "        (32, 1024, 1024), dtype=torch.int64, device='cuda'\n"
        "    )\n"
        "    dummy.fill_(42)\n"
        "    torch.cuda.synchronize()\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if completed.returncode != 0:
        print("  [verifier] L2 cache thrash skipped (best-effort subprocess failed)")



def _pick_free_port() -> int:
    """Bind to a free port and return it. Below 32768, Linux's ephemeral range, so a readiness
    poll can never be handed the probed port as its own source port and read its request back
    (observed once as `BadStatusLine: GET /health HTTP/1.1` on port 54498). Constrained so sglang
    v0.5.12+'s derived grpc port (SGLANG_GRPC_PORT = port + 10000) stays <= 65535
    (an OS-assigned ephemeral port like 57213 -> grpc 67213 fails sglang's
    1..65535 validation). Small TOCTOU window but fine — the server binds the
    same port immediately after."""
    import socket, random
    for _ in range(64):
        cand = random.randint(20000, 32767)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", cand))
                return s.getsockname()[1]
            except OSError:
                continue
    for _ in range(16):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if p <= 55000:
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server_ready(port: int, timeout: float, server_proc) -> bool:
    """Poll the engine's readiness endpoint until ready / timeout / death."""
    import urllib.request
    import urllib.error
    if ENGINE == "sglang":
        url = f"http://127.0.0.1:{port}/health"
    else:
        url = f"http://127.0.0.1:{port}/v1/models"
    _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    _seen_probe_errors = set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_proc.poll() is not None:
            return False
        try:
            with _opener.open(url, timeout=2) as r:
                if 200 <= r.status < 300:
                    return True
        # BROAD ON PURPOSE: http.client.BadStatusLine derives from HTTPException, NOT from OSError,
        # so a narrower except escaped and killed a whole trial on the first poll. A poller whose job
        # is "keep trying until ready or the process dies" must not die on one malformed response.
        except Exception as _e:
            _kind = type(_e).__name__
            if _kind not in _seen_probe_errors:
                _seen_probe_errors.add(_kind)
                print(f"[serve-wait] :{port} probe raised {_kind}: {str(_e)[:160]} "
                      f"-- treating as not-ready and continuing to poll", flush=True)
        time.sleep(2)
    return False



def _start_engine_serve(
    *,
    model: str,
    venv: Path,
    repo: Path,
    port: int,
    pythonpath_overlay: bool,
    log_path: Path,
    apply_sglk_overlay: bool = False,
):
    """Spawn `vllm serve` in its own process group."""
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").strip()
    _cuda_devs_for_wait = [d for d in _visible.split(",") if d.strip()] or ["0"]
    for _dev in _cuda_devs_for_wait:
        _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=2.0)
        try:
            _wait_for_gpu_free_memory(cuda_device=_dev,
                                      required_gib=60.0, timeout_sec=180.0)
        except TimeoutError as _e:
            print(f"[gpu-wait] WARN: {_e}; proceeding with spawn anyway", flush=True)

    cmd = [
        str(venv / "bin" / "python"), "-P",
        "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tensor-parallel-size", str(TP_SIZE),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.85",
        "--no-enable-prefix-caching",
    ]
    if ENGINE == "sglang":
        cmd = [
            str(venv / "bin" / "python"), "-P",
            "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--tp-size", str(TP_SIZE),
            "--mem-fraction-static", "0.85",
            "--trust-remote-code",
            "--context-length", "16384",
            "--chunked-prefill-size", "8192",
            "--disable-radix-cache",
            "--moe-runner-backend", "flashinfer_mxfp4",
        ]

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    _set_jit_cache_env(env, log_path.parent, "cand" if apply_sglk_overlay else "pris")
    pp_parts = []
    if pythonpath_overlay:
        pp_parts.append(str(repo))
    if ENGINE == "sglang":
        pp_parts += _sglang_venv_site_packages(venv, label="server")
    if (fi_overlay := _flashinfer_overlay_path()) is not None:
        pp_parts.insert(0, fi_overlay)  # overlay BEFORE venv site-packages -> the writable
                                        # flashinfer_cubin wins the import (the venv copy is read-only)
    # sgl-kernel rebuild overlay: CANDIDATE-ONLY. Gated on apply_sglk_overlay and NOT on
    # pythonpath_overlay, which is True for BOTH sides -- gating the kernel on it leaked the rebuilt
    # kernel onto the PRISTINE side too and zeroed a real csrc speedup (both sides ran the candidate
    # kernel). This single insert IS the candidate/pristine asymmetry.
    # ORDER: this insert(0) runs AFTER the flashinfer insert(0) above, so the final order is
    # [sglk, flashinfer, repo?, venv-sites]; swapping the two changes which overlay shadows which.
    if apply_sglk_overlay and (_sglk := _sgl_kernel_overlay_path()) is not None:
        pp_parts.insert(0, _sglk)
    if pp_parts:
        env["PYTHONPATH"] = ":".join(pp_parts)
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    if ENGINE == "sglang":
        _syslibs = _sglang_syslib_dirs()
        if _syslibs:
            _prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(_syslibs + ([_prev] if _prev else []))
    _set_offline_env(env)
    env.setdefault("CUDA_VISIBLE_DEVICES", _default_visible_devices())

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w")

    _sglk_status = _sgl_kernel_overlay_path() if (apply_sglk_overlay and _sgl_kernel_overlay_path()) else None
    print(f"[serve] starting {ENGINE} serve on :{port} overlay={pythonpath_overlay} "
          f"sglk_overlay={'REBUILT:'+_sglk_status if _sglk_status else 'BASELINE(venv)'}, log → {log_path}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def _stop_engine_serve(proc) -> None:
    """SIGKILL the entire process group. vllm V1 WorkerProc cleanup is fragile."""
    import signal as _signal
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
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




def _collect_dual_track_metrics(
    raw_dir: Path, workloads: list[dict], num_runs: int
) -> dict[str, dict]:
    """Re-extract per-workload-per-side dual-track metrics from raw bench
    JSONs (see InferenceBench A/B/C alignment)."""
    if _extract_metrics_dict is None or _trimmed_mean is None:
        return {}

    def _avg_per_metric(files: list[Path]) -> dict[str, float]:
        per_metric: dict[str, list[float]] = {}
        for fp in files:
            if not fp.exists():
                continue
            try:
                bench = json.loads(fp.read_text())
            except (OSError, ValueError):
                continue
            for k, v in _extract_metrics_dict(bench).items():
                per_metric.setdefault(k, []).append(v)
        return {k: _trimmed_mean(v) for k, v in per_metric.items() if v}

    out: dict[str, dict] = {}
    for wl in workloads:
        wl_name = wl["name"]
        profiles = wl.get("profiles") or []
        out[wl_name] = {}
        for side in ("cand", "pristine"):
            if profiles:
                per_profile: dict[str, dict[str, float]] = {}
                for prof in profiles:
                    pname = prof["name"]
                    files = [
                        raw_dir / f"bench_{wl_name}_{wl_name}_{side}_{pname}_meas{i}.json"
                        for i in range(num_runs)
                    ]
                    per_profile[pname] = _avg_per_metric(files)
                out[wl_name][side] = per_profile
            else:
                files = [
                    raw_dir / f"bench_{wl_name}_{wl_name}_{side}_meas{i}.json"
                    for i in range(num_runs)
                ]
                out[wl_name][side] = _avg_per_metric(files)
    return out


def _run_bench_serve(
    *,
    workload: dict,
    model_path: str,
    venv: Path,
    port: int,
    run_label: str,
    raw_dir: Path,
    profile: dict | None = None,
    warmup: bool = False,
    client_repo: Path | None = None
) -> dict | None:
    """Run one `vllm bench serve --base-url http://127.0.0.1:$PORT` against
    the running server. The agent-facing twin is environment/scripts/_run_benchmark_serve.py."""
    if _resolve_workload_runtime is not None:
        eff_num_prompts, max_concurrency = _resolve_workload_runtime(workload)
    else:
        eff_num_prompts = int(workload.get("num_prompts", 30))
        max_concurrency = int(workload.get("max_concurrency", 1))
    request_rate = "inf"
    burstiness = "1.0"
    if profile is not None:
        eff_num_prompts = profile.get(
            "num_requests", profile.get("num_prompts", eff_num_prompts)
        )
        pattern = profile.get("pattern", "burst")
        if pattern in ("burst", "closed_loop"):
            request_rate = "inf"
        else:
            rate = profile.get("rate_per_s")
            if rate is None:
                raise ValueError(
                    f"profile {profile.get('name')!r}: pattern={pattern!r} "
                    f"requires rate_per_s"
                )
            request_rate = str(rate)
            if pattern == "constant":
                burstiness = "inf"
        max_concurrency = int(profile.get("concurrency", max_concurrency))

    if warmup:
        eff_num_prompts = min(
            int(os.environ.get("INFB_WORKLOAD_WARMUP_NUM_PROMPTS", "1")),
            eff_num_prompts,
        )

    out_file = raw_dir / f"bench_{workload['name']}_{run_label}.json"
    wl_json  = raw_dir / f"wl_{workload['name']}_{run_label}.json"
    eff_output_len = int(workload["output_len"])
    wl_json.write_text(json.dumps({
        "model":           model_path,
        "input_len":       workload["input_len"],
        "output_len":      eff_output_len,
        "num_prompts":     eff_num_prompts,
        "port":            port,
        "request_rate":    request_rate,
        "max_concurrency": max_concurrency,
        "burstiness":      burstiness,
        "seed":            _RUN_SEED,
    }))

    runner = THIS_DIR / "_candidate_runner.py"
    python = venv / "bin" / "python"

    cmd = [str(python), "-P", str(runner),
           "serve_bench",
           "--workload-json", str(wl_json),
           "--out-json",      str(out_file)]

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    _set_jit_cache_env(env, raw_dir, "client")
    if ENGINE == "sglang":
        _repo = client_repo if client_repo is not None else (
            Path(os.environ.get("INFB_WORKSPACE", "/app")) / "repo"
        )
        _pp = [str(_repo)] + _sglang_venv_site_packages(venv, label="bench-serve")
        if (fi_overlay := _flashinfer_overlay_path()) is not None:
            _pp.insert(0, fi_overlay)
        env["PYTHONPATH"] = ":".join(_pp)
        _syslibs = _sglang_syslib_dirs()
        if _syslibs:
            _prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(_syslibs + ([_prev] if _prev else []))
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    _set_offline_env(env)

    print(f"  [bench-serve] {workload['name']} [{run_label}]", flush=True)
    _clear_l2_cache()
    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                start_new_session=True, timeout=int(os.environ.get("INFB_BENCH_SUBPROCESS_TIMEOUT_SEC", "1800")))
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", flush=True)
        return None
    elapsed = time.time() - t0
    print(f"    rc={result.returncode}  elapsed={elapsed:.1f}s", flush=True)

    if out_file.exists():
        try:
            data = json.loads(out_file.read_text())
            if not data.get("error") and any(
                k in data for k in ("total_token_throughput", "tokens_per_second",
                                    "request_throughput")
            ):
                if result.returncode != 0:
                    print(f"    (rc={result.returncode} but valid TPS in file — trusting)",
                          flush=True)
                return data
        except Exception:
            pass

    if result.returncode != 0:
        print(f"    STDERR:\n{result.stderr[-1500:]}", flush=True)
        return None
    if not out_file.exists():
        return None
    try:
        return json.loads(out_file.read_text())
    except Exception:
        return None





JIT_STACK_FLOOR_BYTES = 64 * 1024 * 1024
JIT_HOST_COMPILERS = ("clang++", "clang++-18", "clang++-17")



FLASHINFER_UNDERALIGNED = ("alignas(64) CUtensorMap", "alignas(128) CUtensorMap")


def _patch_flashinfer_underaligned_tensormap() -> str:
    """Return what happened, for the log. See FLASHINFER_UNDERALIGNED."""
    import glob as _glob
    old, new = FLASHINFER_UNDERALIGNED
    hits, patched, refused = 0, 0, 0
    for path in _glob.glob("/opt/venv/lib/python*/site-packages/flashinfer/data/csrc/**/*.cu",
                           recursive=True):
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        if old not in text:
            continue
        hits += 1
        try:
            Path(path).write_text(text.replace(old, new), encoding="utf-8")
            patched += 1
        except OSError:
            refused += 1
    if not hits:
        return "no under-aligned CUtensorMap found (already patched, or this image does not ship it)"
    return f"{patched} file(s) re-aligned to 128, {refused} refused (read-only)"

def _raise_stack_limit_for_jit() -> str:
    """Lift RLIMIT_STACK so the engine's trial-time JIT can compile its heaviest translation unit."""
    import os as _os
    import resource
    import shutil as _shutil
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    inf = resource.RLIM_INFINITY
    outcome = None
    for want in ((inf, inf), (hard, hard)):
        if (soft, hard) == want:
            outcome = "already " + str(soft) + "/" + str(hard)
            break
        try:
            resource.setrlimit(resource.RLIMIT_STACK, want)
            outcome = str(soft) + "/" + str(hard) + " -> " + str(want[0]) + "/" + str(want[1])
            break
        except (ValueError, OSError):
            continue
    if outcome is None:
        outcome = "unchanged " + str(soft) + "/" + str(hard)
    now = resource.getrlimit(resource.RLIMIT_STACK)[0]
    if now != inf and now < JIT_STACK_FLOOR_BYTES:
        try:
            declared = str(_pd_metadata().get("jit_host_compiler") or "").strip()
        except Exception:
            declared = ""
        chosen = None
        if declared:
            for _cand in (declared, *(c for c in JIT_HOST_COMPILERS if c != declared)):
                chosen = _shutil.which(_cand)
                if chosen:
                    break
        if declared and chosen is None:
            outcome += ("; the task declares jit_host_compiler=" + repr(declared) + " and it is NOT on PATH,"
                        " so nvcc keeps its default and a deep-template unit may still exhaust the stack")
        elif chosen is not None:
            existing = _os.environ.get("NVCC_APPEND_FLAGS", "").strip()
            if "-ccbin" not in existing:
                _os.environ["NVCC_APPEND_FLAGS"] = (existing + " -ccbin " + chosen).strip()
            outcome += ("; stack below " + str(JIT_STACK_FLOOR_BYTES) + " and the task declares "
                        "jit_host_compiler=" + repr(declared) + ", so NVCC_APPEND_FLAGS='"
                        + _os.environ["NVCC_APPEND_FLAGS"] + "'. This is NOT a relapse: the -ccbin "
                        "refutation is about DeepGEMM's JIT, and a task may only declare this when that "
                        "JIT is off its critical path -- see the task's own comment for the evidence")
        else:
            outcome += ("; stack below " + str(JIT_STACK_FLOOR_BYTES) + " and left there -- the clang "
                        "-ccbin route is REFUTED by default (measured twice: nvcc's frontend "
                        "fails on glibc's _FloatN in clang mode), so a deep-template JIT unit could still "
                        "exhaust it")
    return outcome


def _pd_metadata() -> dict:
    """The PD table, resolved where the verifier can see it."""
    import tomllib

    def _experiment_overrides(table: dict) -> dict:
        """INFB_PD_REPLY_LEN, applied HERE so every reader of the table sees one value."""
        if not (os.environ.get("INFB_PD_REPLY_LEN", "").strip()
                or os.environ.get("INFB_PD_TURN_LEN", "").strip()):
            return table
        from _verifier_lib.pd_topology import (  # noqa: PLC0415
            apply_workload_overrides,
        )

        return apply_workload_overrides(table)

    here = Path(__file__).resolve().parent
    blob = here / "_verifier_lib" / "pd_topology.json"
    if blob.is_file():
        table = (json.loads(blob.read_text(encoding="utf-8")).get("pd") or {})
        if table:
            return _experiment_overrides(table)
    task_toml = here.parents[0] / "task.toml"
    if not task_toml.is_file():
        if len(here.parents) > 1:
            task_toml = here.parents[1] / "task.toml"
    if task_toml.is_file():
        doc = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        table = ((doc.get("metadata") or {}).get("inference_bench") or {}).get("pd") or {}
        if table:
            return _experiment_overrides(table)
    if (here / "_verifier_lib" / "pd_topology.py").is_file():
        raise SystemExit(
            "FATAL: this task ships the PD modules but no PD table is readable. Looked for "
            + str(blob) + " and " + str(task_toml) + ". When the verifier is mounted at /tests, "
            "task.toml is NOT one directory up -- the json beside this file is what has to be there."
        )
    return {}


def _pd_shared(name: str):
    """Import a shared PD module from beside this verifier."""
    import importlib.util
    here = Path(__file__).resolve().parent
    candidates = [here / "_verifier_lib" / (name + ".py"), here / (name + ".py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(name, candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise SystemExit(
        "FATAL: the shared PD module " + name + ".py is not beside this verifier. Looked in "
        + str([str(c) for c in candidates]) + ". When the verifier is mounted at /tests it has to "
        "be in _verifier_lib/, beside this file."
    )


def _pd_router_module() -> Path:
    """pd_router.py, found without building a path that cannot exist."""
    here = Path(__file__).resolve().parent
    candidates = [here / "_verifier_lib" / "pd_router.py"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "FATAL: pd_router.py is not beside this verifier. Looked in "
        + str([str(c) for c in candidates])
    )


def _pd_task_int(section: str, key: str, default: int) -> int:
    """One integer from the task's own metadata, resolved from /tests too."""
    import tomllib
    here = Path(__file__).resolve().parent
    blob = here / "_verifier_lib" / "pd_topology.json"
    if blob.is_file():
        table = ((json.loads(blob.read_text(encoding="utf-8")).get("sections") or {})
                 .get(section) or {})
        if key in table:
            try:
                return int(table[key])
            except (TypeError, ValueError):
                pass
    for up in (0, 1):
        if len(here.parents) <= up:
            break
        task_toml = here.parents[up] / "task.toml"
        if not task_toml.is_file():
            continue
        doc = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        table = ((doc.get("metadata") or {}).get("inference_bench") or {}).get(section) or {}
        if key in table:
            try:
                return int(table[key])
            except (TypeError, ValueError):
                break
    return default


_PD_SCORED_METRIC = PRIMARY_METRIC.get("pd_multiturn") or ("pd_latency_itl_geomean_ms",
                                                          "lower_is_better")
_PD_SCORED_KEY, _PD_SCORED_DIRECTION = _PD_SCORED_METRIC[0], _PD_SCORED_METRIC[1]


def _pd_score_from(extract: dict) -> float:
    """The value the paired driver compares, in ITS convention.

    The driver computes speedup = cand/pris and treats bigger as better; compute_primary_score is
    what puts every other workload into that convention, via 1000.0/v for a lower_is_better
    metric. It cannot do it here -- extract_metrics_dict maps six vLLM bench-serve keys and no
    pd_* one, so it returns None. An uninverted value makes a real PD improvement score 0 and a
    REGRESSION score positive, so the rule is mirrored here and cross-checked against the
    function that owns it.
    """
    value = float(extract[_PD_SCORED_KEY])
    if value <= 0:
        raise SystemExit(
            "FATAL: " + _PD_SCORED_KEY + " came back as " + repr(value) + "; a non-positive latency "
            "cannot be scored and would divide by zero in the driver's ratio."
        )
    if _PD_SCORED_DIRECTION != "lower_is_better":
        return value
    if _compute_primary_score is not None:
        probe = None
        for _wl, (_metric, _dir) in PRIMARY_METRIC.items():
            if _dir == "lower_is_better" and _metric == "ttft_ms":
                probe = _compute_primary_score({"median_ttft_ms": 2.0}, _wl)
                break
        if probe is not None and abs(probe - 500.0) > 1e-9:
            raise SystemExit(
                "FATAL: compute_primary_score no longer inverts lower_is_better as 1000/v (probe "
                "returned " + repr(probe) + " for 2.0 ms), so this mirror of that rule is wrong and "
                "a PD improvement would score backwards."
            )
        if probe is None and PRIMARY_METRIC:
            raise SystemExit(
                "FATAL: no lower_is_better/ttft_ms workload in PRIMARY_METRIC to probe, so the check "
                "that this mirror of compute_primary_score is still correct is VACUOUS."
            )
    return 1000.0 / value


def _pd_engine_env(repo: Path, venv: Path, pythonpath_overlay: bool, log_dir: Path,
                   side: str) -> dict:
    """The engine environment, EXTRACTED from _start_vllm_serve.

    The PD path used to set none of this -- most importantly not PYTHONPATH -- so both arms
    imported the engine from the venv and the candidate's edited tree was never loaded. That is
    a speedup of ~1.0 by construction.
    """
    log_path = Path(log_dir).parent / "engine.log"
    apply_sglk_overlay = side.lower().startswith("cand")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    _set_jit_cache_env(env, log_path.parent, "cand" if apply_sglk_overlay else "pris")
    pp_parts = []
    if pythonpath_overlay:
        pp_parts.append(str(repo))
    if ENGINE == "sglang":
        pp_parts += _sglang_venv_site_packages(venv, label="server")
    if (fi_overlay := _flashinfer_overlay_path()) is not None:
        pp_parts.insert(0, fi_overlay)  # overlay BEFORE venv site-packages -> the writable
                                        # flashinfer_cubin wins the import (the venv copy is read-only)
    # sgl-kernel rebuild overlay: CANDIDATE-ONLY. Gated on apply_sglk_overlay and NOT on
    # pythonpath_overlay, which is True for BOTH sides -- gating the kernel on it leaked the rebuilt
    # kernel onto the PRISTINE side too and zeroed a real csrc speedup (both sides ran the candidate
    # kernel). This single insert IS the candidate/pristine asymmetry.
    # ORDER: this insert(0) runs AFTER the flashinfer insert(0) above, so the final order is
    # [sglk, flashinfer, repo?, venv-sites]; swapping the two changes which overlay shadows which.
    if apply_sglk_overlay and (_sglk := _sgl_kernel_overlay_path()) is not None:
        pp_parts.insert(0, _sglk)
    if pp_parts:
        env["PYTHONPATH"] = ":".join(pp_parts)
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    if ENGINE == "sglang":
        _syslibs = _sglang_syslib_dirs()
        if _syslibs:
            _prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(_syslibs + ([_prev] if _prev else []))
    _set_offline_env(env)
    env.setdefault("CUDA_VISIBLE_DEVICES", _default_visible_devices())

    return env




_PD_TAIL_TTFT_BY_SIDE: dict[str, float] = {}


def _measure_one_workload_pd(
    *,
    side: str,
    wl: dict,
    model_path: str,
    venv: Path,
    repo: Path,
    pythonpath_overlay: bool,
    num_runs: int,
    warmup_runs: int,
    raw_dir: Path,
    server_boot_timeout: int = int(os.environ.get("INFB_SERVE_READY_TIMEOUT_SEC", "1800")),
    client_repo=None,
) -> list[float]:
    """One side of the pair on a disaggregated deployment. Returns {workload: [p95_ms per run]}.

    LOWER IS BETTER here, unlike every other path in this file. PRIMARY_METRIC["pd_multiturn"]
    declares lower_is_better and compute_primary_score inverts via 1000.0/v, so the existing
    reward maths becomes max(0, pristine_p95 / candidate_p95 - 1) with no new arithmetic.
    """
    pd_meta = _pd_metadata()
    topo = _pd_shared("pd_topology")
    metrics = _pd_shared("pd_turn_metrics")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", _default_visible_devices())
    _pd_aggregated = os.environ.get("INFB_PD_AGGREGATED", "").strip().lower() in ("1", "true", "yes", "on")
    topology = topo.Topology.from_metadata(pd_meta, visible, aggregated=_pd_aggregated)
    width = topo.tensor_parallel_size(topology)
    served = "model"
    print("[verify:" + side + "] stack rlimit for JIT: " + _raise_stack_limit_for_jit(), flush=True)
    print("[verify:" + side + "] PD topology " + topology.label() + " on devices " + visible
          + " (TP=" + str(width) + " per pool)", flush=True)

    if _kill_stray_gpu_processes is not None:
        _pd_devs = [d.strip() for d in visible.split(",") if d.strip()] or ["0"]
        _kill_stray_gpu_processes(cuda_devices=",".join(_pd_devs), grace_sec=2.0)
        if _wait_for_gpu_free_memory is not None:
            try:
                _pool_util = max(
                    float(topo.PRODUCER_GPU_UTIL),
                    float(topo.consumer_gpu_util(topo.pd_protocol(topology),
                                                 topology.consumer_gpu_util)),
                )
            except Exception:
                _pool_util = float(topo.PRODUCER_GPU_UTIL)
            try:
                _wait_for_gpu_free_memory(cuda_devices=",".join(_pd_devs),
                                          gpu_memory_utilization=_pool_util,
                                          timeout_sec=180.0)
            except TimeoutError as _e:
                print("[verify:" + side + "] [gpu-wait] WARN: " + str(_e)
                      + "; starting the pools anyway", flush=True)

    workload = wl
    if workload.get("name") != "pd_multiturn":
        raise SystemExit(
            "FATAL: a PD task routed workload " + repr(workload.get("name")) + " into the PD "
            "measure path; the scored workload must be named `pd_multiturn`."
        )

    user_turns = max(1, int(pd_meta.get("conversation_turns", 8)) // 2)
    need = (int(pd_meta.get("first_turn_prefix_tokens", 8000))
            + user_turns * (int(pd_meta["new_tokens_per_turn"])
                            + int(pd_meta["reply_tokens"])))
    ceiling = int(_pd_task_int("model", "context_window", need))
    max_len = min(ceiling, int(need * 1.25))
    engine_argv = [
        str(venv / "bin" / "python"), "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "0.0.0.0",
        "--tp-size", str(width),
        "--served-model-name", served,
        "--trust-remote-code",
        "--random-seed", "1024",
        "--context-length", str(max_len),
        "--max-running-requests", "256",
        "--chunked-prefill-size", str(int(pd_meta["chunked_prefill_size"])),
        "--dtype", "bfloat16",
        "--moe-runner-backend", "flashinfer_mxfp4",
    ]


    results: list[float] = []
    log_dir = raw_dir / ("pd_" + side)
    router = None
    try:
        router = topo.start(topology, engine_argv=engine_argv, venv=venv,
                            router_module=_pd_router_module(), log_dir=log_dir,
                            served_model_name=served,
                            env=_pd_engine_env(repo, venv, pythonpath_overlay,
                                               log_dir, side))
        topo.wait_ready(topology, served, timeout_s=float(server_boot_timeout))
        profiles = metrics.profile_points(workload)
        per_profile: dict[str, list[float]] = {p["name"]: [] for p in profiles}
        _pd_ttft_by_profile: dict[str, list[float]] = {}
        for index in range(warmup_runs + num_runs):
            is_warmup = index < warmup_runs
            label = "warmup" if is_warmup else "run" + str(index - warmup_runs)
            for prof in (profiles[:1] if is_warmup else profiles):
                plabel = label + "_" + prof["name"]
                stats = log_dir / ("stats_" + plabel + ".json")
                rc = _pd_run_bench(pd_meta=pd_meta, workload=workload, model_path=model_path, venv=venv,
                                   repo=repo, router_port=topology.router_port, stats_out=stats,
                                   log_path=log_dir / ("bench_" + plabel + ".log"),
                                   served_model_name=served, concurrency=int(prof["concurrency"]))
                topo.assert_pd_happened(topology)
                if is_warmup:
                    continue
                if rc != 0 or not stats.is_file():
                    print("[verify:" + side + "] " + plabel + " bench rc=" + str(rc)
                          + "; no sample recorded", flush=True)
                    continue
                extract = metrics.extract(stats, **metrics.expected_counts(pd_meta))
                per_profile[prof["name"]].append(_pd_score_from(extract))
                _pd_ttft_by_profile.setdefault(prof["name"], []).append(
                    float(extract.get("continuation_p95_ttft_ms")
                          or extract.get("continuation_mean_ttft_ms") or 0.0))
                print("[verify:" + side + "] " + plabel + " " + _PD_SCORED_KEY + " "
                      + ("%.1f" % extract[_PD_SCORED_KEY]) + " ms (n="
                      + str(extract["continuation_count"]) + ")", flush=True)
        results.extend(metrics.aggregate_profiles(per_profile, profiles))
        if any(_pd_ttft_by_profile.values()):
            _ttft_rounds = metrics.aggregate_profiles(_pd_ttft_by_profile, profiles)
            if _ttft_rounds:
                _PD_TAIL_TTFT_BY_SIDE[side] = _trimmed_mean(_ttft_rounds)
        for prof in profiles:
            _vals = per_profile[prof["name"]]
            print("[verify:" + side + "] " + prof["name"] + " n=" + str(len(_vals)) + " score(1000/ms)="
                  + str([round(v, 2) for v in _vals]) + " <- " + _PD_SCORED_KEY + " "
                  + str([round(1000.0 / v, 1) if v else None for v in _vals]) + " ms"
                  + ("" if prof.get("scored", True) else "  (measured, NOT scored)"), flush=True)
        print("[verify:" + side + "] aggregated over " + str(len(profiles)) + " point(s): score(1000/ms)="
              + str([round(v, 2) for v in results]), flush=True)
        topo.assert_pd_happened(topology)
    finally:
        topo.stop(topology, router)
    return results


_PD_PESSIMISTIC_TOKENS_PER_SEC = 500
_PD_REQUEST_TIMEOUT_FLOOR_SEC = 900


def _pd_request_timeout_sec(pd_meta: dict, workload: dict, concurrency: int) -> int:
    """Seconds to allow one conversation turn before calling it hung."""
    override = os.environ.get("INFB_PD_REQUEST_TIMEOUT_SEC", "").strip()
    if override:
        return int(override)
    prefix = int(pd_meta.get("first_turn_prefix_tokens", 8000))
    new_tokens = int(pd_meta["new_tokens_per_turn"])
    reply = int(pd_meta["reply_tokens"])
    worst_turn_tokens = (prefix + new_tokens + reply) * max(1, concurrency)
    derived = worst_turn_tokens // _PD_PESSIMISTIC_TOKENS_PER_SEC
    return max(_PD_REQUEST_TIMEOUT_FLOOR_SEC, derived)


def _pd_run_bench(*, pd_meta: dict, workload: dict, model_path: str, venv: Path, repo: Path,
                  router_port: int, stats_out: Path, log_path: Path,
                  served_model_name: str = "model", concurrency: int | None = None) -> int:
    """One multi-turn bench run against the router, writing per-request stats as JSON."""
    client = repo / "benchmarks" / "multi_turn" / "benchmark_serving_multi_turn.py"
    if not client.is_file():
        vendored = Path(__file__).resolve().parent / "_verifier_lib" / "multi_turn" \
            / "benchmark_serving_multi_turn.py"
        if vendored.is_file():
            client = vendored
        else:
            raise SystemExit(
                "FATAL: no multi-turn bench client. Looked in the engine tree at " + str(client)
                + " and at the vendored copy " + str(vendored) + ", which every PD task ships."
            )
    from _verifier_lib.pd_client_patch import patch_multi_turn_client  # noqa: PLC0415
    print("[pd] multi-turn client edits: " + patch_multi_turn_client(client), flush=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    spec = stats_out.parent / "conversations.json"
    import random as _rnd
    _convs = int(pd_meta.get("conversations", 16))
    _prefix_tokens = int(pd_meta.get("first_turn_prefix_tokens", 8000))
    _needed_tokens = _convs * _prefix_tokens + 1
    _words_per_line = 18
    _lines = max(24000, -(-(_needed_tokens * 3) // _words_per_line))
    _corpus = stats_out.parent / "corpus_generated.txt"
    if not _corpus.is_file():
        _r = _rnd.Random(int(os.environ.get("INFB_BENCH_SEED", "1024")))
        _words = [
            "system", "request", "response", "latency", "prefill", "decode", "token", "cache",
            "kernel", "tensor", "router", "engine", "memory", "throughput", "attention",
            "sequence", "batch", "schedule", "transfer", "gradient", "context", "window",
        ]
        _corpus.parent.mkdir(parents=True, exist_ok=True)
        with _corpus.open("w", encoding="utf-8") as _fh:
            for _ in range(_lines):
                _fh.write(" ".join(_r.choice(_words) for _ in range(_words_per_line)) + "." + chr(10))
        print("[pd] corpus generated " + str(_corpus) + " (" + str(_corpus.stat().st_size)
              + " bytes, " + str(_lines) + " lines, seeded) for " + str(_convs)
              + " conversations x " + str(_prefix_tokens) + " prefix tokens", flush=True)
    if _lines * _words_per_line < _needed_tokens:
        raise SystemExit(
            "FATAL: the generated PD corpus cannot cover the spec: " + str(_convs)
            + " conversations x " + str(_prefix_tokens) + " prefix tokens = " + str(_needed_tokens)
            + " tokens needed, corpus word floor is " + str(_lines * _words_per_line) + "."
        )
    text_files = [str(_corpus)]
    spec.write_text(json.dumps({
        "filetype": "generate_conversations",
        "num_conversations": int(pd_meta.get("conversations", 16)),
        "text_files": text_files,
        "print_stats": False,
        "prompt_input": {
            "num_turns": {"distribution": "constant",
                          "value": int(pd_meta.get("conversation_turns", 8))},
            "common_prefix_num_tokens": {"distribution": "constant", "value": 0},
            "prefix_num_tokens": {"distribution": "constant",
                                  "value": int(pd_meta.get("first_turn_prefix_tokens", 8000))},
            "num_tokens": {"distribution": "constant",
                           "value": int(pd_meta["new_tokens_per_turn"])},
        },
        "prompt_output": {
            "num_tokens": {"distribution": "constant",
                           "value": int(pd_meta["reply_tokens"])},
        },
    }, indent=2) + "\n", encoding="utf-8")

    if concurrency is None:
        concurrency = int(workload.get("max_concurrency", 16))
    concurrency = int(concurrency)
    argv = [
        str(venv / "bin" / "python"), str(client),
        "--model", model_path,
        "--served-model-name", served_model_name,
        "--url", "http://127.0.0.1:" + str(router_port),
        "--input-file", str(spec),
        "--num-clients", str(concurrency),
        "--max-active-conversations", str(concurrency),
        "--no-early-stop",
        "--request-timeout-sec", str(_pd_request_timeout_sec(pd_meta, workload, concurrency)),
        "--stats-json-output", str(stats_out),
    ]
    env = dict(os.environ)
    loop = "127.0.0.1,localhost"
    for key in ("no_" + "proxy", "NO_" + "PROXY"):
        env[key] = ",".join([x for x in [env.get(key, ""), loop] if x])
    with log_path.open("w") as handle:
        return subprocess.call(argv, cwd=str(client.parent), env=env,
                               stdout=handle, stderr=subprocess.STDOUT)


def _measure_side_serve(
    *,
    side: str,
    workloads: list[dict],
    model_path: str,
    venv: Path,
    repo: Path,
    pythonpath_overlay: bool,
    num_runs: int,
    warmup_runs: int,
    raw_dir: Path,
    server_boot_timeout: int = int(os.environ.get("INFB_SERVE_READY_TIMEOUT_SEC", "1800")),
    client_repo: Path | None = None
) -> dict[str, list[float]]:
    """Run one full side (candidate or pristine) under serve mode."""
    if _pd_metadata():
        raise SystemExit(
            "FATAL: _measure_side_serve reached on a task declaring [metadata.inference_bench.pd]. "
            "PD is measured by _measure_one_workload_pd, dispatched from main()'s per-workload "
            "selection. This path boots ONE aggregated server and would silently measure a "
            "different experiment."
        )
    port = _pick_free_port()
    log_path = raw_dir / f"{ENGINE}_serve_{side}_{port}.log"
    server = _start_engine_serve(
        model=model_path, venv=venv, repo=repo, port=port,
        pythonpath_overlay=pythonpath_overlay, log_path=log_path,
        apply_sglk_overlay=side.lower().startswith("cand"),
    )
    side_tps: dict[str, list[float]] = {w["name"]: [] for w in workloads}
    try:
        if not _wait_for_server_ready(port, server_boot_timeout, server):
            print(f"[serve] server failed to become ready (timeout={server_boot_timeout}s)",
                  flush=True)
            return side_tps
        print(f"[serve] server ready on :{port}", flush=True)

        for wl in workloads:
            for i in range(warmup_runs):
                r = _run_bench_serve(
                    workload=wl, model_path=model_path, venv=venv, port=port,
                    run_label=f"{side}_warmup{i}", raw_dir=raw_dir, warmup=True,
                    client_repo=client_repo,
                )
            for i in range(num_runs):
                r = _run_bench_serve(
                    workload=wl, model_path=model_path, venv=venv, port=port,
                    run_label=f"{side}_meas{i}", raw_dir=raw_dir,
                    client_repo=client_repo,
                )
                if r:
                    tps = _extract_tps(r)
                    if tps:
                        side_tps[wl["name"]].append(tps)
            print(f"  {wl['name']}: n={len(side_tps[wl['name']])} "
                  f"tps={side_tps[wl['name']]}", flush=True)
    finally:
        _stop_engine_serve(server)
        print(f"[serve] {side} server stopped", flush=True)
        _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").strip()
        for _dev in ([d for d in _vis_cleanup.split(",") if d.strip()] or ["0"]):
            _kill_stray_gpu_processes(cuda_device=_dev)
    return side_tps


def _measure_one_workload_serve(
    *,
    wl: dict,
    side: str,
    model_path: str,
    venv: Path,
    repo: Path,
    pythonpath_overlay: bool,
    num_runs: int,
    warmup_runs: int,
    raw_dir: Path,
    server_boot_timeout: int = int(os.environ.get("INFB_SERVE_READY_TIMEOUT_SEC", "1800")),
    client_repo: Path | None = None
) -> list[float]:
    """Per-workload serve PAIRED helper.  Boots a fresh vllm serve, runs
    warmup + meas for ONE workload, tears down + cleans GPU."""
    if _pd_metadata():
        raise SystemExit(
            "FATAL: _measure_one_workload_serve reached on a task declaring [metadata.inference_bench.pd]. "
            "PD is measured by _measure_one_workload_pd, dispatched from main()'s per-workload "
            "selection. This path boots ONE aggregated server and would silently measure a "
            "different experiment."
        )
    port = _pick_free_port()
    num_boots = max(1, int(os.environ.get(f"INFB_WORKLOAD_BOOTS_{wl['name'].upper()}", "1")))
    runs: list[float] = []
    for _boot in range(num_boots):
        _bs = "" if num_boots == 1 else f"_boot{_boot}"
        port = _pick_free_port()
        log_path = raw_dir / f"{ENGINE}_serve_{wl['name']}_{side}{_bs}_{port}.log"
        server = _start_engine_serve(
            model=model_path, venv=venv, repo=repo, port=port,
            pythonpath_overlay=pythonpath_overlay, log_path=log_path,
            apply_sglk_overlay=side.lower().startswith("cand"),
            )
        try:
            if not _wait_for_server_ready(port, server_boot_timeout, server):
                print(f"  [serve] {wl['name']}/{side}{_bs}: server not ready in {server_boot_timeout}s",
                      flush=True)
                continue
            print(f"  [serve] {wl['name']}/{side}{_bs}: server ready on :{port}", flush=True)

            for i in range(warmup_runs):
                _run_bench_serve(
                    workload=wl, model_path=model_path, venv=venv, port=port,
                    run_label=f"{wl['name']}_{side}{_bs}_warmup{i}", raw_dir=raw_dir,
                    warmup=True,
                    client_repo=client_repo,
                )
            for i in range(num_runs):
                r = _run_bench_serve(
                    workload=wl, model_path=model_path, venv=venv, port=port,
                    run_label=f"{wl['name']}_{side}{_bs}_meas{i}", raw_dir=raw_dir,
                    client_repo=client_repo,
                )
                if r:
                    score = _compute_primary_score(r, wl["name"])
                    if score:
                        runs.append(score)
        finally:
            _stop_engine_serve(server)
            _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").strip()
            for _dev in ([d for d in _vis_cleanup.split(",") if d.strip()] or ["0"]):
                _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=5.0)
        if num_boots > 1 and _boot < num_boots - 1:
            time.sleep(10)
    return runs


def _measure_one_workload_serve_multi_profile(
    *,
    wl: dict,
    side: str,
    model_path: str,
    venv: Path,
    repo: Path,
    pythonpath_overlay: bool,
    num_runs: int,
    warmup_runs: int,
    raw_dir: Path,
    server_boot_timeout: int = int(os.environ.get("INFB_SERVE_READY_TIMEOUT_SEC", "1800")),
    client_repo: Path | None = None
) -> list[float]:
    """Multi-profile per-workload serve helper (high_load and the like).
    Boots ONE vllm serve, runs warmup + meas for EACH profile
    listed in `wl['profiles']` back-to-back against the same server, then
    tears down."""
    if _pd_metadata():
        raise SystemExit(
            "FATAL: _measure_one_workload_serve_multi_profile reached on a task declaring [metadata.inference_bench.pd]. "
            "PD is measured by _measure_one_workload_pd, dispatched from main()'s per-workload "
            "selection. This path boots ONE aggregated server and would silently measure a "
            "different experiment."
        )
    profiles = wl.get("profiles") or []
    if not profiles:
        raise ValueError(
            f"{wl['name']!r}: multi-profile path called without 'profiles' field"
        )

    num_boots = max(1, int(os.environ.get(f"INFB_WORKLOAD_BOOTS_{wl['name'].upper()}", "1")))
    per_profile_runs: dict[str, list[float]] = {p["name"]: [] for p in profiles}
    for _boot in range(num_boots):
        _bs = "" if num_boots == 1 else f"_boot{_boot}"
        port = _pick_free_port()
        log_path = raw_dir / f"{ENGINE}_serve_{wl['name']}_{side}{_bs}_{port}.log"
        server = _start_engine_serve(
            model=model_path, venv=venv, repo=repo, port=port,
            pythonpath_overlay=pythonpath_overlay, log_path=log_path,
            apply_sglk_overlay=side.lower().startswith("cand"),
            )
        try:
            if not _wait_for_server_ready(port, server_boot_timeout, server):
                print(f"  [serve-mp] {wl['name']}/{side}{_bs}: server not ready in {server_boot_timeout}s",
                      flush=True)
                continue
            print(f"  [serve-mp] {wl['name']}/{side}{_bs}: server ready on :{port}", flush=True)

            for i in range(warmup_runs):
                _run_bench_serve(
                    workload=wl, model_path=model_path, venv=venv, port=port,
                    run_label=f"{wl['name']}_{side}{_bs}_warmup{i}", raw_dir=raw_dir,
                    warmup=True,
                    profile=profiles[0],
                    client_repo=client_repo,
                )
            for i in range(num_runs):
                for prof in profiles:
                    pname = prof["name"]
                    r = _run_bench_serve(
                        workload=wl, model_path=model_path, venv=venv, port=port,
                        run_label=f"{wl['name']}_{side}{_bs}_{pname}_meas{i}", raw_dir=raw_dir,
                        profile=prof,
                        client_repo=client_repo,
                    )
                    if r:
                        score = _compute_primary_score(r, wl["name"])
                        if score:
                            per_profile_runs[pname].append(score)
        finally:
            _stop_engine_serve(server)
            _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").strip()
            for _dev in ([d for d in _vis_cleanup.split(",") if d.strip()] or ["0"]):
                _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=5.0)
        if num_boots > 1 and _boot < num_boots - 1:
            time.sleep(10)

    aggregated: list[float] = []
    n_meas = min((len(v) for v in per_profile_runs.values()), default=0)
    for i in range(n_meas):
        scores_at_i = [per_profile_runs[p["name"]][i] for p in profiles]
        if all(s > 0 for s in scores_at_i):
            log_sum = sum(math.log(s) for s in scores_at_i)
            aggregated.append(math.exp(log_sum / len(scores_at_i)))

    wl.setdefault("_runtime_per_profile_runs", {})[side] = per_profile_runs
    return aggregated








def _dump_failed_compare_attempt(raw_dir, attempt, result):
    """Persist a crashed correctness-compare attempt's full output; return the path."""
    try:
        target = Path(raw_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"correctness_compare_attempt{attempt}.log"
        path.write_text(
            f"[infb] correctness compare attempt {attempt} "
            f"rc={getattr(result, 'returncode', None)}\n"
            f"===== stdout =====\n{getattr(result, 'stdout', '') or ''}\n"
            f"===== stderr =====\n{getattr(result, 'stderr', '') or ''}\n"
        )
        return str(path)
    except Exception as exc:
        return f"<dump failed: {exc}>"


def _run_correctness(
    *,
    prompts_json: Path,
    ref_npz: Path,
    model_path: str,
    venv: Path,
    repo: Path,
    raw_dir: Path,
    atol: float = 0.05,
    rtol: float = 0.05,
    min_pass_rate: float = 0.95,
    max_tokens: int = 100,
) -> dict:
    out_file = raw_dir / "correctness_result.json"
    runner   = THIS_DIR / "_candidate_runner.py"
    python   = venv / "bin" / "python"

    cmd = [str(python), "-P", str(runner),
           "correctness",
           "--prompts-json", str(prompts_json),
           "--ref-npz",      str(ref_npz),
           "--out-json",     str(out_file),
           "--model",        model_path,
           "--tp",           str(TP_SIZE),
           "--max-tokens",   str(max_tokens),
           "--atol",         str(atol),
           "--rtol",         str(rtol),
           "--min-pass-rate", str(min_pass_rate)]

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    _set_jit_cache_env(env, raw_dir, "cand")
    pp_parts = [str(repo)]
    if (fi_overlay := _flashinfer_overlay_path()) is not None:
        pp_parts.insert(0, fi_overlay)
    env["PYTHONPATH"] = ":".join(pp_parts)
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    _set_offline_env(env)
    env.setdefault("CUDA_VISIBLE_DEVICES", _default_visible_devices())
    _ac = env.get("INFB_ARTIFACT_CACHE")
    _apply_nccl_compat(env, (f"{_ac}/builds/vllm/dsv4_flash_pd_8xb200/nccl/libnccl_cu12.so.2"
                             if _ac else None))
    import json as _json
    env["INFB_CORRECTNESS_LLM_KWARGS_JSON"] = _json.dumps(_correctness_llm_kwargs(model_path))
    _force_spawn_workers(env)

    print("[correctness] running...", flush=True)
    _corr_retries = int(os.environ.get("INFB_CORRECTNESS_RETRIES", "2"))
    _lr_timeout = int(os.environ.get("INFB_LIVE_REF_TIMEOUT_SEC", "3600"))
    result = None
    t0 = time.time()
    _determinism = None
    for _attempt in range(_corr_retries + 1):
        if _attempt:
            if "/health not ready in" in (getattr(result, "stderr", "") or ""):
                print("  [correctness] NOT retrying: the candidate server missed the start limit "
                      "(INFB_SERVE_READY_TIMEOUT_SEC).", flush=True)
                break
            _determinism = _deterministic_signature(raw_dir)
            if _determinism:
                print(f"  [correctness] NOT retrying: {_determinism['signature']} is a known deterministic "
                      f"failure ({_determinism['meaning']}). {_determinism['remedy']}", flush=True)
                break
            _dump = _dump_failed_compare_attempt(raw_dir, _attempt, result)
            _prev_err = (getattr(result, "stderr", "") or "")[-1500:]
            print(f"  [correctness] compare attempt {_attempt}/{_corr_retries}: prior "
                  f"attempt exited rc={result.returncode} with no result — retrying. "
                  f"full output: {_dump}\n  prior stderr tail:\n{_prev_err}", flush=True)
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                    start_new_session=True, timeout=_lr_timeout)
        except subprocess.TimeoutExpired:
            return {"correctness_ok": False, "error": "timeout"}
        if out_file.exists() or result.returncode == 0:
            break
    elapsed = time.time() - t0
    if result.returncode != 0 and not out_file.exists():
        print(f"  [correctness] final attempt failed; full output: "
              f"{_dump_failed_compare_attempt(raw_dir, 'final', result)}", flush=True)
    print(f"  rc={result.returncode}  elapsed={elapsed:.1f}s", flush=True)

    if out_file.exists():
        try:
            data = json.loads(out_file.read_text())
            if "overall_pass_rate" in data:
                if result.returncode != 0:
                    print(f"  (rc={result.returncode} but out_file present + valid; "
                          f"likely V1 worker cleanup ABORT — trusting file)",
                          flush=True)
                return data
        except Exception:
            pass

    if result.returncode != 0:
        _out = {"correctness_ok": False, "error": result.stderr[-2000:]}
        if _determinism:
            _out["deterministic"] = _determinism
        return _out

    if not out_file.exists():
        return {"correctness_ok": False,
                "error": "rc=0 but out_file missing: " + result.stderr[-1500:]}

    try:
        return json.loads(out_file.read_text())
    except Exception as e:
        return {"correctness_ok": False, "error": str(e)}




def _load_task_correctness_config(task_dir=None):
    """Returns the per-task calibrated atol/rtol/min_pass_rate from the
    module-level CORRECTNESS_CALIBRATION constant (baked in from
    task.toml [metadata.inference_bench.correctness]).

    Why baked-in: the verifier container cannot reliably read task.toml at
    runtime, because the task is materialised in a location where task.toml is
    not a sibling of tests/. Reading at runtime silently fell through to
    defaults.

    task.toml remains the human-facing source of truth. The constant below is
    the same data baked in for runtime reliability. If you update task.toml,
    also update the constant."""
    cfg = dict(CORRECTNESS_CALIBRATION)
    for _k, _env_key in (("atol",          "INFB_TASK_CORRECTNESS_ATOL"),
                         ("rtol",          "INFB_TASK_CORRECTNESS_RTOL"),
                         ("min_pass_rate", "INFB_TASK_CORRECTNESS_MIN_PASS_RATE")):
        _v = os.environ.get(_env_key)
        if _v is not None:
            try:
                cfg[_k] = float(_v)
            except ValueError:
                pass
    return cfg


def _correctness_llm_kwargs(model_path: str) -> dict:
    kwargs = {
        "model":                   model_path,
        "tensor_parallel_size":    TP_SIZE,
        "dtype":                   "bfloat16",
        "trust_remote_code":       True,
        "max_model_len":           4096,
        "gpu_memory_utilization":  float(os.environ.get("INFB_CORRECTNESS_GPU_MEM_UTIL", "0.85")),
        "enforce_eager":           False,
        "seed":                    0,
        "kv_cache_dtype":          globals().get("KV_CACHE_DTYPE", "auto"),
    }
    max_num_seqs = globals().get("CORRECTNESS_MAX_NUM_SEQS")
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    block_size = globals().get("KV_BLOCK_SIZE")
    if block_size is not None:
        kwargs["block_size"] = int(block_size)
    return kwargs


def _resolve_correctness(mode: str) -> dict:
    """Single on/off switch for the correctness gate (unit-tested)."""
    run = (mode != "disabled")
    return {"run": run, "ref": "live" if run else None, "fail_missing": False}


def _resolve_live_prompts(this_dir: Path):
    """Locate the prompt bank for live reference generation."""
    if env := os.environ.get("INFB_CORRECTNESS_PROMPTS"):
        p = Path(env)
        if p.exists():
            return p
    local = this_dir / "correctness_prompts.json"
    if local.exists():
        return local
    return None


def _npz_arrays_present(path, required=("token_ids", "logprobs", "prompt_token_ids")):
    """Stdlib-only (NO numpy) check that an .npz holds the required arrays."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except Exception as e:
        return False, f"not a readable npz: {e}"
    missing = {f"{r}.npy" for r in required} - names
    if missing:
        return False, f"missing {sorted(missing)}; has {sorted(names)}"
    return True, ",".join(sorted(names))


def _gen_live_reference(
    *,
    prompts_json: Path,
    out_npz: Path,
    model_path: str,
    venv: Path,
    repo: Path,
    workdir: Path,
    max_tokens: int = 100,
) -> dict:
    """Generate the teacher-forced reference LIVE from pristine code."""
    if disposable_pristine_worktree is None:
        return {"ok": False, "error": "disposable_pristine_worktree unavailable (helper not overlaid)"}

    runner = THIS_DIR / "_candidate_runner.py"
    python = venv / "bin" / "python"
    workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        with disposable_pristine_worktree(
            repo, workdir=workdir, label="_correctness_ref_worktree",
        ) as pris_repo:
            cmd = [str(python), "-P", str(runner),
                   "gen_reference",
                   "--prompts-json", str(prompts_json),
                   "--out-npz",      str(out_npz),
                   "--model",        model_path,
                   "--tp",           str(TP_SIZE),
                   "--max-tokens",   str(max_tokens)]
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
            _set_jit_cache_env(env, Path(out_npz).parent, "pris")
            pp_parts = [str(pris_repo)]
            if (fi_overlay := _flashinfer_overlay_path()) is not None:
                pp_parts.insert(0, fi_overlay)
            env["PYTHONPATH"] = ":".join(pp_parts)
            env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
            if ENGINE == "sglang":
                env.pop("HF_HUB_OFFLINE", None)
                env.pop("TRANSFORMERS_OFFLINE", None)
            else:
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
            env.setdefault("CUDA_VISIBLE_DEVICES", _default_visible_devices())
            _ac = env.get("INFB_ARTIFACT_CACHE")
            _apply_nccl_compat(env, (f"{_ac}/builds/vllm/dsv4_flash_pd_8xb200/nccl/libnccl_cu12.so.2"
                                     if _ac else None))
            import json as _json
            env["INFB_CORRECTNESS_LLM_KWARGS_JSON"] = _json.dumps(_correctness_llm_kwargs(model_path))
            _force_spawn_workers(env)
            print(f"[live-ref] generating reference from pristine worktree {pris_repo}", flush=True)
            _boot_retries = int(os.environ.get("INFB_LIVE_REF_BOOT_RETRIES", "3"))
            result = None
            for _attempt in range(_boot_retries + 1):
                if _attempt:
                    print(f"  [live-ref] golden-gen boot RETRY {_attempt}/{_boot_retries} "
                          f"(prev rc={result.returncode}, npz not written — transient "
                          f"EngineCore boot kill; re-attempting)", flush=True)
                _lr_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").strip()
                _lr_devs = [d for d in _lr_visible.split(",") if d.strip()] or ["0"]
                for _lr_dev in _lr_devs:
                    _kill_stray_gpu_processes(cuda_device=_lr_dev, grace_sec=2.0)
                    try:
                        _wait_for_gpu_free_memory(cuda_device=_lr_dev, required_gib=60.0, timeout_sec=180.0)
                    except TimeoutError as _lr_e:
                        print(f"[live-ref][gpu-wait] WARN: {_lr_e}; proceeding with reference spawn anyway", flush=True)
                result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                        start_new_session=True, timeout=int(os.environ.get("INFB_LIVE_REF_TIMEOUT_SEC", "3600")))
                if out_npz.exists() or result.returncode == 0:
                    break
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"ok": False, "error": f"worktree/spawn: {e}", "elapsed_sec": time.time() - t0}
    elapsed = time.time() - t0
    print(f"  [live-ref] rc={result.returncode} elapsed={elapsed:.1f}s", flush=True)

    try:
        (out_npz.parent / "live_reference_gen.log").write_text(
            f"rc={result.returncode}\n--- STDOUT ---\n{result.stdout or ''}\n"
            f"--- STDERR ---\n{result.stderr or ''}")
    except Exception:
        pass

    if out_npz.exists():
        _ok, _detail = _npz_arrays_present(out_npz)
        if _ok:
            if result.returncode != 0:
                print(f"  (rc={result.returncode} but npz present + valid — trusting file)", flush=True)
            return {"ok": True, "out_npz": str(out_npz), "elapsed_sec": elapsed}
        return {"ok": False, "error": f"npz unreadable: {_detail}",
                "elapsed_sec": elapsed, "stderr_tail": result.stderr[-4000:]}
    return {"ok": False, "error": "npz not written: " + result.stderr[-4000:],
            "elapsed_sec": elapsed}



def _load_task_scoring_config(task_dir=None):
    """Returns per-task calibrated scoring config from the module-level
    SCORING_CALIBRATION constant (baked in from task.toml [metadata.inference_bench.scoring]).
    Verifier container can\'t reliably read task.toml at runtime — see comment in
    _load_task_correctness_config / _paired_helper.py.
    Env var INFB_*_OVERRIDE still wins for num_runs/warmup_runs."""
    import os
    cfg = dict(SCORING_CALIBRATION)
    for _k, _env_key, _conv in (
        ("num_runs",                "INFB_TASK_SCORING_NUM_RUNS",               int),
        ("warmup_runs",             "INFB_TASK_SCORING_WARMUP_RUNS",            int),
    ):
        _v = os.environ.get(_env_key)
        if _v is not None:
            try:
                cfg[_k] = _conv(_v)
            except ValueError:
                pass
    if (_v := os.environ.get("INFB_NUM_RUNS_OVERRIDE")) is not None:
        cfg["num_runs"] = int(_v)
    if (_v := os.environ.get("INFB_WARMUP_RUNS_OVERRIDE")) is not None:
        cfg["warmup_runs"] = int(_v)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace",  required=True)
    parser.add_argument("--tests-dir",  required=True)
    parser.add_argument("--reward",     required=True)
    parser.add_argument("--metrics",    required=True)
    parser.add_argument("--raw-dir",    required=True)
    args = parser.parse_args()

    workspace  = Path(args.workspace).resolve()
    task_dir   = Path(args.tests_dir).resolve().parent
    _scoring_cfg = _load_task_scoring_config(task_dir)
    repo       = workspace / "repo"
    raw_dir    = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("[real] stack rlimit for JIT: " + _raise_stack_limit_for_jit(), flush=True)
    print("[real] flashinfer CUtensorMap alignment: "
          + _patch_flashinfer_underaligned_tensormap(), flush=True)

    metrics: dict = {"verifier_mode": "real_engine"}
    _preflight_void = run_preflight(
        workspace=workspace,
        task_dir=task_dir,
        model_id=MODEL_ID,
        expected_gpu_types=EXPECTED_GPU_TYPES,
        expected_gpu_count=EXPECTED_GPU_COUNT,
        find_model=_find_model,
        find_venv=_find_venv,
        metrics=metrics,
    )
    if _preflight_void is not None:
        metrics["infra_void"] = _preflight_void
        print(
            f"[real] INFRA-VOID preflight={_preflight_void.get('reason')} "
            f"-> RE-RUN (not scored 0): {_preflight_void}",
            flush=True,
        )
        # metrics.json for forensics, and deliberately NO reward.json: a reward
        # file — even {"reward": 0.0, "infra_void": 1} — is BANKED as a score by
        # any runner that does not know the infra_void convention, which is every
        # public one. Withholding it makes Pier raise RewardFileNotFoundError, so
        # the void is never banked as a model score: on the public path a TERMINAL
        # error needing a manual re-queue (RewardFileNotFoundError is in Pier's
        # default RetryConfig.exclude_exceptions, which beats include_exceptions,
        # and max_retries defaults to 0).
        # The non-zero exit propagates through tests/test.sh (`set -euo pipefail`).
        write_void_metrics(args.metrics, metrics, reward_path=args.reward)
        return PREFLIGHT_VOID_EXIT_CODE
    t_start = time.time()

    try:
        venv = _find_venv(task_dir)
    except RuntimeError as e:
        _fail(args, metrics, str(e), "venv_missing")
        return 0

    model_path = _find_model(workspace)
    metrics["model_path"] = model_path
    metrics["venv"] = str(venv)

    _correctness_mode   = os.environ.get("INFB_CORRECTNESS_MODE", "disabled")
    _corr_plan = _resolve_correctness(_correctness_mode)
    metrics["correctness_plan"] = {
        **_corr_plan,
        "mode": _correctness_mode,
    }
    if not _corr_plan["run"]:
        print("  [correctness] mode=disabled → skipping", flush=True)

    _task_wl = Path(args.tests_dir) / "workloads.json"
    if not _task_wl.exists():
        _fail(args, metrics, "workloads.json missing", "workloads_missing")
        return 0
    metrics["workloads_source"] = str(_task_wl)
    workloads = json.loads(_task_wl.read_text())
    if isinstance(workloads, dict) and "workloads" in workloads:
        workloads = workloads["workloads"]

    _only_wl = os.environ.get("INFB_ONLY_WORKLOADS", "").strip()
    if _only_wl:
        _keep = {n.strip() for n in _only_wl.split(",") if n.strip()}
        _b4 = [w["name"] for w in workloads]
        workloads = [w for w in workloads if w.get("name") in _keep]
        print(f"[verifier] INFB_ONLY_WORKLOADS={_only_wl} -> {[w['name'] for w in workloads]} (from {_b4})", flush=True)
    metrics["num_workloads"] = len(workloads)


    if not repo.exists() or not (repo / ".git").exists():
        _fail(args, metrics, f"workspace/repo/ missing or not a git tree", "repo_missing")
        return 0

    try:
        _capture_record = capture_untracked_candidate(repo)
    except CandidateCaptureError as _capture_exc:
        metrics["infra_void"] = capture_void_reason(_capture_exc)
        print(
            "[capture] INFRA VOID: could not capture the candidate's untracked "
            f"files -> RE-RUN (not scored 0): {metrics['infra_void']}",
            flush=True,
        )
        # metrics.json only, NO reward.json — a reward file is banked as a score by a runner
        # that does not know the infra_void convention; non-zero exit -> RE-RUN, not a 0.
        write_void_metrics(args.metrics, metrics, reward_path=args.reward)
        return PREFLIGHT_VOID_EXIT_CODE
    metrics["candidate_capture"] = _capture_record
    diff_classification = classify_repo_diff(repo)
    kind = diff_classification.get("kind", "unknown")
    metrics["classification"] = diff_classification
    metrics["classification_kind"] = kind
    _agent_notes_record = collect_agent_notes(repo, raw_dir / "agent_notes")
    metrics["agent_notes"] = _agent_notes_record

    if kind == "empty":
        _fail(args, metrics, "no patch applied", "no_patch")
        return 0

    if kind in NARROW_REJECTED_KINDS:
        _write_results(
            args,
            rewards={"reward": 0.0, "correctness": 1.0, "performance": 0.0},
            metrics={**metrics, "reason": f"category_gate_rejected_{kind}",
                     "category_gate_rejected": True},
        )
        return 0

    if not _corr_plan["run"]:
        print("\n=== Correctness check === SKIPPED (INFB_CORRECTNESS_MODE=disabled)", flush=True)
        correctness_ok = True
        metrics["correctness_result"] = {"correctness_ok": True, "skipped": True,
                                         "plan": _corr_plan}
        correctness_result = metrics["correctness_result"]
    else:
        _corr_cfg = _load_task_correctness_config(task_dir)
        metrics["correctness_calibration"] = _corr_cfg
        print(f"\n=== Correctness check (reference={_corr_plan['ref']}) ===", flush=True)
        if _corr_plan["ref"] == "live" and not LIVE_CORRECTNESS_SUPPORTED:
            print(f"  [correctness] LIVE_CORRECTNESS_SUPPORTED=False — skipping live ref",
                  flush=True)
            metrics["correctness_result"] = {"correctness_ok": True, "skipped": True,
                                              "reason": "live_correctness_unsupported"}
            correctness_result = metrics["correctness_result"]
            correctness_ok = True
        else:
            if _corr_plan["ref"] == "live":
                _live_prompts = _resolve_live_prompts(THIS_DIR)
                if _live_prompts is None:
                    _fail(args, metrics,
                          "live correctness requested but no prompt bank found "
                          "(INFB_CORRECTNESS_PROMPTS / tests/correctness_prompts.json)",
                          "live_prompts_missing")
                    return 0
                _live_ref = raw_dir / "live_reference.npz"
                _gen_meta = _gen_live_reference(
                    prompts_json=_live_prompts,
                    out_npz=_live_ref,
                    model_path=model_path,
                    venv=venv,
                    repo=repo,
                    workdir=(raw_dir.parent if raw_dir is not None
                             else repo.parent / "_verifier_workdir"),
                )
                metrics["live_reference"] = _gen_meta
                if not _gen_meta.get("ok"):
                    metrics["infra_void"] = {
                        "reason": "live_reference_failed",
                        "note": (
                            "the PRISTINE correctness reference engine failed to boot -> "
                            "RE-RUN, do not score 0 against the model"
                        ),
                        "detail": _gen_meta.get("error"),
                        "elapsed_sec": _gen_meta.get("elapsed_sec"),
                    }
                    print(
                        "[real] INFRA-VOID live_reference_failed -> RE-RUN (not scored 0): "
                        f"{_gen_meta.get('error')}",
                        flush=True,
                    )
                    # metrics.json only, NO reward.json — a reward file is banked as a score by a
                    # runner that does not know the infra_void convention; non-zero exit -> RE-RUN,
                    # not a 0.
                    write_void_metrics(args.metrics, metrics, reward_path=args.reward)
                    return PREFLIGHT_VOID_EXIT_CODE
                prompts_json = _live_prompts
                ref_npz = _live_ref
                if _kill_stray_gpu_processes is not None:
                    _vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").strip()
                    for _dev in ([d for d in _vis.split(",") if d.strip()] or ["0"]):
                        _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=2.0)
                        if _wait_for_gpu_free_memory is not None:
                            try:
                                _wait_for_gpu_free_memory(cuda_device=_dev,
                                                          required_gib=60.0, timeout_sec=120.0)
                            except TimeoutError as _e:
                                print(f"[gpu-wait] WARN: {_e}; proceeding with candidate correctness", flush=True)
            correctness_result = _run_correctness(
                prompts_json=prompts_json,
                ref_npz=ref_npz,
                model_path=model_path,
                venv=venv,
                repo=repo,
                raw_dir=raw_dir,
                atol=_corr_cfg["atol"],
                rtol=_corr_cfg["rtol"],
                min_pass_rate=_corr_cfg["min_pass_rate"],
            )
            metrics["correctness_result"] = correctness_result
            correctness_ok = correctness_result.get("correctness_ok", False)
    correctness_ok = correctness_result.get("correctness_ok", False)
    print(f"  correctness_ok={correctness_ok}  "
          f"pass_rate={correctness_result.get('overall_pass_rate', 'N/A')}", flush=True)
    if not correctness_ok and correctness_result.get("error"):
        print(f"  [correctness] failure detail (compare error / stderr tail): "
              f"{str(correctness_result.get('error'))[-1500:]}", flush=True)

    if not correctness_ok:
        bundle = compute_reward(
            speedup_per_workload={},
            correctness_passed=False,
        )
        bundle["metrics"].update(metrics)
        bundle["metrics"]["reason"] = "correctness_failed"
        _write_results(args, rewards=bundle["rewards"], metrics=bundle["metrics"])
        return 0

    bench_mode = "serve"
    print(f"\n=== Throughput measurement (bench_mode={bench_mode}, mode=per-workload PAIRED) ===", flush=True)
    num_runs    = _scoring_cfg["num_runs"]
    warmup_runs = _scoring_cfg["warmup_runs"]

    candidate_tps: dict[str, list[float]] = {w["name"]: [] for w in workloads}
    pristine_tps:  dict[str, list[float]] = {w["name"]: [] for w in workloads}

    saved_HEAD: str | None = None
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            saved_HEAD = r.stdout.strip()
            print(f"  saved candidate HEAD: {saved_HEAD[:12]}", flush=True)
    except Exception as e:
        print(f"  WARN: could not capture HEAD: {e}", flush=True)

    _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").strip()
    _cuda_devs_for_cleanup = [d for d in _vis_cleanup.split(",") if d.strip()] or ["0"]
    cuda_dev = _cuda_devs_for_cleanup[0]

    def _bench_one_workload(wl: dict, side_label: str, side_repo: Path) -> list[float]:
        """Run warmup + meas for a single workload + side."""
        wl_tps: list[float] = []
        _wlk = wl["name"].upper()
        _wl_declared = _declared_runs_for(wl["name"], num_runs)
        wl_num_runs = int(os.environ.get(f"INFB_WORKLOAD_NUM_RUNS_{_wlk}", _wl_declared))
        wl_warmup   = int(os.environ.get(f"INFB_WORKLOAD_WARMUP_RUNS_{_wlk}", warmup_runs))
        if _pd_metadata():
            return _measure_one_workload_pd(
                wl=wl, side=side_label, model_path=model_path,
                venv=venv, repo=side_repo, pythonpath_overlay=True,
                num_runs=wl_num_runs, warmup_runs=wl_warmup, raw_dir=raw_dir,
                client_repo=_worktree_for_pristine,
            )
        if wl.get("profiles"):
            return _measure_one_workload_serve_multi_profile(
                wl=wl, side=side_label, model_path=model_path,
                venv=venv, repo=side_repo, pythonpath_overlay=True,
                num_runs=wl_num_runs, warmup_runs=wl_warmup, raw_dir=raw_dir,
                client_repo=_worktree_for_pristine,
            )
        return _measure_one_workload_serve(
            wl=wl, side=side_label, model_path=model_path,
            venv=venv, repo=side_repo, pythonpath_overlay=True,
            num_runs=wl_num_runs, warmup_runs=wl_warmup, raw_dir=raw_dir,
            client_repo=_worktree_for_pristine,
        )

    _csredit_install_prefix = None
    if prepare_csredit_rebuild_if_enabled is not None:
        try:
            _csredit_install_prefix = prepare_csredit_rebuild_if_enabled(
                task_dir=task_dir, workspace=workspace, repo=repo,
                venv=venv, raw_dir=raw_dir,
            )
        except Exception as _e:
            metrics["csredit_rebuild"] = f"failed: {_e}"
        if _csredit_install_prefix is not None:
            metrics["csredit_rebuild"] = "ok"
            metrics["csredit_install_prefix"] = str(_csredit_install_prefix)
            if swap_so_to_venv is not None:
                swap_so_to_venv(repo, venv=venv)

    global _SGLK_OVERLAY
    _SGLK_OVERLAY = _maybe_build_sgl_kernel(
        workspace=workspace, repo=repo, venv=venv, raw_dir=raw_dir, metrics=metrics,
    )

    if disposable_pristine_worktree is None:
        raise RuntimeError(
            "[verifier] PAIRED requires _paired_helper "
            "(it has to be present in tests/_verifier_lib/ beside this verifier)"
        )
    _worktree_workdir = raw_dir.parent if raw_dir is not None else (repo.parent / "_verifier_workdir")
    _worktree_workdir.mkdir(parents=True, exist_ok=True)
    with disposable_pristine_worktree(repo, workdir=_worktree_workdir) as _worktree_for_pristine:
        _exp_prewarm = os.environ.get("INFB_PAIRED_PREWARM", "1") == "1"
        _exp_prewarm_2side = os.environ.get("INFB_PAIRED_PREWARM_2SIDE", "1") == "1"
        if _exp_prewarm:
            def _do_prewarm(prewarm_repo: Path, label: str) -> None:
                if cleanup_stale_ipc is not None:
                    cleanup_stale_ipc(verbose=True)
                _port = _pick_free_port()
                _log = raw_dir / f"{ENGINE}_serve_{label}_{_port}.log"
                if os.environ.get("INFB_PAIRED_KEEP_SERVER_LOGS") == "1":
                    _kdir = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "paired_server_logs"
                    _kdir.mkdir(parents=True, exist_ok=True)
                    _log = _kdir / f"{ENGINE}_serve_{label}_{_port}.log"
                print(f"[verifier-prewarm] starting throwaway vllm serve on :{_port} (repo={prewarm_repo})", flush=True)
                _server = _start_engine_serve(
                    model=model_path, venv=venv, repo=prewarm_repo, port=_port,
                    pythonpath_overlay=True, log_path=_log,
                    apply_sglk_overlay=("cand" in label),
                )
                _t0 = time.time()
                try:
                    if _wait_for_server_ready(_port, int(os.environ.get("INFB_SERVE_READY_TIMEOUT_SEC", "2400")), _server):
                        print(f"[verifier-prewarm] {label} ready in {time.time()-_t0:.1f}s; killing "
                              f"(disk caches now populated)", flush=True)
                    else:
                        print(f"[verifier-prewarm] {label} WARN: server didn't reach ready; proceeding", flush=True)
                finally:
                    _stop_engine_serve(_server)
                    for _dev in _cuda_devs_for_cleanup:
                        _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=2.0)
                    time.sleep(8)
                    if cleanup_stale_ipc is not None:
                        cleanup_stale_ipc(verbose=True)
            print(f"[verifier] EXPERIMENT: INFB_PAIRED_PREWARM=1 — throwaway init spawn(s) "
                  f"to populate disk caches "
                  f"({'BOTH cand+pris paths' if _exp_prewarm_2side else 'cand path only'})", flush=True)
            _do_prewarm(prewarm_repo=repo, label="prewarm_cand")
            if _exp_prewarm_2side:
                _do_prewarm(prewarm_repo=_worktree_for_pristine, label="prewarm_pris")

        _arm_order = os.environ.get("INFB_PD_ARM_ORDER", "cand_first").strip().lower()
        if _arm_order not in ("cand_first", "pristine_first"):
            raise SystemExit(
                f"FATAL: INFB_PD_ARM_ORDER={_arm_order!r} is not one of cand_first, pristine_first. "
                "Refusing rather than silently measuring the default order under an experiment label."
            )

        def _bench_cand(wl: dict) -> None:
            wl_name = wl["name"]
            print(f"\n--- {wl_name}: candidate side ---", flush=True)
            if cleanup_stale_ipc is not None:
                cleanup_stale_ipc(verbose=True)
            if _csredit_install_prefix is not None and swap_so_to_built is not None:
                swap_so_to_built(repo, _csredit_install_prefix)
            cand_runs = _bench_one_workload(wl, "cand", repo)
            candidate_tps[wl_name] = cand_runs
            print(f"  {wl_name}/cand: n={len(cand_runs)} tps={cand_runs}", flush=True)

        def _bench_pristine(wl: dict) -> None:
            wl_name = wl["name"]
            print(f"--- {wl_name}: pristine side (in worktree at inference_base) ---", flush=True)
            if cleanup_stale_ipc is not None:
                cleanup_stale_ipc(verbose=True)
            pris_runs = _bench_one_workload(wl, "pristine", _worktree_for_pristine)
            pristine_tps[wl_name] = pris_runs
            print(f"  {wl_name}/pris: n={len(pris_runs)} tps={pris_runs}", flush=True)

        metrics["pd_arm_order"] = _arm_order
        for wl in workloads:
            print(f"[verifier] arm order: {_arm_order}", flush=True)
            if _arm_order == "pristine_first":
                _bench_pristine(wl)
                _bench_cand(wl)
            else:
                _bench_cand(wl)
                _bench_pristine(wl)


    if _csredit_install_prefix is not None and swap_so_to_venv is not None:
        swap_so_to_venv(repo, venv=venv)

    def _pool_estimate(_vals, _wl):
        _vals = [v for v in _vals if v and v > 0]
        if not _vals:
            return _trimmed_mean(_vals)
        _est = os.environ.get(f"INFB_BENCH_ESTIMATOR_{_wl.upper()}", "trimmed_mean").strip().lower()
        if _est == "median":
            return statistics.median(_vals)
        if _est == "mean":
            return statistics.fmean(_vals)
        return _trimmed_mean(_vals)
    candidate_mean_tps = {
        name: _pool_estimate(vals, name)
        for name, vals in candidate_tps.items()
    }
    pristine_mean_tps = {
        name: _pool_estimate(vals, name)
        for name, vals in pristine_tps.items()
    }
    metrics["candidate_tps_per_workload"] = candidate_mean_tps
    metrics["candidate_tps_runs"] = {name: vals for name, vals in candidate_tps.items()}
    metrics["pristine_tps_per_workload"] = pristine_mean_tps
    metrics["pristine_tps_runs"] = {name: vals for name, vals in pristine_tps.items()}
    metrics["bench_mode"] = bench_mode
    metrics["paired_mode"] = "per_workload"

    metrics["dual_track_metrics_per_workload"] = (
        _collect_dual_track_metrics(raw_dir, workloads, num_runs)
    )
    metrics["primary_metric_per_workload"] = {
        wl["name"]: PRIMARY_METRIC.get(
            wl["name"], ("token_throughput", "higher_is_better")
        )[0]
        for wl in workloads
    }

    insufficient_workloads = []
    _min_valid_by_wl: dict[str, int] = {}
    for name in {**candidate_tps, **pristine_tps}:
        _wl_nr = int(os.environ.get(
            f"INFB_WORKLOAD_NUM_RUNS_{name.upper()}",
            _declared_runs_for(name, num_runs)))
        min_valid = int(os.environ.get("INFB_MIN_VALID_RUNS", str(min(3, _wl_nr))))
        _min_valid_by_wl[name] = min_valid
        cn = len(candidate_tps.get(name, []))
        pn = len(pristine_tps.get(name, []))
        if cn < min_valid or pn < min_valid:
            insufficient_workloads.append(
                {"workload": name, "candidate_n": cn, "pristine_n": pn, "min_valid": min_valid})
    if insufficient_workloads:
        metrics["insufficient_data"] = {
            "min_valid_runs_by_workload": _min_valid_by_wl,
            "configured_num_runs": num_runs,
            "workloads_below_floor": insufficient_workloads,
        }
        print(
            f"\n[real] INSUFFICIENT DATA: {len(insufficient_workloads)} workload(s) "
            f"below their per-workload min_valid floor.",
            flush=True,
        )
        for w in insufficient_workloads:
            print(f"  {w['workload']}: candidate_n={w['candidate_n']}, "
                  f"pristine_n={w['pristine_n']} (need >={w['min_valid']})", flush=True)

        _pristine_failed = [w for w in insufficient_workloads
                            if w["pristine_n"] < w["min_valid"]]
        if _pristine_failed:
            _deterministic = deterministic_pristine_signature(raw_dir)
            metrics["infra_void"] = {
                "reason": "pristine_baseline_below_floor",
                "workloads": _pristine_failed,
                "deterministic": _deterministic,
                "note": (
                    ("unmodified baseline could not be measured, and the logs carry a DETERMINISTIC signature "
                     f"({_deterministic['signature']}): {_deterministic['meaning']}. Do NOT re-run -- "
                     f"{_deterministic['remedy']}. Still not scored 0: this is not a candidate regression.")
                    if _deterministic else
                    ("unmodified baseline could not be measured -> engine/GPU "
                     "infra flakiness, not a candidate regression; RE-RUN, do "
                     "not score 0 against the model")
                ),
            }
            if _deterministic:
                print("[real] INFRA-VOID: pristine baseline below floor on "
                      f"{[w['workload'] for w in _pristine_failed]} — DETERMINISTIC "
                      f"({_deterministic['signature']} in {_deterministic['log']}): "
                      f"{_deterministic['remedy']}. Re-running will fail identically.",
                      flush=True)
            else:
                print("[real] INFRA-VOID: pristine baseline below floor on "
                      f"{[w['workload'] for w in _pristine_failed]} — engine/GPU flakiness, "
                      "not a candidate failure; this run should be RE-RUN, not scored 0.",
                      flush=True)

    speedup_per_workload: dict[str, float] = {}
    for name in candidate_mean_tps:
        cand = candidate_mean_tps[name]
        pris = pristine_mean_tps.get(name, 0.0)
        if pris > 0 and cand > 0:
            speedup_per_workload[name] = cand / pris
    metrics["speedup_per_workload"] = speedup_per_workload

    if not speedup_per_workload:
        cand_workloads_with_data = sum(1 for v in candidate_mean_tps.values() if v > 0)
        pris_workloads_with_data = sum(1 for v in pristine_mean_tps.values() if v > 0)
        if cand_workloads_with_data == 0 and pris_workloads_with_data == 0:
            metrics["reason"] = "both_sides_failed"
        elif cand_workloads_with_data == 0:
            metrics["reason"] = "candidate_failed"
        elif pris_workloads_with_data == 0:
            metrics["reason"] = "pristine_failed"
        else:
            metrics["reason"] = "no_overlapping_workloads"
        metrics["candidate_workloads_with_data"] = cand_workloads_with_data
        metrics["pristine_workloads_with_data"] = pris_workloads_with_data

    _pd_gates = _pd_shared("pd_gates") if _pd_metadata() else None
    if _pd_gates is not None and len(_PD_TAIL_TTFT_BY_SIDE) == 2:
        _fired = _pd_gates.ttft_guardrail({
            "pristine": {"continuation_p95_ttft_ms": _PD_TAIL_TTFT_BY_SIDE.get("pristine")},
            "candidate": {"continuation_p95_ttft_ms": _PD_TAIL_TTFT_BY_SIDE.get("cand")},
        })
        for _token, _message in _fired:
            print("[verifier] GATE " + _token + ": " + _message, flush=True)
            metrics["pd_gate_failed"] = {"reason": _token, "message": _message}
        metrics["pd_tail_ttft_ms"] = dict(_PD_TAIL_TTFT_BY_SIDE)
    elif _pd_gates is not None:
        print("[verifier] GATE ttft_guardrail DID NOT RUN: tail TTFT present for "
              + repr(sorted(_PD_TAIL_TTFT_BY_SIDE)) + ", need both arms", flush=True)

    if (metrics.get("insufficient_data") or metrics.get("acceptance_gate_failed")
            or metrics.get("pd_gate_failed")):
        bundle = {
            "metrics": dict(metrics),
            "rewards": {"reward": 0.0,
                        "correctness": 1.0 if correctness_result.get("correctness_ok") else 0.0,
                        "performance": 0.0},
        }
        _ar = (metrics.get("acceptance_gate_failed") or {}).get("reason")
        _pdr = (metrics.get("pd_gate_failed") or {}).get("reason")
        if _pdr:
            bundle["metrics"]["reason"] = _pdr
        elif metrics.get("insufficient_data"):
            bundle["metrics"]["reason"] = "insufficient_data"
        elif _ar == "acceptance_length_below_floor":
            bundle["metrics"]["reason"] = "acceptance_length_below_floor"
        elif _ar == "acceptance_pristine_no_drafts":
            bundle["metrics"]["reason"] = "acceptance_pristine_no_drafts"
        else:
            bundle["metrics"]["reason"] = "acceptance_no_drafts"
        bundle["metrics"]["speedup_per_workload"] = speedup_per_workload
        _partial_geomean = (
            geomean(list(speedup_per_workload.values())) if speedup_per_workload else None
        )
        # None, not 0.0: a geomean over an empty set is undefined, and 0.0 reads as "we measured
        # zero speedup" rather than "we measured nothing".
        bundle["metrics"]["speedup_geomean"] = _partial_geomean
        # The partial numbers go under their OWN names. A disclaimed number must not occupy the slot
        # a scored number occupies, or the next reader averages it into a leaderboard.
        bundle["rewards"]["speedup_geomean"] = None
        bundle["rewards"]["speedup_per_workload_partial"] = dict(speedup_per_workload or {})
        bundle["rewards"]["void_reason"] = bundle["metrics"]["reason"]
        if _partial_geomean is not None:
            bundle["rewards"]["partial_geomean_unscored"] = _partial_geomean
    else:
        bundle = compute_reward(
            speedup_per_workload=speedup_per_workload,
            correctness_passed=True,
        )
    bundle["metrics"].update(metrics)
    bundle["metrics"]["elapsed_total_sec"] = time.time() - t_start
    _write_results(args, rewards=bundle["rewards"], metrics=bundle["metrics"])

    reward = bundle["rewards"]["reward"]
    speedup = metrics.get("speedup_per_workload", {})
    print(f"\n[real] reward={reward:.3f}  speedup={speedup}", flush=True)
    return 0



def _fail(args, metrics: dict, reason: str, reason_key: str):
    metrics["reason"] = reason_key
    metrics["error"] = reason
    _write_results(
        args,
        rewards={"reward": 0.0, "correctness": 0.0, "performance": 0.0},
        metrics=metrics,
    )
    print(f"[real] FAIL reason={reason_key}: {reason}", flush=True)


def _write_results(args, rewards: dict, metrics: dict):
    if metrics.get("infra_void") and "infra_void" not in rewards:
        rewards = {**rewards, "infra_void": 1}
    if (
        metrics.get("preflight_gpu_mismatch_allowed")
        and "preflight_gpu_mismatch_allowed" not in rewards
    ):
        rewards = {**rewards, "preflight_gpu_mismatch_allowed": 1}
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reward).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
                                  encoding="utf-8")
    Path(args.reward).write_text(json.dumps(rewards, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
