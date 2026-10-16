#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for optimize_glm_5_2_8xb200_mtp_open.

Protocol
--------
1. Load workloads & the teacher-forced correctness reference. The reference
   is always generated live from the pristine worktree.
2. Run correctness check: candidate source via PYTHONPATH overlay on venv.
3. Run throughput on candidate (N workloads × num_runs passes; first
   warmup_runs discarded).
4. Re-measure throughput over the same workloads in a disposable pristine
   worktree, so the candidate's edits are never unwound in place.
5. Compute speedup = candidate TPS / pristine-recheck TPS per workload.
6. Apply the category gate: the patch is classified, and this verifier accepts
   every category (NARROW_REJECTED_KINDS is empty, so no kind is rejected).
7. Compute reward via _verifier_lib.scoring.compute_reward.

Environment
-----------
INFB_WORKSPACE          Root workspace dir (workspace/repo/ = vLLM source)
INFB_TASK_DIR           Task folder (tasks/optimize_glm_5_2_8xb200_mtp_open)
INFB_CORRECTNESS_MODE   "disabled" (off) | else e.g. "loose"/"strict" (on) —
                        the ONLY correctness knob. The reference is always
                        generated live from the pristine worktree.
INFB_CORRECTNESS_PROMPTS  Override the live-mode prompt bank path.
INFB_CORRECTNESS_GPU_MEM_UTIL  Override correctness/reference engine gpu-mem (0.85).
INFB_VENV               Override path to the vllm venv (default: auto-detected from task)
CUDA_VISIBLE_DEVICES    GPU selection (default: "0")

Correctness reference
---------------------
LIVE, always: the verifier spins a disposable pristine worktree and runs
`_candidate_runner.py gen_reference` (two-pass teacher forcing) to produce
reference.npz in the task container from the baseline code. Prompts come from
tests/correctness_prompts.json (shipped with the task).
"""
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
    _extract_tps = None  # type: ignore
    _gpu_pids_via_proc = None  # type: ignore
    _kill_stray_gpu_processes = None  # type: ignore
    _resolve_workload_runtime = None  # type: ignore
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
from _verifier_lib.generation_judge import (
    after_measure as _gen_judge_after_measure,
    gate as _gen_judge_gate,
    judge_all as _gen_judge_judge_all,
    read_from_raw_dir as _gen_judge_read,
    record_judge_error as _gen_judge_record_error,
    strip_speculative_argv as _gen_judge_strip_speculative,
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

_INFB_SPECULATIVE = True


MODEL_ID   = "zai-org/GLM-5.2-FP8"
EXPECTED_GPU_TYPES = ['B200']
EXPECTED_GPU_COUNT = 8
TP_SIZE    = 8

KV_CACHE_DTYPE = "fp8"

LIVE_CORRECTNESS_SUPPORTED = True

CORRECTNESS_CALIBRATION = {
    "atol":          0.4,
    "rtol":          0.15,
    "min_pass_rate": 0.85,
}

SCORING_CALIBRATION = {
    "num_runs":               5,
    "warmup_runs":            1,
}

TASK_SCORING = {
    "num_runs":              5,
    "warmup_runs":           2,
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
    """Prefer the NCCL shim named by INFB_NCCL_PRELOAD, else fallback_path when one is
    given. Pins BOTH LD_PRELOAD (prepended, preserving any existing value) and
    VLLM_NCCL_SO_PATH so vLLM's own loader picks the compat nccl too."""
    for _nccl in (env.get("INFB_NCCL_PRELOAD"), fallback_path):
        if not _nccl or not os.path.exists(_nccl):
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
    ws = Path(os.environ.get("INFB_WORKSPACE", "/app"))
    overlay = ws / ".flashinfer_overlay"
    return str(overlay) if overlay.exists() else None


def _find_model(workspace: Path) -> str:
    """Resolve the model path."""
    if env := os.environ.get("INFB_MODEL_PATH"):
        return env
    ws_model = workspace / "model"
    if ws_model.exists():
        return str(ws_model)
    hf_home = Path(os.environ.get("HF_HOME",
                   Path(os.environ.get("HOME", "/root")) / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub" / "models--zai-org--GLM-5.2-FP8" / "snapshots"
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
        candidate = Path(env) / "builds" / "vllm" / "glm_5_2_8xb200" / "venv"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Cannot find vllm venv. Set INFB_VENV, INFB_VLLM_PYTHON, "
        "INFB_VLLM_TARGET_ENV_ARTIFACT, or INFB_ARTIFACT_CACHE."
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



_COMPILE_CACHE_DIR_LINE = (
    r"Using cache directory: (?P<key>\S+?/torch_compile_cache/[0-9A-Za-z_.-]+)/rank_\d+_\d+/"
)


def _compile_cache_keys_loaded_by(serve_log: Path) -> list[Path]:
    """torch_compile_cache/<key> dirs a vLLM serve log says it LOADED graphs from ("Directly load").
    Empty for a missing log, a disabled cache, or a fresh compile ("Cache the graph ...")."""
    import re
    try:
        text = Path(serve_log).read_text(errors="replace")
    except OSError:
        return []
    if "Directly load" not in text:
        return []
    keys: list[Path] = []
    for m in re.finditer(_COMPILE_CACHE_DIR_LINE, text):
        key = Path(m.group("key"))
        if key not in keys:
            keys.append(key)
    return keys


def _drop_compile_cache_keys(keys: list[Path]) -> list[Path]:
    """Remove whole torch_compile_cache/<key> dirs; refuses anything that is not such a key dir."""
    import shutil
    removed: list[Path] = []
    for key in keys:
        key = Path(key)
        if key.parent.name != "torch_compile_cache" or not key.is_dir():
            print(f"[compile-cache] refusing to drop {key}: not a torch_compile_cache key directory",
                  flush=True)
            continue
        shutil.rmtree(key, ignore_errors=True)
        removed.append(key)
        print(f"[compile-cache] dropped {key} (candidate key; the pristine worktree hashes to another)",
              flush=True)
    return removed


def _pick_free_port() -> int:
    """Bind to port 0, let OS pick, close. Small TOCTOU window but fine in
    practice — server starts on the same port immediately after."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server_ready(port: int, timeout: float, server_proc) -> bool:
    """Poll /v1/models until 200 or until timeout/server-died."""
    import urllib.request
    import urllib.error
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    first_err: tuple[str, str] | None = None
    last_err:  tuple[str, str] | None = None
    poll_count = 0
    while time.time() < deadline:
        if server_proc.poll() is not None:
            print(f"[serve] _wait_for_server_ready: server_proc died (rc={server_proc.returncode}) "
                  f"after {poll_count} polls; first_err={first_err} last_err={last_err}", flush=True)
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    if poll_count > 0:
                        print(f"[serve] _wait_for_server_ready: ready after {poll_count} polls "
                              f"(first_err={first_err})", flush=True)
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            err_pair = (type(e).__name__, str(e)[:160])
            if first_err is None:
                first_err = err_pair
            last_err = err_pair
        poll_count += 1
        time.sleep(2)
    print(f"[serve] _wait_for_server_ready: TIMEOUT after {poll_count} polls in {timeout}s; "
          f"first_err={first_err} last_err={last_err}", flush=True)
    return False



def _start_vllm_serve(
    *,
    model: str,
    venv: Path,
    repo: Path,
    port: int,
    pythonpath_overlay: bool,
    log_path: Path,
    speculative: bool = True,
):
    """Spawn `vllm serve` in its own process group."""
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
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
        "--dtype", "auto",
        "--kv-cache-dtype", "fp8",
        "--trust-remote-code",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.9",
        "--no-enable-prefix-caching",
        "--block-size", "128",
        "--speculative-config", '{"draft_sample_method": "greedy", "method": "mtp", "num_speculative_tokens": 1, "rejection_sample_method": "standard"}',
    ]
    if not speculative:
        cmd = _gen_judge_strip_speculative(cmd)

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    pp_parts = []
    if pythonpath_overlay:
        pp_parts.append(str(repo))
    if (fi_overlay := _flashinfer_overlay_path()) is not None:
        pp_parts.append(fi_overlay)
    if pp_parts:
        env["PYTHONPATH"] = ":".join(pp_parts)
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    _artifact_cache = env.get("INFB_ARTIFACT_CACHE")
    _apply_nccl_compat(env, (f"{_artifact_cache}/builds/vllm/glm_5_2_8xb200/nccl/libnccl_cu12.so.2"
                             if _artifact_cache else None))
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("CUDA_VISIBLE_DEVICES", _default_visible_devices())

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w")

    print(f"[serve] starting vllm serve on :{port} overlay={pythonpath_overlay}, log → {log_path}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def _stop_vllm_serve(proc) -> None:
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
    raw_dir, workloads, num_runs
):
    """Re-extract per-workload-per-side dual-track metrics from raw bench
    JSONs. Returns a flat
    {workload_name: {side: {metric_name: trim_mean}}} dict (legacy
    single-profile shape). Multi-profile workloads add an inner profile
    layer; this helper handles both.
    """
    if _extract_metrics_dict is None or _trimmed_mean is None:
        return {}

    def _avg_per_metric(files):
        per_metric = {}
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

    out = {}
    for wl in workloads:
        wl_name = wl["name"]
        profiles = wl.get("profiles") or []
        out[wl_name] = {}
        for side in ("cand", "pristine"):
            if profiles:
                per_profile = {}
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
        "dataset":         workload.get("dataset"),
        "prompt_bank":     workload.get("prompt_bank"),
        "shots":           workload.get("shots"),
        "ignore_eos":      workload.get("ignore_eos"),
        "save_detailed":   workload.get("save_detailed"),
    }))

    runner = THIS_DIR / "_candidate_runner.py"
    python = venv / "bin" / "python"

    cmd = [str(python), "-P", str(runner),
           "serve_bench",
           "--workload-json", str(wl_json),
           "--out-json",      str(out_file)]

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    server_boot_timeout: int = 1800,
) -> dict[str, list[float]]:
    """Run one full side (candidate or pristine) under serve mode."""
    port = _pick_free_port()
    log_path = raw_dir / f"vllm_serve_{side}_{port}.log"
    server = _start_vllm_serve(
        model=model_path, venv=venv, repo=repo, port=port,
        pythonpath_overlay=pythonpath_overlay, log_path=log_path,
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
                )
            for i in range(num_runs):
                r = _run_bench_serve(
                    workload=wl, model_path=model_path, venv=venv, port=port,
                    run_label=f"{side}_meas{i}", raw_dir=raw_dir,
                )
                if r:
                    tps = _extract_tps(r)
                    if tps:
                        side_tps[wl["name"]].append(tps)
            print(f"  {wl['name']}: n={len(side_tps[wl['name']])} "
                  f"tps={side_tps[wl['name']]}", flush=True)
    finally:
        _stop_vllm_serve(server)
        print(f"[serve] {side} server stopped", flush=True)
        _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
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
    server_boot_timeout: int = 1800,
) -> list[float]:
    """Per-workload serve PAIRED helper.  Boots a fresh vllm serve, runs
    warmup + meas for ONE workload, tears down + cleans GPU."""
    port = _pick_free_port()
    log_path = raw_dir / f"vllm_serve_{wl['name']}_{side}_{port}.log"
    server = _start_vllm_serve(
        model=model_path, venv=venv, repo=repo, port=port,
        pythonpath_overlay=pythonpath_overlay, log_path=log_path,
    )
    runs: list[float] = []
    try:
        if not _wait_for_server_ready(port, server_boot_timeout, server):
            print(f"  [serve] {wl['name']}/{side}: server not ready in {server_boot_timeout}s",
                  flush=True)
            return runs
        print(f"  [serve] {wl['name']}/{side}: server ready on :{port}", flush=True)

        for i in range(warmup_runs):
            _run_bench_serve(
                workload=wl, model_path=model_path, venv=venv, port=port,
                run_label=f"{wl['name']}_{side}_warmup{i}", raw_dir=raw_dir,
            )
        for i in range(num_runs):
            r = _run_bench_serve(
                workload=wl, model_path=model_path, venv=venv, port=port,
                run_label=f"{wl['name']}_{side}_meas{i}", raw_dir=raw_dir,
            )
            if r:
                score = _compute_primary_score(r, wl["name"]) \
                    if _compute_primary_score is not None else _extract_tps(r)
                if score:
                    runs.append(score)
    finally:
        _stop_vllm_serve(server)
        _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
        for _dev in ([d for d in _vis_cleanup.split(",") if d.strip()] or ["0"]):
            _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=5.0)
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
    server_boot_timeout: int = 1800,
) -> list[float]:
    """Multi-profile per-workload serve helper (the high_load workload).
    Boots ONE vllm serve, runs warmup + meas for EACH profile
    listed in `wl['profiles']` back-to-back against the same server, then
    tears down."""
    profiles = wl.get("profiles") or []
    if not profiles:
        raise ValueError(
            f"{wl['name']!r}: multi-profile path called without 'profiles' field"
        )

    port = _pick_free_port()
    log_path = raw_dir / f"vllm_serve_{wl['name']}_{side}_{port}.log"
    server = _start_vllm_serve(
        model=model_path, venv=venv, repo=repo, port=port,
        pythonpath_overlay=pythonpath_overlay, log_path=log_path,
    )
    per_profile_runs: dict[str, list[float]] = {p["name"]: [] for p in profiles}
    try:
        if not _wait_for_server_ready(port, server_boot_timeout, server):
            print(f"  [serve-mp] {wl['name']}/{side}: server not ready in {server_boot_timeout}s",
                  flush=True)
            return []
        print(f"  [serve-mp] {wl['name']}/{side}: server ready on :{port}", flush=True)

        for i in range(warmup_runs):
            _run_bench_serve(
                workload=wl, model_path=model_path, venv=venv, port=port,
                run_label=f"{wl['name']}_{side}_warmup{i}", raw_dir=raw_dir,
                profile=profiles[0],
            )
        for i in range(num_runs):
            for prof in profiles:
                pname = prof["name"]
                r = _run_bench_serve(
                    workload=wl, model_path=model_path, venv=venv, port=port,
                    run_label=f"{wl['name']}_{side}_{pname}_meas{i}", raw_dir=raw_dir,
                    profile=prof,
                )
                if r:
                    score = _compute_primary_score(r, wl["name"]) \
                        if _compute_primary_score is not None else _extract_tps(r)
                    if score:
                        per_profile_runs[pname].append(score)
        _gen_judge_after_measure(
            engine="vllm", port=port, side=side, workload=wl["name"], raw_dir=raw_dir,
            prompts_path=THIS_DIR / "correctness_prompts.json", speculative=_INFB_SPECULATIVE,
        )
    finally:
        _stop_vllm_serve(server)
        _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
        for _dev in ([d for d in _vis_cleanup.split(",") if d.strip()] or ["0"]):
            _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=5.0)

    aggregated: list[float] = []
    _scored_profiles = [p for p in profiles if p.get("scored", True)]
    if not _scored_profiles:
        raise ValueError(
            f"{wl['name']!r}: every profile is scored=false, so the workload has no score"
        )
    n_meas = min((len(per_profile_runs[p["name"]]) for p in _scored_profiles), default=0)
    for i in range(n_meas):
        scores_at_i = [per_profile_runs[p["name"]][i] for p in _scored_profiles]
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
    pp_parts = [str(repo)]
    if (fi_overlay := _flashinfer_overlay_path()) is not None:
        pp_parts.append(fi_overlay)
    env["PYTHONPATH"] = ":".join(pp_parts)
    env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env.setdefault("HF_HUB_OFFLINE",       "1")
    env.setdefault("TRANSFORMERS_OFFLINE",  "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", _default_visible_devices())
    _ac = env.get("INFB_ARTIFACT_CACHE")
    _apply_nccl_compat(env, (f"{_ac}/builds/vllm/glm_5_2_8xb200/nccl/libnccl_cu12.so.2"
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
        "block_size":              128,
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
            pp_parts = [str(pris_repo)]
            if (fi_overlay := _flashinfer_overlay_path()) is not None:
                pp_parts.append(fi_overlay)
            env["PYTHONPATH"] = ":".join(pp_parts)
            env["PATH"] = f"{venv}/bin:" + env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["INFB_KV_CACHE_DTYPE"] = KV_CACHE_DTYPE
            env.setdefault("CUDA_VISIBLE_DEVICES", _default_visible_devices())
            _ac = env.get("INFB_ARTIFACT_CACHE")
            _apply_nccl_compat(env, (f"{_ac}/builds/vllm/glm_5_2_8xb200/nccl/libnccl_cu12.so.2"
                                     if _ac else None))
            import json as _json
            env["INFB_CORRECTNESS_LLM_KWARGS_JSON"] = _json.dumps(_correctness_llm_kwargs(model_path))
            _force_spawn_workers(env)
            print(f"[live-ref] generating reference from pristine worktree {pris_repo}", flush=True)
            _boot_retries = int(os.environ.get("INFB_LIVE_REF_BOOT_RETRIES", "3"))
            _lr_timeout = int(os.environ.get("INFB_LIVE_REF_TIMEOUT_SEC", "3600"))
            result = None
            for _attempt in range(_boot_retries + 1):
                if _attempt:
                    print(f"  [live-ref] golden-gen boot RETRY {_attempt}/{_boot_retries} "
                          f"(prev rc={result.returncode}, npz not written — transient "
                          f"EngineCore boot kill; re-attempting)", flush=True)
                _lr_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
                _lr_devs = [d for d in _lr_visible.split(",") if d.strip()] or ["0"]
                for _lr_dev in _lr_devs:
                    _kill_stray_gpu_processes(cuda_device=_lr_dev, grace_sec=2.0)
                    try:
                        _wait_for_gpu_free_memory(cuda_device=_lr_dev, required_gib=60.0, timeout_sec=180.0)
                    except TimeoutError as _lr_e:
                        print(f"[live-ref][gpu-wait] WARN: {_lr_e}; proceeding with reference spawn anyway", flush=True)
                result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                        start_new_session=True, timeout=_lr_timeout)
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

    _csredit_install_prefix = None
    if prepare_csredit_rebuild_if_enabled is not None:
        try:
            _csredit_install_prefix = prepare_csredit_rebuild_if_enabled(
                task_dir=task_dir, workspace=workspace, repo=repo,
                venv=venv, raw_dir=raw_dir,
                build_timeout_sec=int(os.environ.get("INFB_CSRC_BUILD_TIMEOUT_SEC", "5400")),
            )
        except Exception as _e:
            metrics["csredit_rebuild"] = f"failed: {_e}"
            if os.environ.get("INFB_EDIT_POLICY_CSRC", "").strip().lower() in ("1", "true", "yes"):
                raise RuntimeError(
                    "[verifier] csrc rebuild FAILED with INFB_EDIT_POLICY_CSRC "
                    "set; refusing to grade an un-rebuilt edit as ~1.0x: " + str(_e)
                ) from _e
        if _csredit_install_prefix is not None:
            metrics["csredit_rebuild"] = "ok"
            metrics["csredit_install_prefix"] = str(_csredit_install_prefix)

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
                    write_void_metrics(args.metrics, metrics, reward_path=args.reward)
                    return PREFLIGHT_VOID_EXIT_CODE
                prompts_json = _live_prompts
                ref_npz = _live_ref
                if _kill_stray_gpu_processes is not None:
                    _vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
                    for _dev in ([d for d in _vis.split(",") if d.strip()] or ["0"]):
                        _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=2.0)
                        if _wait_for_gpu_free_memory is not None:
                            try:
                                _wait_for_gpu_free_memory(cuda_device=_dev,
                                                          required_gib=60.0, timeout_sec=120.0)
                            except TimeoutError as _e:
                                print(f"[gpu-wait] WARN: {_e}; proceeding with candidate correctness", flush=True)
            if _csredit_install_prefix is not None and swap_so_to_built is not None:
                swap_so_to_built(repo, _csredit_install_prefix)
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
    print(f"  correctness_ok={correctness_ok}  "
          f"pass_rate={correctness_result.get('overall_pass_rate', 'N/A')}", flush=True)

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
    print(f"\n=== Speed measurement (bench_mode={bench_mode}, paired_mode=per_workload) ===", flush=True)
    num_runs    = int(os.environ.get("INFB_NUM_RUNS_OVERRIDE", _scoring_cfg["num_runs"]))
    warmup_runs = int(os.environ.get("INFB_WARMUP_RUNS_OVERRIDE", _scoring_cfg["warmup_runs"]))

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

    _vis_cleanup = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
    _cuda_devs_for_cleanup = [d for d in _vis_cleanup.split(",") if d.strip()] or ["0"]
    cuda_dev = _cuda_devs_for_cleanup[0]

    def _bench_one_workload(wl: dict, side_label: str, side_repo: Path) -> list[float]:
        """Run warmup + meas for a single workload + side."""
        wl_tps: list[float] = []
        _wlk = wl["name"].upper()
        _wl_declared = _declared_runs_for(wl["name"], num_runs)
        wl_num_runs = int(os.environ.get(f"INFB_WORKLOAD_NUM_RUNS_{_wlk}", _wl_declared))
        wl_warmup   = int(os.environ.get(f"INFB_WORKLOAD_WARMUP_RUNS_{_wlk}", warmup_runs))
        _measure = (_measure_one_workload_serve_multi_profile
                    if wl.get("profiles") else _measure_one_workload_serve)
        _valid_floor = int(os.environ.get("INFB_MIN_VALID_RUNS", str(min(3, wl_num_runs))))
        _max_retries = int(os.environ.get("INFB_WORKLOAD_BENCH_RETRIES", "3"))
        for _attempt in range(_max_retries + 1):
            wl_tps = _measure(
                wl=wl, side=side_label, model_path=model_path,
                venv=venv, repo=side_repo, pythonpath_overlay=True,
                num_runs=wl_num_runs, warmup_runs=wl_warmup, raw_dir=raw_dir,
            )
            if len(wl_tps) >= _valid_floor or _attempt >= _max_retries:
                return wl_tps
            print(f"[verifier] {wl['name']}/{side_label}: only {len(wl_tps)} valid "
                  f"run(s) (< floor {_valid_floor}) — serve likely aborted mid-bench; "
                  f"cleaning GPU + re-bench {_attempt + 1}/{_max_retries}", flush=True)
            for _dev in _cuda_devs_for_cleanup:
                if _kill_stray_gpu_processes is not None:
                    _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=5.0)
            if cleanup_stale_ipc is not None:
                cleanup_stale_ipc(verbose=True)
            time.sleep(8)
        return wl_tps

    if _csredit_install_prefix is not None and swap_so_to_venv is not None:
        swap_so_to_venv(repo, venv=venv)

    if disposable_pristine_worktree is None:
        raise RuntimeError(
            "[verifier] PAIRED requires _paired_helper "
            "(it is overlaid into tests/_verifier_lib/ when the verifier tree is assembled)"
        )
    _worktree_workdir = raw_dir.parent if raw_dir is not None else (repo.parent / "_verifier_workdir")
    _worktree_workdir.mkdir(parents=True, exist_ok=True)
    with disposable_pristine_worktree(repo, workdir=_worktree_workdir) as _worktree_for_pristine:
        _exp_prewarm = os.environ.get("INFB_PAIRED_PREWARM", "1") == "1"
        _exp_prewarm_2side = os.environ.get("INFB_PAIRED_PREWARM_2SIDE", "1") == "1"
        if _exp_prewarm:
            def _do_prewarm(prewarm_repo: Path, label: str) -> Path:
                if cleanup_stale_ipc is not None:
                    cleanup_stale_ipc(verbose=True)
                _port = _pick_free_port()
                _log = raw_dir / f"vllm_serve_{label}_{_port}.log"
                if os.environ.get("INFB_PAIRED_KEEP_SERVER_LOGS") == "1":
                    _kdir = Path(os.environ.get("INFB_ARTIFACTS_DIR", "/logs/artifacts")) / "paired_server_logs"
                    _kdir.mkdir(parents=True, exist_ok=True)
                    _log = _kdir / f"vllm_serve_{label}_{_port}.log"
                print(f"[verifier-prewarm] starting throwaway vllm serve on :{_port} (repo={prewarm_repo})", flush=True)
                _server = _start_vllm_serve(
                    model=model_path, venv=venv, repo=prewarm_repo, port=_port,
                    pythonpath_overlay=True, log_path=_log,
                )
                _t0 = time.time()
                try:
                    if _wait_for_server_ready(_port, 2400, _server):
                        print(f"[verifier-prewarm] {label} ready in {time.time()-_t0:.1f}s; killing "
                              f"(disk caches now populated)", flush=True)
                    else:
                        print(f"[verifier-prewarm] {label} WARN: server didn't reach ready; proceeding", flush=True)
                finally:
                    _stop_vllm_serve(_server)
                    for _dev in _cuda_devs_for_cleanup:
                        _kill_stray_gpu_processes(cuda_device=_dev, grace_sec=2.0)
                    time.sleep(8)
                    if cleanup_stale_ipc is not None:
                        cleanup_stale_ipc(verbose=True)
                return _log
            print(f"[verifier] EXPERIMENT: INFB_PAIRED_PREWARM=1 — throwaway init spawn(s) "
                  f"to populate disk caches "
                  f"({'BOTH cand+pris paths' if _exp_prewarm_2side else 'cand path only'})", flush=True)
            if _csredit_install_prefix is not None and swap_so_to_built is not None:
                swap_so_to_built(repo, _csredit_install_prefix)
                _cand_prewarm_log = _do_prewarm(prewarm_repo=repo, label="prewarm_cand")
                _stale_keys = _compile_cache_keys_loaded_by(_cand_prewarm_log)
                if _stale_keys:
                    print(f"[verifier-prewarm] prewarm_cand loaded a pre-existing torch.compile entry "
                          f"({', '.join(str(k) for k in _stale_keys)}): not this warm-up's, so its graphs "
                          f"may not match the rebuilt .so; dropping it and warming up again", flush=True)
                    _drop_compile_cache_keys(_stale_keys)
                    _do_prewarm(prewarm_repo=repo, label="prewarm_cand_fresh")
            else:
                _do_prewarm(prewarm_repo=repo, label="prewarm_cand")
            if _exp_prewarm_2side:
                _do_prewarm(prewarm_repo=_worktree_for_pristine, label="prewarm_pris")

        for wl in workloads:
            wl_name = wl["name"]
            print(f"\n--- {wl_name}: candidate side ---", flush=True)
            if cleanup_stale_ipc is not None:
                cleanup_stale_ipc(verbose=True)
            if _csredit_install_prefix is not None and swap_so_to_built is not None:
                swap_so_to_built(repo, _csredit_install_prefix)
            cand_runs = _bench_one_workload(wl, "cand", repo)
            candidate_tps[wl_name] = cand_runs
            print(f"  {wl_name}/cand: n={len(cand_runs)} tps={cand_runs}", flush=True)

            print(f"--- {wl_name}: pristine side (in worktree at inference_base) ---", flush=True)
            if cleanup_stale_ipc is not None:
                cleanup_stale_ipc(verbose=True)
            pris_runs = _bench_one_workload(wl, "pristine", _worktree_for_pristine)
            pristine_tps[wl_name] = pris_runs
            print(f"  {wl_name}/pris: n={len(pris_runs)} tps={pris_runs}", flush=True)


        if _INFB_SPECULATIVE:
            _gj_wls = [_w["name"] for _w in workloads]
            try:
                if cleanup_stale_ipc is not None:
                    cleanup_stale_ipc(verbose=True)
                _gj_port = _pick_free_port()
                _gj_server = _start_vllm_serve(
                    model=model_path, venv=venv, repo=_worktree_for_pristine, port=_gj_port,
                    pythonpath_overlay=True, log_path=raw_dir / f"vllm_serve_gen_judge_{_gj_port}.log",
                    speculative=False,
                )
                try:
                    _gj_ready = _wait_for_server_ready(_gj_port, 1800, _gj_server)
                    _gen_judge_judge_all(engine="vllm", port=_gj_port if _gj_ready else None,
                                         raw_dir=raw_dir, workloads=_gj_wls)
                finally:
                    _stop_vllm_serve(_gj_server)
                    _gj_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
                    for _gj_dev in ([d for d in _gj_vis.split(",") if d.strip()] or ["0"]):
                        _kill_stray_gpu_processes(cuda_device=_gj_dev, grace_sec=5.0)
            except Exception as _gj_err:              # noqa: BLE001
                _gen_judge_record_error(raw_dir, _gj_wls, f"{type(_gj_err).__name__}: {_gj_err}")

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

    _pp: dict = {}
    for _wl in workloads:
        _runs = _wl.get("_runtime_per_profile_runs")
        if _runs:
            _pp[_wl["name"]] = {
                _side: {_p: list(_v) for _p, _v in _by_profile.items()}
                for _side, _by_profile in _runs.items()
            }
    if _pp:
        metrics["scenario_c_per_profile_runs"] = _pp
        metrics["scenario_c_scored_profiles"] = {
            _wl["name"]: [p["name"] for p in (_wl.get("profiles") or []) if p.get("scored", True)]
            for _wl in workloads if _wl.get("profiles")
        }

    _wl_names = [_w["name"] for _w in workloads]
    try:
        _acc = _acceptance_gate(
            _acceptance_read(raw_dir, _wl_names), _wl_names,
            speculative=_INFB_SPECULATIVE,
        )
    except Exception as _acc_err:                     # noqa: BLE001
        _acc = {"failed": False, "reason": None,
                "error": f"{type(_acc_err).__name__}: {_acc_err}"}
    metrics["acceptance"] = _acc
    for _n, _d in (_acc.get("per_workload") or {}).items():
        print(f"  [acceptance] {_n}: status={_d.get('status')} ratio={_d.get('ratio')} "
              f"cand={(_d.get('cand') or {}).get('mean_acceptance_length')} "
              f"pristine={(_d.get('pristine') or {}).get('mean_acceptance_length')}", flush=True)
    if _acc.get("failed"):
        metrics["acceptance_gate_failed"] = _acc
        print(f"\n[real] ACCEPTANCE GATE FAILED ({_acc.get('reason')}): "
              f"{_acc.get('failures')}", flush=True)

    try:
        _gj = _gen_judge_gate(
            _gen_judge_read(raw_dir, _wl_names), _wl_names, speculative=_INFB_SPECULATIVE,
        )
    except Exception as _gj_err:                      # noqa: BLE001
        _gj = {"failed": False, "reason": None,
               "error": f"{type(_gj_err).__name__}: {_gj_err}"}
    metrics["generation_judge"] = _gj
    for _n, _d in (_gj.get("per_workload") or {}).items():
        _j1 = _d.get("j1") or {}
        _j2 = _d.get("j2") or {}
        print(f"  [gen-judge] {_n}: status={_gj.get('status')} "
              f"j1_cand={(_j1.get('cand') or {}).get('violations')} "
              f"j1_pristine={(_j1.get('pristine') or {}).get('violations')} "
              f"j2_z_top1={_j2.get('z_top1_share')} j2_z_logp={_j2.get('z_mean_logp')} "
              f"judge_error={_d.get('judge_error')}", flush=True)
    if _gj.get("failed"):
        metrics["generation_judge_failed"] = _gj
        print(f"\n[real] GENERATION JUDGE FAILED ({_gj.get('reason')}): "
              f"{_gj.get('failures')}", flush=True)

    try:
        _eqv = _equivalence_read(raw_dir, _wl_names, num_runs=num_runs)
        metrics["multi_token_equivalence"] = _equivalence_summarise(_eqv)
        metrics["multi_token_equivalence_detail"] = _eqv
    except Exception as _eqv_err:                     # noqa: BLE001
        metrics["multi_token_equivalence"] = {
            "error": f"{type(_eqv_err).__name__}: {_eqv_err}"}
    for _k, _v in ((metrics.get("multi_token_equivalence") or {}).get("per_point") or {}).items():
        print(f"  [equivalence] {_k}: cand_vs_pris={_v.get('cand_vs_pristine_identical_frac')} "
              f"pris_vs_self={_v.get('pristine_vs_self_identical_frac')} "
              f"interpretable={_v.get('interpretable')}", flush=True)

    try:
        _osh = _output_shape_read(raw_dir, _wl_names)
        metrics["output_shape"] = _output_shape_summarise(_osh)
        metrics["output_shape_detail"] = _osh
    except Exception as _osh_err:                      # noqa: BLE001
        metrics["output_shape"] = {"error": f"{type(_osh_err).__name__}: {_osh_err}"}
    for _k, _v in ((metrics.get("output_shape") or {}).get("per_point") or {}).items():
        if _v.get("repetition_p50") is None:
            print(f"  [output-shape] {_k}: BLIND -- {_v.get('status') or 'no completions'} "
                  f"(files={_v.get('completions', 0)}, blind_files={_v.get('blind_files', 0)})",
                  flush=True)
        else:
            print(f"  [output-shape] {_k}: rep_p50={_v.get('repetition_p50')} "
                  f"rep_p90={_v.get('repetition_p90')} words_p50={_v.get('words_p50')} "
                  f"above30pct={_v.get('share_above_30pct_repetition')}", flush=True)

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
    _unscored_wls = {
        w["name"] for w in workloads
        if isinstance(w, dict) and w.get("name") and not w.get("scored", True)
    }
    if _unscored_wls:
        print(f"[workload-scored] measured but OUTSIDE the geomean: {sorted(_unscored_wls)}", flush=True)
    for name in {**candidate_tps, **pristine_tps}:
        if name in _unscored_wls:
            continue
        _wl_nr = int(os.environ.get(
            f"INFB_WORKLOAD_NUM_RUNS_{name.upper()}",
            _declared_runs_for(name, num_runs)))
        MIN_VALID_RUNS = int(os.environ.get("INFB_MIN_VALID_RUNS", str(min(3, _wl_nr))))
        cn = len(candidate_tps.get(name, []))
        pn = len(pristine_tps.get(name, []))
        if cn < MIN_VALID_RUNS or pn < MIN_VALID_RUNS:
            insufficient_workloads.append({"workload": name, "candidate_n": cn, "pristine_n": pn})
    if insufficient_workloads:
        metrics["insufficient_data"] = {
            "min_valid_runs": MIN_VALID_RUNS,
            "configured_num_runs": num_runs,
            "workloads_below_floor": insufficient_workloads,
        }
        print(
            f"\n[real] INSUFFICIENT DATA: {len(insufficient_workloads)} workload(s) "
            f"landed fewer than {MIN_VALID_RUNS} valid measurements (out of {num_runs} attempted).",
            flush=True,
        )
        for w in insufficient_workloads:
            print(f"  {w['workload']}: candidate_n={w['candidate_n']}, "
                  f"pristine_n={w['pristine_n']}", flush=True)

        _pristine_failed = [w for w in insufficient_workloads
                            if w["pristine_n"] < MIN_VALID_RUNS]
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

    _scored_speedup = {k: v for k, v in speedup_per_workload.items() if k not in _unscored_wls}
    if _unscored_wls:
        metrics["unscored_workloads"] = sorted(_unscored_wls)
        metrics["scored_speedup_per_workload"] = _scored_speedup
    if speedup_per_workload and not _scored_speedup:
        raise ValueError(
            "every workload is scored=false, so this run has no score -- at least one must be scored"
        )

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

    if metrics.get("insufficient_data") or metrics.get("acceptance_gate_failed") or metrics.get("generation_judge_failed"):  # noqa: E501 -- deliberately one line
        bundle = {
            "metrics": dict(metrics),
            "rewards": {"reward": 0.0,
                        "correctness": 1.0 if correctness_ok else 0.0,
                        "performance": 0.0},
        }
        _ar = (metrics.get("acceptance_gate_failed") or {}).get("reason")
        if metrics.get("insufficient_data"):
            bundle["metrics"]["reason"] = "insufficient_data"
        elif _ar == "acceptance_length_below_floor":
            bundle["metrics"]["reason"] = "acceptance_length_below_floor"
        elif _ar == "acceptance_pristine_no_drafts":
            bundle["metrics"]["reason"] = "acceptance_pristine_no_drafts"
        elif _ar is None:
            if metrics["generation_judge_failed"].get("reason") == "spec_output_rejected":
                bundle["metrics"]["reason"] = "spec_output_rejected"
            else:
                bundle["metrics"]["reason"] = "generation_judge_unavailable"
        else:
            bundle["metrics"]["reason"] = "acceptance_no_drafts"
        bundle["metrics"]["speedup_per_workload"] = speedup_per_workload
        _partial_geomean = (
            geomean(list(_scored_speedup.values())) if _scored_speedup else None
        )
        bundle["metrics"]["speedup_geomean"] = _partial_geomean
        bundle["rewards"]["speedup_geomean"] = None
        bundle["rewards"]["speedup_per_workload_partial"] = dict(speedup_per_workload or {})
        bundle["rewards"]["void_reason"] = bundle["metrics"]["reason"]
        if _partial_geomean is not None:
            bundle["rewards"]["partial_geomean_unscored"] = _partial_geomean
    else:
        bundle = compute_reward(
            speedup_per_workload=_scored_speedup,
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
