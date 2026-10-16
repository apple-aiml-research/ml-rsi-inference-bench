#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail-loud infra preflight."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, MutableMapping, Optional, Sequence

PREFLIGHT_VOID_EXIT_CODE = 79

_KNOWN_GPU_ARCHES = frozenset({"H100", "H200", "B200", "A100"})

ALLOW_GPU_MISMATCH_ENV = "INFB_PREFLIGHT_ALLOW_GPU_MISMATCH"
GPU_MISMATCH_ALLOWED_METRIC = "preflight_gpu_mismatch_allowed"

SMI_TIMEOUT_SEC = 30.0
TOOLCHAIN_TIMEOUT_SEC = 60.0
GIT_TIMEOUT_SEC = 60.0

_FALLBACK_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# What to do about an unresolved model. Named in the reason dict because the
# operator reading metrics.json is the person who has to fix it.
_MODEL_REMEDY = (
    "mount the weights and point the task at them: "
    "pier run ... --mounts-json '{\"<host>/model\": \"/model\"}' "
    "--ae INFB_MODEL_PATH=/model --ve INFB_MODEL_PATH=/model "
    "(the verifier's _find_model reads INFB_MODEL_PATH first, then <workspace>/model, "
    "then the HF hub cache)"
)


def _void(reason: str, **kw) -> dict:
    """Build an infra_void reason dict. ``note`` documents the intended outer
    action so the signal is self-describing in metrics.json.
    A caller-supplied ``note=`` in ``kw`` replaces the default."""
    return {
        "reason": reason,
        "note": (
            "infra precondition failed -> RE-RUN on correct hardware/mounts, "
            "do not score 0 against the model"
        ),
        **kw,
    }


def write_void_metrics(metrics_path, metrics, *, reward_path=None) -> None:
    """Record an infra void: write ``metrics.json`` and NEVER ``reward.json``."""
    p = Path(metrics_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if reward_path is not None:
        rp = Path(reward_path)
        if rp.exists():
            rp.unlink()
            print(
                f"[preflight] removed a pre-existing {rp} -- an infra void must "
                "leave NO reward file, or the runner banks it as a score",
                flush=True,
            )


def _arch_tokens(reported_name: str) -> set:
    """Tokenize an nvidia-smi product name for exact per-token arch matching."""
    token = ""
    tokens = set()
    for ch in reported_name.upper():
        if ch.isalnum():
            token += ch
        elif token:
            tokens.add(token)
            token = ""
    if token:
        tokens.add(token)
    return tokens


def _check_declared_arches(expected: Sequence[str]) -> Optional[dict]:
    """A statement about the TASK declaration, not about the host, so it is
    checked before nvidia-smi is consulted at all."""
    unknown = [tok for tok in expected if tok not in _KNOWN_GPU_ARCHES]
    if not unknown:
        return None
    return _void(
        "expected_gpu_type_unknown",
        expected_gpu_types=list(expected),
        unknown_gpu_types=unknown,
        known_gpu_types=sorted(_KNOWN_GPU_ARCHES),
        note=(
            "the task declares a GPU arch this preflight cannot recognise, so it can "
            "neither confirm nor deny the hardware -> RE-RUN after teaching "
            "preflight._KNOWN_GPU_ARCHES the token. NOT a hardware mismatch; do not "
            "score 0 against the model"
        ),
    )


def _probe_gpu_names(
    expected: Sequence[str], *, timeout_sec: float = SMI_TIMEOUT_SEC
) -> tuple:
    """Ask nvidia-smi what is visible. Returns ``(names, void_or_None)``."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [], _void(
            "no_gpu_visible",
            expected_gpu_types=list(expected),
            detail="nvidia-smi not on PATH",
        )
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return [], _void(
            "nvidia_smi_timeout",
            expected_gpu_types=list(expected),
            nvidia_smi_timeout_sec=timeout_sec,
            note=(
                "nvidia-smi did not answer within the timeout -- NVML is wedged, which "
                "means the driver/GPU is in a bad state and nothing measured here would "
                "be trustworthy -> RE-RUN on a healthy host; do not score 0 against the "
                "model"
            ),
        )
    except OSError as exc:
        return [], _void(
            "no_gpu_visible",
            expected_gpu_types=list(expected),
            detail=f"nvidia-smi could not be executed: {type(exc).__name__}: {exc}",
        )
    if out.returncode != 0:
        return [], _void(
            "no_gpu_visible",
            expected_gpu_types=list(expected),
            nvidia_smi_returncode=out.returncode,
            nvidia_smi_stderr=(out.stderr or "")[-400:],
        )
    names = [line.strip() for line in (out.stdout or "").splitlines() if line.strip()]
    if not names:
        return [], _void(
            "no_gpu_visible",
            expected_gpu_types=list(expected),
            detail="nvidia-smi reported no devices",
        )
    return names, None


def _match_arch(expected: Sequence[str], names: Sequence[str]) -> Optional[dict]:
    """``gpu_arch_mismatch`` -- the declared arch must be one that is present.
    A baseline calibrated on one architecture is not comparable to a run on
    another, so this is the precondition every measured number rests on.
    Pure: takes the already-probed names."""
    seen = set()
    for name in names:
        seen |= _arch_tokens(name) & _KNOWN_GPU_ARCHES
    if seen & set(expected):
        return None
    return _void(
        "gpu_arch_mismatch",
        expected_gpu_types=list(expected),
        detected=list(names),
        detected_arches=sorted(seen),
    )


def _check_gpu_count(expected_count, names: Sequence[str]) -> Optional[dict]:
    """An 8xH100 task handed 1 GPU used to sail through: the arch matched, and the
    engine then either failed to start TP=8 or silently measured something else."""
    if not expected_count:
        return None
    if len(names) >= int(expected_count):
        return None
    return _void(
        "gpu_count_insufficient",
        expected_gpu_count=int(expected_count),
        detected_gpu_count=len(names),
        detected=list(names),
        note=(
            "the task declares more GPUs than are visible, so the declared parallel "
            "shape cannot run and any number measured here is not the task's number "
            "-> RE-RUN on a host with the declared GPU count; do not score 0 against "
            "the model"
        ),
    )


def _check_model(find_model: Callable, workspace, model_id: str) -> Optional[dict]:
    """The unresolved/absent-model bug."""
    p = find_model(workspace)
    if p == model_id:
        return _void(
            "model_unresolved",
            model_id=model_id,
            resolved_value=str(p),
            remedy=_MODEL_REMEDY,
            note=(
                "_find_model fell through to the bare HF model id, so the engine would "
                "download the weights DURING scored verification (perturbing the "
                "measurement) or fail offline mid-run -> RE-RUN with the weights mounted; "
                "do not score 0 against the model"
            ),
        )
    pp = Path(p)
    if not pp.exists() or not (pp / "config.json").exists():
        return _void(
            "model_path_invalid",
            model_path=str(p),
            remedy=_MODEL_REMEDY,
            missing_config_json=pp.exists(),
        )
    return None


def _check_shm(floor_gib: float) -> Optional[dict]:
    """Too-small ``/dev/shm`` starves NCCL/torch shared-memory transports and
    can make an otherwise-valid engine crash -> a false 0."""
    try:
        st = os.statvfs("/dev/shm")
    except OSError as exc:
        return _void(
            "dev_shm_unavailable",
            path="/dev/shm",
            floor_gib=floor_gib,
            error=f"{type(exc).__name__}: {exc}",
            note=(
                "/dev/shm could not be inspected (absent, or not a mounted tmpfs), so "
                "the engine's shared-memory transports have nowhere to live -> RE-RUN "
                "with a tmpfs /dev/shm (docker: --shm-size); do not score 0 against the "
                "model"
            ),
        )
    have = st.f_bavail * st.f_frsize / (1 << 30)
    if have < floor_gib:
        return _void("dev_shm_too_small", have_gib=round(have, 2), floor_gib=floor_gib)
    return None


def _check_repo(workspace, *, timeout_sec: float = GIT_TIMEOUT_SEC) -> Optional[dict]:
    """Missing repo checkout or missing baseline tag: the paired verifier
    cannot build a pristine worktree at ``inference_base``, so there is no
    honest baseline to compare against."""
    repo = Path(workspace) / "repo"
    if not (repo / ".git").exists():
        return _void("repo_missing")
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "tag"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return _void(
            "git_probe_timeout",
            git_timeout_sec=timeout_sec,
            note=(
                "`git tag` did not answer within the timeout -- the repo mount is "
                "wedged, so nothing measured against it would be trustworthy -> "
                "RE-RUN on a healthy host; do not score 0 against the model"
            ),
        )
    except OSError as exc:
        return _void(
            "git_probe_failed",
            detail=f"git could not be executed: {type(exc).__name__}: {exc}",
        )
    if out.returncode != 0:
        return _void(
            "git_probe_failed",
            git_returncode=out.returncode,
            git_stderr=(out.stderr or "")[-400:],
        )
    if "inference_base" not in out.stdout.split():
        return _void("baseline_tag_missing")
    return None


def _resolve_probe_env(find_venv: Optional[Callable], task_dir) -> tuple:
    """Resolve the interpreter + PATH the ENGINE will run under."""
    inherited = os.environ.get("PATH") or _FALLBACK_PATH
    if find_venv is not None and task_dir is not None:
        try:
            venv = Path(find_venv(task_dir))
        except Exception:
            venv = None
        if venv is not None:
            for candidate in (venv / "bin" / "python3", venv / "bin" / "python"):
                if candidate.exists():
                    return str(candidate), f"{venv}/bin:{inherited}"
    return sys.executable, inherited


def _check_nvcc(probe_path: str) -> Optional[dict]:
    if shutil.which("nvcc", path=probe_path):
        return None
    return _void("nvcc_missing", probed_path=probe_path)


def _check_nccl(
    probe_python: str, *, timeout_sec: float = TOOLCHAIN_TIMEOUT_SEC
) -> Optional[dict]:
    try:
        out = subprocess.run(
            [probe_python, "-c", "import nvidia.nccl"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except Exception as exc:
        return _void(
            "nccl_unimportable",
            probed_python=probe_python,
            error=f"{type(exc).__name__}: {exc}",
        )
    if out.returncode != 0:
        return _void(
            "nccl_unimportable",
            probed_python=probe_python,
            stderr=(out.stderr or "")[-400:],
        )
    return None


def _gpu_mismatch_is_allowed() -> bool:
    return os.environ.get(ALLOW_GPU_MISMATCH_ENV) == "1"


def run_preflight(
    *,
    workspace,
    task_dir,
    model_id: str,
    expected_gpu_types,
    find_model: Callable,
    expected_gpu_count=None,
    find_venv: Optional[Callable] = None,
    shm_floor_gib: float = 1.0,
    smi_timeout_sec: float = SMI_TIMEOUT_SEC,
    metrics: Optional[MutableMapping] = None,
) -> Optional[dict]:
    """Run every infra precondition; return the first ``infra_void`` reason dict
    (or ``None`` when everything passes)."""
    expected_types = list(expected_gpu_types or [])
    if expected_types or expected_gpu_count:
        if expected_types:
            declared = _check_declared_arches(expected_types)
            if declared is not None:
                return declared
        names, probe_void = _probe_gpu_names(
            expected_types, timeout_sec=smi_timeout_sec
        )
        if probe_void is not None:
            return probe_void
        if expected_types:
            arch = _match_arch(expected_types, names)
            if arch is not None:
                if _gpu_mismatch_is_allowed():
                    waiver = {
                        "expected": arch.get("expected_gpu_types"),
                        "detected": arch.get("detected"),
                        "override_env": ALLOW_GPU_MISMATCH_ENV,
                        "note": (
                            "GPU arch mismatch WAIVED by explicit override: this run is a "
                            "FUNCTIONAL SMOKE on undeclared hardware, NOT a valid scored "
                            "measurement. Its speedup is not comparable to the calibrated "
                            "baseline and must not be published."
                        ),
                    }
                    if metrics is not None:
                        metrics[GPU_MISMATCH_ALLOWED_METRIC] = waiver
                    print(
                        f"[preflight] GPU-ARCH MISMATCH WAIVED by {ALLOW_GPU_MISMATCH_ENV}=1 "
                        f"expected={waiver['expected']} detected={waiver['detected']} "
                        "-> FUNCTIONAL SMOKE ONLY, this run is NOT a valid scored measurement",
                        flush=True,
                    )
                else:
                    return arch
        count = _check_gpu_count(expected_gpu_count, names)
        if count is not None:
            return count

    for fn in (
        lambda: _check_model(find_model, workspace, model_id),
        lambda: _check_shm(shm_floor_gib),
        lambda: _check_repo(workspace),
    ):
        r = fn()
        if r is not None:
            return r
    if os.environ.get("INFB_PREFLIGHT_TOOLCHAIN") == "1":
        probe_python, probe_path = _resolve_probe_env(find_venv, task_dir)
        for fn in (
            lambda: _check_nvcc(probe_path),
            lambda: _check_nccl(probe_python),
        ):
            r = fn()
            if r is not None:
                return r
    return None
