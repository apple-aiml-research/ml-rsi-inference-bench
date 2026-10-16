#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared target-env (vllm venv) resolution.

Mirrors the v4 task pattern (deepseek_v4_pro_base_vllm) so external
users can swap between bind-mount packaging and managed-image packaging
without editing task scripts.

Resolution chain (highest priority first):
  1. INFB_VENV_PYTHON / INFB_VLLM_PYTHON env var (explicit override).
  2. <INFB_VLLM_TARGET_ENV_ARTIFACT>/managed_target_env.json::python_executable
     (preferred when the artifact has been promoted to a managed image).
  3. <INFB_VLLM_TARGET_ENV_ARTIFACT>/manifest.json::python_executable
     (fallback for older managed manifests).
  4. <INFB_VLLM_TARGET_ENV_ARTIFACT>/venv/bin/python3 (the bind-mount
     layout this task originally shipped with — still the production
     packaging today).

For each manifest hit, applies host->container path translation if the
manifest path CONTAINS the `/.cache/inference-bench/` marker but the
container sees a different mount root (INFB_ARTIFACT_CACHE), so a manifest
that records a host-absolute python path still resolves inside the
container.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_venv_python() -> str:
    """Return the venv's python3 binary path as visible inside the container."""
    if env := os.environ.get("INFB_VENV_PYTHON"):
        return env
    if env := os.environ.get("INFB_VLLM_PYTHON"):
        return env

    target_env = os.environ.get("INFB_VLLM_TARGET_ENV_ARTIFACT")
    if not target_env:
        raise RuntimeError(
            "Cannot resolve venv python; set INFB_VLLM_PYTHON or "
            "INFB_VLLM_TARGET_ENV_ARTIFACT"
        )

    for manifest_name in ("managed_target_env.json", "manifest.json"):
        mf_path = Path(target_env) / manifest_name
        if not mf_path.exists():
            continue
        try:
            data = json.loads(mf_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        py = data.get("python_executable")
        if not py:
            continue
        resolved = _translate_host_to_container(py)
        if Path(resolved).exists():
            return resolved

    p = Path(target_env) / "venv" / "bin" / "python3"
    if p.exists():
        return str(p)

    raise RuntimeError(
        f"Cannot resolve venv python under {target_env}. "
        f"Tried: managed_target_env.json/manifest.json (python_executable), "
        f"{target_env}/venv/bin/python3"
    )


def resolve_venv_dir() -> Path:
    """Return the venv root directory (.../venv), inferred from python binary."""
    return Path(resolve_venv_python()).parent.parent


def _translate_host_to_container(path: str) -> str:
    """Remap a host-absolute path to the container view if needed."""
    if Path(path).exists():
        return path
    artifact_cache = os.environ.get("INFB_ARTIFACT_CACHE")
    if not artifact_cache:
        return path
    marker = "/.cache/inference-bench/"
    if marker in path:
        suffix = path.split(marker, 1)[1]
        translated = str(Path(artifact_cache) / suffix)
        if Path(translated).exists():
            return translated
    return path
