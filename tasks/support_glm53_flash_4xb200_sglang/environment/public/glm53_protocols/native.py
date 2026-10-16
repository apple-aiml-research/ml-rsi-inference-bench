#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The tree installs, and the engine resolves the checkpoint's declared architecture to a native model class."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

MARKER = "RESOLVED "

_RESOLVE = """
import json
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.utils import get_model_architecture
cfg = ModelConfig(model_path={model_dir!r}, trust_remote_code=False)
cls, arch = get_model_architecture(cfg)
print({marker!r} + json.dumps({{"module": cls.__module__, "cls": cls.__name__, "arch": arch,
      "model_type": getattr(cfg.hf_config, "model_type", None),
      "architectures": list(getattr(cfg.hf_config, "architectures", []) or [])}}))
"""


def install_editable(app_dir: Path, python: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, SGLANG_BUILD_RUST_EXTS="none", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return subprocess.run(
        [python, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--no-build-isolation", "--no-index", "-e",
         str(Path(app_dir) / "python")],
        env=env, capture_output=True, text=True, timeout=1800, cwd=str(app_dir),
    )


def sglang_location(python: str) -> subprocess.CompletedProcess:
    return subprocess.run([python, "-c", "import os, sglang; print(os.path.realpath(sglang.__file__))"],
                          capture_output=True, text=True, timeout=600, cwd="/")


def resolve_architecture(model_dir: str, python: str) -> subprocess.CompletedProcess:
    code = _RESOLVE.format(model_dir=str(model_dir), marker=MARKER)
    return subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=900, cwd="/")


def resolution_record(stdout: str) -> dict | None:
    lines = [ln for ln in stdout.splitlines() if ln.startswith(MARKER)]
    return json.loads(lines[-1][len(MARKER):]) if lines else None


def native_problems(record: dict, architecture: str, model_type: str) -> list[str]:
    problems = []
    if record["arch"] != architecture:
        problems.append(f"resolved architecture {record['arch']!r}, not the checkpoint's {architecture!r}: {record}")
    if not record["module"].startswith("sglang.srt.models."):
        problems.append(f"resolved outside the engine's model package: {record}")
    if record["module"].endswith(".transformers"):
        problems.append(f"resolved to the Transformers fallback: {record}")
    if record["model_type"] != model_type:
        problems.append(f"model type {record['model_type']!r}, not the checkpoint's {model_type!r}: {record}")
    return problems
