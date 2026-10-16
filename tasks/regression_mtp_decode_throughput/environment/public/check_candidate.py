#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cheap local smoke check: does the staged engine tree still compile and import?"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path(__file__).resolve().parents[1]


def _engine_python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def main() -> int:
    repo = _workspace() / "repo"
    out: dict[str, object] = {"repo": str(repo)}

    if not (repo / "vllm").is_dir():
        out.update(ok=False, error=f"no engine tree at {repo}")
        print(json.dumps(out, indent=2))
        return 2

    proc = subprocess.run(
        [_engine_python(), "-m", "compileall", "-q", str(repo / "vllm")],
        capture_output=True,
        text=True,
    )
    out["compiles"] = proc.returncode == 0
    if proc.returncode != 0:
        out["compile_error"] = (proc.stderr or proc.stdout)[-2000:]

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}".rstrip(":")
    proc = subprocess.run(
        [_engine_python(), "-c", "import vllm; print(vllm.__file__)"],
        capture_output=True,
        text=True,
        env=env,
        cwd="/tmp",
    )
    resolved = (proc.stdout or "").strip()
    out["imports"] = proc.returncode == 0
    out["resolved_from"] = resolved or None
    if proc.returncode != 0:
        out["import_error"] = (proc.stderr or "")[-2000:]
    elif not resolved.startswith(str(repo)):
        out["imports"] = False
        out["import_error"] = (
            f"resolved to {resolved}, not the staged tree -- an edit there would have no effect"
        )

    out["ok"] = bool(out["compiles"] and out["imports"])
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
