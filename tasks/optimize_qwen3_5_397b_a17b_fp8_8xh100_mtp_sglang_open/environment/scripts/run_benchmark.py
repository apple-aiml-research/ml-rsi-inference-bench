#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Agent-facing benchmark entrypoint for throughput tasks."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _select_impl(script_dir: Path) -> Path:
    """The helper that EXISTS here, which is the same measurement the verifier scores either way."""
    pd = script_dir / "_run_benchmark_pd.py"
    if pd.exists():
        return pd
    path = script_dir / "_run_benchmark_serve.py"
    if not path.exists():
        raise SystemExit(f"No benchmark implementation found under {script_dir}")
    return path


if __name__ == "__main__":
    target = _select_impl(Path(__file__).resolve().parent)
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
