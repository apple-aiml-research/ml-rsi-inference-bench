#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The grading copy of the task's evaluation materials, and the one way the hidden wrappers import them.

The agent works with `/app/eval/deepseek_v41_flash/` and may edit it; test.sh restores it before any candidate check,
and the grader never imports it. The verifier image carries the same files at `/tests/eval/deepseek_v41_flash/`
(copied from `tests/eval/` by the verifier tail; the repository holds `environment/eval/` and `tests/eval/` equal,
file for file). Every candidate-checking module the hidden nodes use is loaded from there, by path, under a name of
its own, so no module search path can reach the agent's copy. The private inputs, thresholds and reward handling
stay in the hidden modules and in config.json.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
EVAL = TESTS_DIR / "eval" / "deepseek_v41_flash"


def load(rel: str):
    """The grading copy's module at `rel` (e.g. "tests/quality.py"), loaded once per process."""
    name = "infb_eval_" + rel.removesuffix(".py").replace("/", "_")
    if name in sys.modules:
        return sys.modules[name]
    path = EVAL / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no grading copy at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod
