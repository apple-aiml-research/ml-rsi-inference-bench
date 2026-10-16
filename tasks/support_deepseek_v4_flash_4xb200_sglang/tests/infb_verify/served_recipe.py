#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The release recipe as served, read from the graded launch itself (the node is test_hidden.py's
`test_recipe_conformance`, with config.json `infb.recipe.attention_kernel_family`).

The checker is the shared module the agent is given,
`/app/eval/deepseek_v4_flash/tests/served_recipe.py`, loaded here from the tree test.sh restored to the base and
checked against config.json's pinned tree before any import from it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

EVAL_DIR = Path(os.environ.get("APP_DIR", "/app")) / "eval" / "deepseek_v4_flash"
NAME = "deepseek_v4_flash_eval_served_recipe"

if NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(NAME, EVAL_DIR / "tests" / "served_recipe.py")
    if _spec is None or _spec.loader is None:  # pragma: no cover - a missing file is a verifier fault
        raise RuntimeError(f"cannot load the served-recipe checker at {EVAL_DIR / 'tests' / 'served_recipe.py'}")
    _m = importlib.util.module_from_spec(_spec)
    sys.modules[NAME] = _m
    _spec.loader.exec_module(_m)
_m = sys.modules[NAME]
TRTLLM_MXFP4 = _m.TRTLLM_MXFP4
TRITON_FUSED_MOE = _m.TRITON_FUSED_MOE
BACKEND_KEYS = _m.BACKEND_KEYS
PROMPT = _m.PROMPT
server_info = _m.server_info
max_model_len = _m.max_model_len
profile_ranks = _m.profile_ranks
check = _m.check
