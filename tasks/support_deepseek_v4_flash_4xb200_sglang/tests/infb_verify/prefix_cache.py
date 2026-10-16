#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prefix caching, measured as a behaviour of the served tree (the node is test_hidden.py's `test_prefix_cache`).

The protocol and comparison are the shared module the agent is given,
`/app/eval/deepseek_v4_flash/fidelity/prefix_cache.py`, loaded here from the tree test.sh restored to the base and
checked against config.json's pinned tree before any import from it. The node keeps the
grader's decisions (config.json `infb.bars.prefix_min_cached` and `prefix_T`, and the set's re-draw).

Cold and warm answers are each checked in full against the continuation's span before either is used, and
`cached_tokens` must be a count; a malformed answer fails the node and is recorded, never re-drawn.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

EVAL_DIR = Path(os.environ.get("APP_DIR", "/app")) / "eval" / "deepseek_v4_flash"
NAME = "deepseek_v4_flash_eval_prefix_cache"

if NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(NAME, EVAL_DIR / "fidelity" / "prefix_cache.py")
    if _spec is None or _spec.loader is None:  # pragma: no cover - a missing file is a verifier fault
        raise RuntimeError(f"cannot load the prefix-cache module at {EVAL_DIR / 'fidelity' / 'prefix_cache.py'}")
    _m = importlib.util.module_from_spec(_spec)
    sys.modules[NAME] = _m
    _spec.loader.exec_module(_m)
_m = sys.modules[NAME]
validate_item = _m.validate_item
load_items = _m.load_items
flush = _m.flush
teacher_forced = _m.teacher_forced
prime = _m.prime
measure_item = _m.measure_item
measure = _m.measure
