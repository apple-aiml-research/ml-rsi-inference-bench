#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Recipe conformance: the graded launch serves the recipe exactly as it was launched."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from glm53_protocols.recipe import NOT_RECIPE, compare, expected_fields, recipe_flags, server_info  # noqa: E402,F401

RECORD_DIR = Path(os.environ.get("INFB_CONFORMANCE_DIR", "/logs/verifier/conformance"))


def write_record(label: str, result: dict) -> str:
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORD_DIR / f"{label}.json"
        path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
        return str(path)
    except Exception as exc:  # noqa: BLE001 - a record, never a verdict
        return f"not written: {exc!r}"[:300]


def conformance(base_url: str, label: str, argv: list[str], resolved: dict[str, Any] | None = None) -> dict:
    resolved = dict(resolved or {})
    expected = expected_fields(list(argv), resolved)
    info = server_info(base_url)
    result = compare(expected, info)
    result["label"] = label
    result["argv"] = list(argv)
    result["resolved_expected"] = resolved
    result["record"] = write_record(label, result)
    return result
