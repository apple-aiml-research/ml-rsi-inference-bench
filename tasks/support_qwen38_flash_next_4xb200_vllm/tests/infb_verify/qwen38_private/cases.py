#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The private component case inventory and its token content.

Cases come from `hidden/expanded/components.json`, each validated against the public `scope.json`
by the trusted public validator (`qwen38.measurement.cases`); a nonconforming case is an authoring
defect, raised here, never an extra requirement. Token content is drawn deterministically by each
case's id and seed from the hidden fidelity pool (natural text), so reference and candidate see
identical requests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from qwen38.measurement.cases import CaseError, scope, tokens as _tokens, validate

ROOT = Path(os.environ.get("TESTS_DIR", "/tests"))
INVENTORY = ROOT / "hidden/expanded/components.json"
POOL = ROOT / "hidden/fidelity.json"

__all__ = ["CaseError", "inventory", "scope", "tokens", "validate"]


def inventory(recipe: str) -> list[dict]:
    data = json.loads(INVENTORY.read_text())
    if data.get("schema") != "qwen38-component-cases/2":
        raise CaseError("unexpected component inventory schema")
    bounds = scope()
    cases = [validate(dict(c), bounds) for c in data[recipe]]
    if len({c["id"] for c in cases}) != len(cases):
        raise CaseError("duplicate case IDs")
    if len(cases) > 32:
        raise CaseError("a matrix contains at most 32 cases")
    return cases


def _pool() -> list[list[int]]:
    return [row["token_ids"] for row in json.loads(POOL.read_text())["sequences"]]


def tokens(case: dict, *, tag: str, pool: list[list[int]] | None = None) -> dict:
    return _tokens(case, pool if pool is not None else _pool(), tag=tag)
