#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The launch configuration a server reports back, compared with the configuration it was launched with."""
from __future__ import annotations

from typing import Any

import requests

NOT_RECIPE = frozenset({"host", "port", "device", "model_path", "api_key"})


def recipe_flags(argv: list[str]) -> dict[str, Any]:
    """`--name value` pairs and bare `--flag` switches of a launch argv, keyed by the engine's field name
    (`--moe-a2a-backend` -> `moe_a2a_backend`). A switch expects True; a value keeps its argv string."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ValueError(f"unexpected argv token {tok!r} at position {i}: {argv!r}")
        name = tok[2:].replace("-", "_")
        if "=" in name:
            name, value = name.split("=", 1)
            out[name] = value
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[name] = argv[i + 1]
            i += 2
        else:
            out[name] = True
            i += 1
    return {k: v for k, v in out.items() if k not in NOT_RECIPE}


def expected_fields(argv: list[str], resolved: dict[str, Any] | None = None) -> dict[str, Any]:
    """The argv's fields, plus each `resolved` field the argv does not name."""
    expected = recipe_flags(list(argv))
    for key, value in dict(resolved or {}).items():
        expected.setdefault(key, value)
    return expected


def server_info(base_url: str, timeout: float = 120) -> dict:
    return requests.get(base_url + "/server_info", timeout=timeout).json()


def _same(expected: Any, served: Any) -> bool:
    if expected is True:
        return served is True
    if isinstance(served, bool) or served is None:
        return False
    return str(served) == str(expected)


def compare(expected: dict[str, Any], info: dict) -> dict:
    """Every expected flag against the top-level resolution and each data-parallel scheduler's `internal_states`
    entry."""
    views: list[tuple[str, dict]] = [("top_level", info)]
    states = info.get("internal_states")
    if not isinstance(states, list) or not states:
        return {"expected": expected, "mismatches": [{"view": "internal_states", "flag": None,
                                                      "expected": "a non-empty list", "served": states}],
                "n_dp_schedulers": 0}
    views += [(f"internal_states[{i}]", s if isinstance(s, dict) else {}) for i, s in enumerate(states)]
    mismatches = []
    for view, d in views:
        for flag, want in expected.items():
            got = d.get(flag, "<absent>")
            if not _same(want, got):
                mismatches.append({"view": view, "flag": flag, "expected": want, "served": got})
    return {"expected": expected, "mismatches": mismatches, "n_dp_schedulers": len(states)}
