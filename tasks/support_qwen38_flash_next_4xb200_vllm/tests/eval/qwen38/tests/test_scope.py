#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The examples stay inside the public scope; the budgets are well formed."""
import json
from pathlib import Path

from qwen38.measurement import budget, cases

ROOT = Path(__file__).resolve().parents[1]


def test_component_examples_are_in_scope():
    data = json.loads((ROOT / "examples/components.json").read_text())
    bounds = cases.scope()
    for recipe in ("ordinary", "mtp3"):
        for case in data[recipe]:
            cases.validate(dict(case), bounds)
    assert {c["mode"] for c in data["mtp3"]} == {"prefill", "decode", "spec"}


def test_every_budget_is_a_positive_number_of_seconds():
    data = json.loads((ROOT / "budgets.json").read_text())
    assert data["operations"] and all(type(r["seconds"]) is int and r["seconds"] > 0 for r in data["operations"].values())
    assert budget.seconds("component_collection") == 1800


def test_an_operation_bounds_every_request_inside_it():
    with budget.operation(5):
        assert budget.timeout(600) <= 5
        with budget.operation(1800):
            assert budget.timeout(600) <= 5          # an inner operation never outlives its parent
    assert budget.timeout(600) == 600
