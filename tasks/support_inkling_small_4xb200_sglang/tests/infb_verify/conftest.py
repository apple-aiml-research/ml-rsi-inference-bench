#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The registered launches' fixed node order: efficiency first on the fresh launch, profiling last.

pytest orders a unittest class's methods by name; this hook puts each class's nodes in this module's order and keeps
every other item where it was. The node ids do not change.
"""
from __future__ import annotations

PRODUCTION_ORDER = (
    "test_0_efficiency_decode_c1",
    "test_0_efficiency_throughput",
    "test_served_configuration",
    "test_api_probes",
    "test_gsm8k_hidden_slice",
    "test_gsm8k",  # upstream's GSM8K on the public rows: a diagnostic in the same launch
    "test_generation_replay",
    "test_prefix_branch_replay",
    "test_conv_state_restore",
    "test_author_fidelity",
    "test_long_context",
    "test_execution_witness",
)

DETERMINISTIC_LAST = ("test_conv_state_history",)
# By name alone `test_conv_state_history` would sort first and leave the helpers a cache it had filled.


def ordered(items: list) -> list:
    rank = {name: i for i, name in enumerate(PRODUCTION_ORDER)}
    out, i = [], 0
    while i < len(items):
        j = i + 1
        while j < len(items) and items[j].parent is items[i].parent:
            j += 1
        group = items[i:j]
        parent = getattr(group[0].parent, "name", "")
        if parent == "TestHiddenInklingProduction":
            group = sorted(group, key=lambda it: rank.get(it.name.split("[", 1)[0], len(rank)))
        elif parent == "TestHiddenInklingDeterministic":
            group = sorted(group, key=lambda it: it.name.split("[", 1)[0] in DETERMINISTIC_LAST)
        out += group
        i = j
    return out


def pytest_collection_modifyitems(session, config, items):  # noqa: ARG001
    items[:] = ordered(items)
