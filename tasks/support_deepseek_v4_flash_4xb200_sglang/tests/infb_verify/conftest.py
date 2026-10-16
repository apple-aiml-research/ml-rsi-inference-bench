#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The two long nodes run last.

They send the largest requests of the session: sequences of 32,768 to 262,144 tokens, sent whole with their prompt
log-probabilities.

This hook moves the two long nodes, in that order, after every other node of their class, and keeps pytest's order
among the rest. The node ids do not change, so neither do `f2p_node_ids` nor the grader's reading of the junit report.
The efficiency nodes still run first, and the first node's set-up still launches the server.
"""
from __future__ import annotations

#: The long nodes, in the order they run at the end of their class.
LONG_LAST = ("test_long_context_fidelity", "test_long_reach_fidelity")


def long_nodes_last(items: list) -> list:
    """`items` with every item named in LONG_LAST moved, in LONG_LAST's order, after the other items of its class (its
    `parent`); everything else keeps its place and its order."""
    out, i = [], 0
    while i < len(items):
        j = i + 1
        while j < len(items) and items[j].parent is items[i].parent:
            j += 1
        group = items[i:j]
        names = [it.name.split("[", 1)[0] for it in group]
        last = [it for name in LONG_LAST for it, n in zip(group, names) if n == name]
        out += [it for it, n in zip(group, names) if n not in LONG_LAST] + last
        i = j
    return out


def pytest_collection_modifyitems(session, config, items):  # noqa: ARG001
    items[:] = long_nodes_last(items)
