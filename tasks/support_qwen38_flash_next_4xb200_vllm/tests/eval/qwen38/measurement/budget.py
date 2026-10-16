#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Operation allowances from `budgets.json`: every request inside an operation gets what remains.

    with budget.operation("fidelity_evaluation"):
        ...every serving request made here has timeout min(its own cap, the time left)...

Operations nest (the inner end is never later than the outer one) and apply to every thread of the
process, so concurrent requests of one operation share its allowance. A request made after the
allowance is spent raises `OperationBudgetExceeded` instead of starting.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path

BUDGETS = Path(__file__).resolve().parents[1] / "budgets.json"
_LOCK = threading.Lock()
_ENDS: list[float] = []


class OperationBudgetExceeded(TimeoutError):
    """The enclosing operation's allowance was spent before this request."""


def seconds(operation: str) -> int:
    return json.loads(BUDGETS.read_text())["operations"][operation]["seconds"]


@contextlib.contextmanager
def operation(name_or_seconds):
    limit = seconds(name_or_seconds) if isinstance(name_or_seconds, str) else float(name_or_seconds)
    with _LOCK:
        end = time.monotonic() + limit
        if _ENDS:
            end = min(end, _ENDS[-1])
        _ENDS.append(end)
    try:
        yield end
    finally:
        with _LOCK:
            _ENDS.remove(end)


def timeout(cap: float) -> float:
    """A request's timeout: its own cap, or less when the enclosing operation has less left."""
    with _LOCK:
        end = _ENDS[-1] if _ENDS else None
    if end is None:
        return cap
    left = end - time.monotonic()
    if left <= 0:
        raise OperationBudgetExceeded("the enclosing operation's allowance is spent")
    return min(cap, left)
