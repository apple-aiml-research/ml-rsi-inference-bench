#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Operations and their limits: every request, measurement invocation, test-file process and action the verifier runs
is one operation with a public wall-clock limit (`eval/deepseek_v4_flash/limits.json`, the trusted copy under
`/tests/eval`). A node's wall is a separate, larger bound (config.json `infb.deadline_dag`), and the two expiries mean
different things: an operation that exceeds its own limit on a healthy host is a candidate resource failure, while a
node or group wall that expires while every operation is still within its limit is a verifier fault (censored), never
a candidate failure."""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

_RUNNING: list[dict] = []
_LIMITS: dict | None = None


class OperationLimitExceeded(AssertionError):
    """An operation took longer than its public limit: a candidate resource failure."""

    status = "candidate_fail"


def limits() -> dict:
    """The trusted limits: `/tests/eval/deepseek_v4_flash/limits.json` (TESTS_DIR), falling back to the copy beside
    the task's `tests/` when this module is used from the source tree."""
    global _LIMITS
    if _LIMITS is None:
        tests = Path(os.environ.get("TESTS_DIR", "/tests"))
        here = Path(__file__).resolve().parent.parent
        for p in (tests / "eval" / "deepseek_v4_flash" / "limits.json", here / "eval" / "deepseek_v4_flash" / "limits.json"):
            if p.is_file():
                _LIMITS = json.loads(p.read_text())
                break
        else:
            raise RuntimeError("no limits.json under TESTS_DIR/eval/deepseek_v4_flash or beside this package")
    return _LIMITS


def limit_s(kind: str) -> float:
    return float(limits()["operations"][kind]["limit_s"])


def request_kind(n_input_tokens: int) -> str:
    """The HTTP operation kind for a request whose largest input has `n_input_tokens` tokens."""
    return "http_request_long" if n_input_tokens >= int(limits()["long_input_tokens"]) else "http_request"


@contextlib.contextmanager
def operation(kind: str, label: str, cap_s: float | None = None):
    """Run one operation of `kind` (a limits.json key) under its limit; see the module docstring."""
    entry = {"kind": kind, "label": label, "cap_s": float(cap_s if cap_s is not None else limit_s(kind)),
             "t0": time.monotonic()}
    _RUNNING.append(entry)
    ok = False
    try:
        yield entry
        ok = True
    finally:
        with contextlib.suppress(ValueError):
            _RUNNING.remove(entry)
        elapsed = time.monotonic() - entry["t0"]
        entry["elapsed_s"] = round(elapsed, 3)
        if ok and elapsed > entry["cap_s"]:
            raise OperationLimitExceeded(f"{label}: the {kind} operation took {elapsed:.1f} s, over its "
                                         f"{entry['cap_s']:g} s limit")


def attribution(now: float | None = None) -> dict:
    """At a node or group wall's expiry: `candidate_fail` if a running operation is past its own limit, else
    `verifier_fault` (censored), with the running operations named."""
    now = time.monotonic() if now is None else now
    running = [{"kind": e["kind"], "label": e["label"], "cap_s": e["cap_s"], "elapsed_s": round(now - e["t0"], 3)}
               for e in _RUNNING]
    over = [r for r in running if r["elapsed_s"] > r["cap_s"]]
    return {"status": "candidate_fail" if over else "verifier_fault", "censored": not over,
            "over_limit": over, "running": running}
