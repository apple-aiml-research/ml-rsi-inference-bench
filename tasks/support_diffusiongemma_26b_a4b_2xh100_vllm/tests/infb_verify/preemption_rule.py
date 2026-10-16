#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The preemption node's validity rule; pure, so that its logic can be exercised on constructed
records.

A run is one submission of a load, as serving/preemption.py records it: "pass" (every request completed with
max_tokens and the probe answered) and "preemptions" (the server's count over the run; None when the counter is
unreadable, which counts as none). The node submits the load of record; a run that passes and records no preemption
permits one re-run at the heavier load. The node passes iff every run passes and some run recorded at least one
preemption: zero, then zero, fails as not exercised.
"""
from __future__ import annotations


def preempted(run: dict) -> bool:
    return (run.get("preemptions") or 0) >= 1


def rerun(first: dict) -> bool:
    """Whether the load of record's run permits the re-run at the heavier load."""
    return bool(first.get("pass")) and not preempted(first)


def verdict(runs: list[dict]) -> dict:
    if not runs:
        return {"pass": False, "reason": "no run"}
    if not all(r.get("pass") for r in runs):
        return {"pass": False, "reason": "a request did not complete with max_tokens, or the probe failed"}
    if preempted(runs[0]):
        return {"pass": True, "reason": "exercised at the load of record"}
    if len(runs) > 1 and preempted(runs[1]):
        return {"pass": True, "reason": "exercised at the heavier load"}
    return {"pass": False, "reason": "not exercised: no preemption at the load of record"
            + (" or at the heavier load" if len(runs) > 1 else "")}
