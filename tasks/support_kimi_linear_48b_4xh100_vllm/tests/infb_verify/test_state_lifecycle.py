#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The six state-lifecycle verdicts, emitted once, in one final process.

The candidate sessions (`tp4`, `dp2`, `apc`, `pressure`) and the late independent cold reference ran before this
module as their own bounded steps (`infb_verify/state_lifecycle.py`) and left raw records under
`/logs/state-lifecycle/`: client request/response records per family, the private observer's per-worker files and
the reference's teacher-forced values. Each function here reads them and decides one node; nothing here serves a
model. A node whose evidence a producer could not establish is an explicit non-result with its cause: a candidate
failure (the reached case's offending values), an observation, reference or verifier fault, or blocked by the
producer step that ended at its bound -- never a skip and never a pass (`records.py`, `attribution.py`).
"""

from __future__ import annotations

import contextlib

import pytest

from infb_verify import records
from infb_verify import state_lifecycle as lifecycle


@pytest.fixture(scope="module", autouse=True)
def _collect_observer_events():
    """After the six verdicts, copy the observer's produce and consume events into the collected records.

    Output only, and in a teardown so it runs whatever the verdicts were.
    """
    yield
    with contextlib.suppress(Exception):   # a collection failure must never change a verdict
        kept = lifecycle.collect_observer_events(lifecycle.OBSERVER, records.ROOT / "observer-events")
        print(f"[observer-events] {kept['files']} files, {kept['events']} events, {kept['bytes']} bytes")


def _node(name: str) -> None:
    verdict = lifecycle.Verdict(name)

    def body(record: dict):
        try:
            lifecycle.CHECKS[name](verdict)
        except (records.ReferenceFault, records.ObservationFault, records.VerifierFault, records.Blocked):
            raise
        except Exception as exc:  # noqa: BLE001 - the verifier's own analysis, never the candidate's
            if records.stated_validation(exc):
                raise
            verdict.checker(exc)
        verdict.conclude(record)
        return True, verdict.summary

    records.run(name, {"node": name, "sessions": lifecycle.NODES[name],
                       "policy": str(lifecycle.EVAL / "state_lifecycle/policy.json"),
                       "T_lifecycle": lifecycle.LC.tolerance()}, body)


def test_native_state_metadata_and_replay():
    """Builder-driven validity, turnover, branch and replay schedules on every TP4 rank, and the replay tail
    on both DP2 replicas, under the KDA numerical policy."""
    _node("test_native_state_metadata_and_replay")


def test_fresh_request_state_lifecycle():
    """On TP4, DP2 and APC: fresh one-token requests on reused state storage after observed turnover,
    continued short requests and fresh requests beside decoding ones, against the independent reference."""
    _node("test_fresh_request_state_lifecycle")


def test_dp2_correctness_under_load():
    """Two TP4 replicas doing real work, full decode graphs on both, idle/uneven execution, GSM8K, model-
    author fidelity, concurrency isolation and exact continuations of the load cases."""
    _node("test_dp2_correctness_under_load")


def test_prefix_reuse_and_continuations():
    """Families A1, A3, A4, A5, A6: deliberate real restores at completed retained boundaries, readiness, generated
    history, mixed cached/fresh waves, and every continuation against the independent cold reference."""
    _node("test_prefix_reuse_and_continuations")


def test_prefix_branch_state_and_binding():
    """Family A2 and the restored-state interventions: branches restore the unchanged checkpoint, and perturbing
    the restored recurrent or convolution state changes the continuation while the checkpoint and a sibling stay
    unchanged."""
    _node("test_prefix_branch_state_and_binding")


def test_prefix_turnover_and_resume():
    """Under apc-pressure: turnover then a new hit, resumption with generated history and reuse of it,
    cancellation and release."""
    _node("test_prefix_turnover_and_resume")
