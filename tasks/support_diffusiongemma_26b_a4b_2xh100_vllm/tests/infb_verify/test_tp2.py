#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Deployment B (TP2): the tp2_fixed session, in the verifier's session order.

Every node's expected verdict on the reference is pass. GSM8K is TP1-only.
"""
from __future__ import annotations

from infb_verify import common, serving_nodes as N
from infb_verify.common import SESSIONS

POLICY = common.policy()
RECIPE = "tp2_fixed"


def _bars(key: str):
    return None if POLICY is None else POLICY["reads"][key]["tp2"]


def _record(name, rec):
    common.record(name, rec)
    assert rec["pass"], {k: v for k, v in rec.items() if k not in ("draws",)}


def test_0_efficiency_decode_c1_tp2():
    from infb_verify import efficiency
    _record("test_0_efficiency_decode_c1[tp2]", efficiency.node(SESSIONS.get(RECIPE), "tp2", "c1"))


def test_0_efficiency_throughput_tp2():
    from infb_verify import efficiency
    _record("test_0_efficiency_throughput[tp2]", efficiency.node(SESSIONS.get(RECIPE), "tp2", "c4"))


def test_served_configuration_tp2():
    _record("test_served_configuration[tp2]", N.served_configuration(SESSIONS.get(RECIPE), "tp2"))


def test_api_probes_tp2():
    _record("test_api_probes[tp2]", N.api_probes(SESSIONS.get(RECIPE)))


def test_denoise_reads_tp2():
    _record("test_denoise_reads[tp2]", N.reads_node(SESSIONS.get(RECIPE), "reads", "tp2", _bars("reads")))


def test_image_reads_tp2():
    _record("test_image_reads[tp2]", N.reads_node(SESSIONS.get(RECIPE), "image", "tp2", _bars("image"), chat=True))


def test_step1_reads_tp2():
    _record("test_step1_reads[tp2]", N.step1_node(SESSIONS.get(RECIPE), "tp2", _bars("step1"), _bars("step1_increment")))


def test_commit_encode_replay_tp2():
    from infb_verify import replay
    _record("test_commit_encode_replay[tp2]", replay.replay(SESSIONS.get(RECIPE), "tp2", POLICY))


def test_stale_phase_leak_tp2():
    from infb_verify import replay
    _record("test_stale_phase_leak[tp2]", replay.leak(SESSIONS.get(RECIPE), "tp2", POLICY))


def test_mixed_batch_reads_tp2():
    from infb_verify import replay
    _record("test_mixed_batch_reads[tp2]", replay.mixed(SESSIONS.get(RECIPE), "tp2", POLICY))


def test_semantic_majority_tp2():
    from infb_verify import semantic
    _record("test_semantic_majority[tp2]", semantic.run(SESSIONS.get(RECIPE), "tp2"))


def test_long_context_tp2():
    from infb_verify import replay
    _record("test_long_context[tp2]", replay.long_context(SESSIONS.get(RECIPE), "tp2", POLICY))


def test_execution_witness_tp2():
    from infb_verify import witness_node
    _record("test_execution_witness[tp2]", witness_node.run("tp2"))


def teardown_module(module):
    SESSIONS.stop()
