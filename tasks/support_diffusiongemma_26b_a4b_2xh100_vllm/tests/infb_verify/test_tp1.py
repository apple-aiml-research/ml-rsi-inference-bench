#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Deployment A (TP1): the tp1_fixed session, then the tp1 session, in the verifier's session order.

Every node's expected verdict on the reference is pass. A dead server makes the later nodes of its
session fail as blocked_by; no node relaunches a server.
"""
from __future__ import annotations

import pytest

from infb_verify import common, serving_nodes as N
from infb_verify.common import SESSIONS

POLICY = common.policy()


def _bars(key: str, deployment: str = "tp1"):
    return None if POLICY is None else POLICY["reads"][key][deployment]


def _record(name, rec):
    common.record(name, rec)
    assert rec["pass"], {k: v for k, v in rec.items() if k not in ("draws",)}


def test_0_efficiency_decode_c1_tp1():
    from infb_verify import efficiency
    srv = SESSIONS.get("tp1_fixed")
    _record("test_0_efficiency_decode_c1[tp1]", efficiency.node(srv, "tp1", "c1"))


def test_0_efficiency_throughput_tp1():
    from infb_verify import efficiency
    srv = SESSIONS.get("tp1_fixed")
    _record("test_0_efficiency_throughput[tp1]", efficiency.node(srv, "tp1", "c4"))


def test_denoise_policy_fixed_half():
    from infb_verify import policy_node
    srv = SESSIONS.get("tp1_fixed")
    policy_node.FIXED = policy_node.run_fixed(srv)
    common.record("test_denoise_policy_fixed_half", policy_node.FIXED)


def test_served_configuration_tp1():
    _record("test_served_configuration[tp1]", N.served_configuration(SESSIONS.get("tp1"), "tp1"))


def test_api_probes_tp1():
    _record("test_api_probes[tp1]", N.api_probes(SESSIONS.get("tp1")))


def test_repair_structured_outputs():
    _record("test_repair_structured_outputs", N.repair_structured_outputs(SESSIONS.get("tp1")))


def test_repair_long_output():
    _record("test_repair_long_output", N.repair_long_output(SESSIONS.get("tp1")))


def test_repair_first_token():
    _record("test_repair_first_token", N.repair_first_token(SESSIONS.get("tp1"), common.hidden("first_token_cases.json")["cases"]))


def test_concurrent_logprobs():
    _record("test_concurrent_logprobs", N.concurrent_logprobs(SESSIONS.get("tp1")))


def test_chat_reasoning():
    _record("test_chat_reasoning", N.chat_reasoning(SESSIONS.get("tp1"), common.hidden("chat_cases.json")["reasoning"]))


def test_tool_calls():
    from infb_verify import tools_node
    _record("test_tool_calls", tools_node.run(SESSIONS.get("tp1"), common.hidden("chat_cases.json")["tools"]))


def test_denoise_reads_tp1():
    _record("test_denoise_reads[tp1]", N.reads_node(SESSIONS.get("tp1"), "reads", "tp1", _bars("reads")))


def test_image_reads_tp1():
    _record("test_image_reads[tp1]", N.reads_node(SESSIONS.get("tp1"), "image", "tp1", _bars("image"), chat=True))


def test_step1_reads_tp1():
    _record("test_step1_reads[tp1]", N.step1_node(SESSIONS.get("tp1"), "tp1", _bars("step1"), _bars("step1_increment")))


def test_commit_encode_replay_tp1():
    from infb_verify import replay
    _record("test_commit_encode_replay[tp1]", replay.replay(SESSIONS.get("tp1"), "tp1", POLICY))


def test_stale_phase_leak_tp1():
    from infb_verify import replay
    _record("test_stale_phase_leak[tp1]", replay.leak(SESSIONS.get("tp1"), "tp1", POLICY))


def test_prefix_branches():
    from infb_verify import replay
    _record("test_prefix_branches", replay.branches(SESSIONS.get("tp1"), POLICY))


def test_mixed_batch_reads_tp1():
    from infb_verify import replay
    _record("test_mixed_batch_reads[tp1]", replay.mixed(SESSIONS.get("tp1"), "tp1", POLICY))


def test_semantic_majority_tp1():
    from infb_verify import semantic
    _record("test_semantic_majority[tp1]", semantic.run(SESSIONS.get("tp1"), "tp1"))


def test_denoise_policy():
    from infb_verify import policy_node
    _record("test_denoise_policy", policy_node.verdict(SESSIONS.get("tp1"), POLICY))


def test_long_context_tp1():
    from infb_verify import replay
    _record("test_long_context[tp1]", replay.long_context(SESSIONS.get("tp1"), "tp1", POLICY))


def test_gsm8k():
    floor = None if POLICY is None else POLICY["gsm8k_floor"]
    _record("test_gsm8k", N.gsm8k(SESSIONS.get("tp1"), floor))


def test_execution_witness_tp1():
    from infb_verify import witness_node
    _record("test_execution_witness[tp1]", witness_node.run("tp1"))


def test_preemption_resume():
    _record("test_preemption_resume", N.preemption_node(SESSIONS.get("tp1"), POLICY))


def teardown_module(module):
    SESSIONS.stop()
