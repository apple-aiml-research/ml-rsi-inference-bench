#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The fp8kv deployment (FP8 (E4M3) KV cache): one candidate launch at the public recipe serves the twelve nodes below, in this
order. Efficiency runs first on the fresh, unprofiled launch; profiling runs last. The node bodies are
shared with the other deployment (`serving_checks.py`)."""

from __future__ import annotations

from infb_verify import serving_checks as S

DEPLOYMENT = "fp8kv"
#: the key of this deployment in config.json `infb.efficiency.recipes` (the scoring nodes' junit classname)
RECIPE = "test_fp8kv"


def test_0_efficiency_decode_c1(session):
    S.efficiency(session, "decode_c1", RECIPE)


def test_0_efficiency_throughput(session):
    S.efficiency(session, "throughput", RECIPE)


def test_served_configuration(session):
    S.served_configuration(session)


def test_api_probes(session):
    S.api_probes(session)


def test_chat_reasoning(session):
    S.chat_reasoning(session)


def test_tool_calls(session):
    S.tool_calls(session)


def test_generation_replay(session):
    S.generation_replay(session)


def test_prefix_branch_replay(session):
    S.prefix_branch_replay(session)


def test_author_fidelity(session):
    S.author_fidelity(session)


def test_engine_agreement_262k(session):
    S.engine_agreement(session)


def test_gsm8k_chat_thinking(session):
    S.gsm8k(session)


def test_execution_witness(session):
    S.execution_witness(session)
