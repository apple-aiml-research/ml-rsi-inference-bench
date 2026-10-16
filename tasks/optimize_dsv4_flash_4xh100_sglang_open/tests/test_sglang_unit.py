#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Behavioral unit tests for the SGLang DeepSeek-V4-Flash engine fork (no GPU, no cost)."""
from __future__ import annotations
import importlib
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _runner(engine: str = "sglang"):
    os.environ["INFB_ENGINE"] = engine
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import _candidate_runner
    importlib.reload(_candidate_runner)
    return _candidate_runner


def test_scrub_hf_offline_pops_under_sglang():
    r = _runner("sglang")
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "KEEP": "yes"}
    out = r._scrub_hf_offline(env)
    assert "HF_HUB_OFFLINE" not in out, "HF_HUB_OFFLINE must be popped for sglang"
    assert "TRANSFORMERS_OFFLINE" not in out
    assert out["KEEP"] == "yes"


def test_scrub_hf_offline_noop_under_vllm():
    r = _runner("vllm")
    env = {"HF_HUB_OFFLINE": "1"}
    out = r._scrub_hf_offline(env)
    assert out.get("HF_HUB_OFFLINE") == "1", "vLLM behavior must be unchanged"
    _runner("sglang")


def test_set_offline_is_noop_under_sglang():
    r = _runner("sglang")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    r._set_offline()
    assert "HF_HUB_OFFLINE" not in os.environ, "sglang: _set_offline must NOT set os.environ HF_HUB_OFFLINE"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_engine_defaults_to_sglang_and_disables_live_correctness():
    r = _runner("sglang")
    assert r.ENGINE == "sglang"
    assert r.LIVE_CORRECTNESS_SUPPORTED is False


def test_extract_tps_requires_the_remap():
    import _paired_helper as ph
    assert not ph._extract_tps({"total_throughput": 12345.0}), \
        "_extract_tps unexpectedly knows total_throughput — remap-necessity assumption is stale"
    assert ph._extract_tps({"total_token_throughput": 12345.0}) == 12345.0


def test_derive_total_tps_from_sglang_in_out_tokens():
    r = _runner("sglang")
    data = {
        "total_input_tokens": 8000.0, "total_output_tokens": 2000.0, "duration": 2.0,
        "output_throughput": 1000.0, "request_throughput": 5.0,
    }
    tps = r._derive_total_token_throughput(data)
    assert tps == 5000.0, "must derive (in+out)/wall = 10000/2 = 5000, not decode-only 1000"
    assert tps != 1000.0


def test_derive_total_tps_last_resort_is_output_throughput():
    r = _runner("sglang")
    assert r._derive_total_token_throughput({"output_throughput": 1234.0}) == 1234.0
    assert r._derive_total_token_throughput({}) is None


def test_derive_total_tps_prefers_explicit_total():
    r = _runner("sglang")
    assert r._derive_total_token_throughput({"total_token_throughput": 999.0}) == 999.0
