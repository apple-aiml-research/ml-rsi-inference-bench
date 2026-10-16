#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The six scored component nodes. Each node runs the candidate arm (the tree at /app,
built by L0) through `kernel_worker.py` in a process of its own -- under the node's public operation limit, with the
candidate's own JIT caches and one device -- and then judges the recorded outputs with `kernel_verdict.py` here, in the
verifier's process, which never imports an implementation under test. The efficiency nodes pair every family with the
reference arm's measurement of the same family and seed, recorded by the isolated reference phase before any
candidate code ran (`kernel_phase.py reference`)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_cases as kc  # noqa: E402
import kernel_phase as kp  # noqa: E402


def test_sparse_bf16_correctness():
    kp.assert_node(kp.candidate_node(kc.NODE_BF16))


def test_sparse_paged_correctness():
    kp.assert_node(kp.candidate_node(kc.NODE_PACKED))


def test_compression_state_correctness():
    kp.assert_node(kp.candidate_node(kc.NODE_STATE))


def test_cache_transform_correctness():
    kp.assert_node(kp.candidate_node(kc.NODE_TRANSFORM))


def test_kernel_prefill_efficiency():
    kp.assert_node(kp.candidate_node(kc.NODE_PREFILL))


def test_kernel_decode_efficiency():
    kp.assert_node(kp.candidate_node(kc.NODE_DECODE))
