#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Kimi Linear's task-owned whole-model efficiency gate, derived from the shared Day-0 efficiency kit.

client.py        the measurement client (byte-identical to /app/eval/kimi_linear/serving/client.py)
canary.py        the host canary (byte-identical to /app/eval/kimi_linear/serving/canary.py)
reference.py     the reference-serving step, run by test.sh before any candidate code
snapshot.py      the tests/Dockerfile tail step that builds the reference tree
nodes.py         run_axis: the measurement behind the two scored nodes of infb_verify/test_hidden.py
deadline.py      the per-node alarm (a pytest plugin) and the process-group runner
schema_check.py  the core record's JSON Schema check (efficiency.schema.json)
_paired_helper.py  the benchmark's own launch-retry, teardown, IPC-reaping and GPU-free functions
pool.jsonl.xz    the hidden natural-text pool (pool.provenance.json)

The task owns this code.
"""
