#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""GPT-OSS's task-owned paired serving measurement.

client.py        the measurement client (byte-identical to /public/efficiency/client.py)
reference.py     the reference-serving step, run by test.sh before any candidate code
snapshot.py      the tests/Dockerfile tail step that builds the reference tree
nodes.py         EfficiencyMixin: the two scored nodes for a Day-0 hidden class
deadline.py      the per-node alarm (a pytest plugin) and the process-group runner
schema_check.py  the core record's JSON Schema check (efficiency.schema.json)
canary.py        the environment canary read around every draw on both sides
_paired_helper.py  the retained teardown, IPC-reaping and GPU-free functions
pool.jsonl.xz    the hidden natural-text pool (pool.provenance.json)

The task's own tests govern this package.
"""
