#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""This task's efficiency verifier: the paired measurement of the candidate against the human reference, on the same GPUs.

client.py          the measurement client (byte-identical to /app/eval/glm53_flash/serving/client.py)
reference.py       the reference-serving step, run by test.sh before any candidate code
snapshot.py        the tests/Dockerfile tail step that builds the reference tree
nodes.py           EfficiencyMixin: the two scored nodes of each hidden class
deadline.py        the per-node alarm (a pytest plugin) and the process-group runner
bounds.py          composes every bound in config.json from the verification DAG
canary.py          the environment canary around every draw
schema_check.py    the core record's JSON Schema check (efficiency.schema.json)
_paired_helper.py  the teardown, IPC-reaping and GPU-free functions the reference and the runner call
pool.jsonl.xz      the hidden natural-text pool (pool.provenance.json)

PROVENANCE.md says what this directory holds and what it measures. It depends on nothing outside this
directory and the Python standard library (plus `tokenizers` in the client and `torch` in the canary's
GPU reading).
"""
