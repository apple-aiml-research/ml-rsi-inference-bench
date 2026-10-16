#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""This task's own efficiency verification: the measurement client, the reference phase and the scored nodes.

client.py          the measurement client (the same bytes as the copy the agent is given under /app)
reference.py       the reference-serving step, run by test.sh before any candidate code
snapshot.py        the tests/Dockerfile tail step that builds the reference tree
nodes.py           EfficiencyMixin: the two scored nodes for the hidden class
deadline.py        the per-node alarm (a pytest plugin) and the process-group runner
operations.py      the operation registry: every request, invocation, test file and action with its public limit
schema_check.py    the core record's JSON Schema check (efficiency.schema.json)
_paired_helper.py  the teardown, IPC-reaping and GPU-free functions the reference and the runner call
pool.jsonl.xz      the hidden natural-text pool (pool.provenance.json)

Nothing in building, running or grading this task depends on code outside this directory.
"""
