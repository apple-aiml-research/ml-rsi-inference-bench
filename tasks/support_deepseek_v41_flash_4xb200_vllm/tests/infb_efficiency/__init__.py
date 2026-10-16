#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""support_deepseek_v41_flash_4xb200_vllm's own efficiency gate: the paired decode and throughput comparison.

client.py        the measurement client; the agent's copy is /app/eval/deepseek_v41_flash/serving/client.py, the same bytes
reference.py     the reference-serving step, run by test.sh before any candidate code
snapshot.py      the tests/Dockerfile tail step that builds the reference tree
nodes.py         the efficiency nodes' logic: measurement, the relaunch, the verdicts
deadline.py      the per-node alarm (a pytest plugin) and the process-group runner
canary.py        the environment canary read around every draw
schema_check.py  the core record's JSON Schema check (efficiency.schema.json)
_paired_helper.py  the host-side cleanup, IPC-reaping and GPU-free helpers the steps above call
pool.jsonl.xz    the hidden natural-text pool (pool.provenance.json)

PROVENANCE.json records every file's digest and every local change. Nothing here imports code from
outside this directory.
"""
