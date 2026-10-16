#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Runs the ONE verifier this task has: the full correctness + throughput pipeline.
#
# The verifier writes reward.json + metrics.json under the standard inference-bench paths.
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
verifier_dir="${INFB_VERIFIER_DIR:-/logs/verifier}"
artifacts_dir="${INFB_ARTIFACTS_DIR:-/logs/artifacts}"
tests_dir="${INFB_TESTS_DIR:-/tests}"

mkdir -p "$verifier_dir" "$artifacts_dir/raw"

# JIT caches are working state: count them into raw/jit_cache_summary.json, then remove them.
agent_dir="${INFB_AGENT_DIR:-/logs/agent}"
prune_jit_caches() {
    python "$tests_dir/_verifier_lib/jit_cache_prune.py" \
        --record "$artifacts_dir/raw/jit_cache_summary.json" "$artifacts_dir" "$agent_dir" \
        || echo "[test.sh] JIT cache prune failed; the caches stay in the collected directories"
}
trap 'rc=$?; prune_jit_caches; exit "$rc"' EXIT

echo "[test.sh] Real-engine mode."
python "$tests_dir/verify_throughput_real.py" \
    --workspace "$workspace" \
    --tests-dir "$tests_dir" \
    --reward "$verifier_dir/reward.json" \
    --metrics "$artifacts_dir/metrics.json" \
    --raw-dir "$artifacts_dir/raw"
