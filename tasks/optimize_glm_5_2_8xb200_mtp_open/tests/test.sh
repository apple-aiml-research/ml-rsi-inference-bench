#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Verifier entry point for optimize_glm_5_2_8xb200_mtp_open.
#
# Runs the ONE verifier this task has: the full correctness + throughput pipeline.
#
# If the verifier withholds reward.json on an infra void, or this file aborts, the default is NO
# SCORE rather than a zero.
#
# INFB_OFFICIAL is still baked into [verifier.env] and still has readers -- the profile-capture
# branch fails closed on it -- it just no longer decides which verifier runs.
#
# The verifier writes reward.json + metrics.json under the standard inference-bench paths.
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
verifier_dir="${INFB_VERIFIER_DIR:-/logs/verifier}"
artifacts_dir="${INFB_ARTIFACTS_DIR:-/logs/artifacts}"
tests_dir="${INFB_TESTS_DIR:-/tests}"

mkdir -p "$verifier_dir" "$artifacts_dir/raw"

echo "[test.sh] Real-engine mode."
python "$tests_dir/verify_throughput_real.py" \
    --workspace "$workspace" \
    --tests-dir "$tests_dir" \
    --reward "$verifier_dir/reward.json" \
    --metrics "$artifacts_dir/metrics.json" \
    --raw-dir "$artifacts_dir/raw"
