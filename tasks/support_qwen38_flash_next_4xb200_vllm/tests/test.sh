#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Verifier entrypoint. Patching and grading live in tests/grader.py. This
# script owns the task-specific part: run the suites, write reports under
# /logs/verifier/, and apply any report fixups before grading.
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

timeout --kill-after=10 300 python3 -I /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

# Canonical raw-output log. The task middle SHOULD send every suite's combined
# stdout+stderr here so the reason a test failed is never lost -- use run_log,
# or pipe through `tee -a "$RUN_LOG"` when feeding a reporter. Never 2>/dev/null
# a test run. FRAME_SUFFIX cats this (and any other raw logs) into test-stdout.
export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
export PATH="/opt/vllm/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 CARGO_NET_OFFLINE=true
export INFB_EVALSCOPE_DATASET_ROOT=/public/datasets
PY=/opt/vllm/bin/python
BASE="$(python3 -I -c 'import json; print(json.load(open("/tests/config.json"))["base_commit"])')"
export INFB_EVALUATION_ID="$(python3 -I -c 'import uuid; print(uuid.uuid4())')"

# Restore trusted public checks before importing candidate code. Candidate edits
# are useful during development; grading always uses the published implementation.
export BASE
run_log timeout --kill-after=10 300 bash -c '
    set -e
    rm -rf /app/eval
    mkdir -p /app/eval
    cp -a /tests/eval/qwen38 /app/eval/qwen38
    rm -rf /app/tests
    git checkout -q "$BASE" -- tests
    git clean -fdq -- tests
    git apply --whitespace=nowarn --allow-empty /tests/test.patch
' || exit 4

# Every bound below derives from the public workload budgets (/tests/eval/qwen38/budgets.json)
# through the verifier's composition (/tests/infb_verify/budgets.py): one source for alarms,
# clients, subprocesses and group walls.
B() { python3 -I /tests/infb_verify/budgets.py "$@"; }
REFERENCE_BOUND="$(B phase reference)" || exit 4

# Reference source/products were snapshotted before any candidate patch existed.
# Its complete isolated launch precedes L0, which may rebuild candidate products.
run_log timeout --kill-after=30 600 "$PY" /tests/preflight.py || exit 3
run_log timeout --kill-after=30 "$REFERENCE_BOUND" "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json || exit 3

run_group() {
    local group="$1" report="$2" prefix="$3" root="$4"
    local budget outer
    budget="$(B group "$group")" && outer="$(B outer "$group")" || return 4
    local record_args=()
    if [[ "$group" == test_hidden || "$group" == test_mtp ]]; then
        record_args=(--record "/logs/verifier/efficiency/$group/efficiency.json")
    fi
    if [[ "$group" == test_hidden ]]; then
        record_args+=(--expect-prefix infb_verify.test_fidelity.)
    fi
    shift 4
    run_log timeout --kill-after=30 "$outer" env PYTHONPATH=/tests:/tests/eval "$PY" -I /tests/infb_efficiency/deadline.py run \
        --group "$group" --budget-s "$budget" --config /tests/config.json \
        --junit "/logs/verifier/junit-$report.xml" \
        --results "/logs/verifier/$group-results.jsonl" \
        --summary "/logs/verifier/$group-deadline-summary.json" \
        "${record_args[@]}" --expect-prefix "$prefix" --cuda-devices 0,1,2,3 --always-reap -- \
        "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir="$root" -q -s \
        "$@" --junitxml="/logs/verifier/junit-$report.xml"
}
# L0 bounds its own subprocesses; the generic call alarm must not cap a build.
run_log timeout --kill-after=30 "$(B outer l0)" "$PY" -I /tests/infb_efficiency/deadline.py run --group l0 \
    --budget-s "$(B group l0)" --config /tests/config.json --junit /logs/verifier/junit-l0.xml \
    --results /logs/verifier/l0-results.jsonl --summary /logs/verifier/l0-deadline-summary.json \
    --expect-prefix infb_verify.test_l0. --cuda-devices 0,1,2,3 --always-reap -- \
    "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q /tests/infb_verify/test_l0.py \
    --junitxml=/logs/verifier/junit-l0.xml

# Continue after candidate failures so every scored obligation has a result.
run_log timeout --kill-after=30 "$(B outer l3)" "$PY" -I /tests/infb_efficiency/deadline.py run --group l3 \
    --budget-s "$(B group l3)" --config /tests/config.json --junit /logs/verifier/junit-l3.xml \
    --results /logs/verifier/l3-results.jsonl --summary /logs/verifier/l3-deadline-summary.json \
    --expect-prefix tests.v1. --expect-prefix tests.kernels. --expect-prefix tests.models.test_registry. \
    --expect-prefix tests.quantization. --cuda-devices 0,1,2,3 --always-reap -- \
    "$PY" -m pytest -p no:cacheprovider --continue-on-collection-errors --rootdir=/app -q \
    tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_single_type_kv_cache_manager.py \
    tests/v1/core/test_prefix_caching.py tests/v1/core/test_scheduler.py \
    tests/v1/worker/test_mamba_utils.py tests/kernels/mamba/test_precopy_mamba_align.py \
    tests/kernels/moe/test_block_fp8.py tests/models/test_registry.py tests/quantization/test_fp8.py \
    --junitxml=/logs/verifier/junit-l3.xml

# Whole-serving performance runs first on each grading launch. The same session then performs
# API/quality checks and the complete native component collection. The retained MTP3 production
# interventions and runner coverage use their own diagnostic launch.
run_group test_hidden hidden infb_verify.test_hidden. /tests \
    /tests/infb_verify/test_hidden.py /tests/infb_verify/test_fidelity.py
run_group test_mtp mtp infb_verify.test_mtp. /tests /tests/infb_verify/test_mtp.py
run_group native native infb_verify.test_native. /tests /tests/infb_verify/test_native.py
run_group l2_accuracy l2 tests.evals.qwen4_exp.test_accuracy. /app \
    tests/evals/qwen4_exp/test_accuracy.py --config-list-file=configs/models-b200.txt

# The reward disposition: a valid candidate failure anywhere leaves the grader's
# reward; otherwise a censored or blocked scored node voids the trial (the -1 sentinel), never a zero.
python3 -I /tests/infb_verify/disposition.py || { log "void: see /logs/verifier/disposition.json; no reward (sentinel -1)"; exit 3; }
# >>> END RUN TESTS <<<

# Surface raw suite output into our stdout (the harness captures it into
# test-stdout.txt) so failures are debuggable even when the framework report
# omits the reason (e.g. cargo-nextest). Reasons-per-test come from grade below.
_seen=""
for _rl in "$RUN_LOG" /logs/verifier/*_run.log /logs/verifier/*-run.log /logs/verifier/*-mocha.log /logs/verifier/*.log /logs/verifier/*.out; do
  [ -f "$_rl" ] && [ -s "$_rl" ] || continue
  case " $_seen " in *" $_rl "*) continue ;; esac
  case "${_rl##*/}" in *convert*.log|ctrf*.log|junit*.log) continue ;; esac
  _seen="$_seen $_rl"
  echo "===== raw suite output: ${_rl##*/} ====="
  cat "$_rl"
done 2>/dev/null
echo "===== grade ====="

# Grading is bounded too; a grade that never ends leaves no reward.json and the trap writes -1.
timeout --kill-after=10 110 python3 /tests/grader.py grade
log "reward.json=$(cat /logs/verifier/reward.json 2>/dev/null)"

# Uniform top level: keep only the canonical artifacts at /logs/verifier and
# tuck every framework-native report/log under reports/ (full provenance, no
# data dropped -- just moved). Canonical: reward.json, ctrf.json, run.log, and
# the harness-written test-stdout.txt.
mkdir -p /logs/verifier/reports 2>/dev/null
for _f in /logs/verifier/*; do
  case "${_f##*/}" in
    reward.json|reward.txt|ctrf.json|run.log|test-stdout.txt|reports) continue ;;
  esac
  [ -f "$_f" ] && mv -f "$_f" /logs/verifier/reports/ 2>/dev/null
done
