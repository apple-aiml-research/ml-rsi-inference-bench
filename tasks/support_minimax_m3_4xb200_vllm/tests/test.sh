#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Verifier entrypoint.
# Patching and grading live in tests/grader.py. This script owns the
# task-specific part: run the suites, write reports under /logs/verifier/,
# and apply any report fixups before grading.
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

timeout --kill-after=10 300 python3 -I /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

# Canonical raw-output log. Every suite's combined stdout+stderr SHOULD go here so the reason
# a test failed is never lost -- use run_log, or pipe through `tee -a "$RUN_LOG"` when feeding
# a reporter. Never 2>/dev/null a test run. The tail of this script cats this (and any other
# raw logs) into stdout.
export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
export PATH="/opt/vllm/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 CARGO_NET_OFFLINE=true
PY=/opt/vllm/bin/python
BASE="$(python3 -I -c 'import json; print(json.load(open("/tests/config.json"))["base_commit"])')"

# The verifier blocks the network itself, on every route: each phase below runs in a fresh network
# namespace with loopback up (unshare -r -n; netblock.py proves the outside unreachable, then execs the phase). Where
# the runtime refuses the namespace, the phases run as they are only if the outside is already unreachable (the
# backend's own block, e.g. network_mode none); otherwise the trial is void, never graded with the network reachable.
if unshare -r -n python3 -I /tests/infb_verify/netblock.py -- true 2>/dev/null; then
    NB=(unshare -r -n python3 -I /tests/infb_verify/netblock.py --)
elif python3 -I /tests/infb_verify/netblock.py --probe; then
    NB=()
else
    log "void: the outside is reachable and the runtime refuses a network namespace; no reward (sentinel -1)"
    exit 3
fi
export INFB_EVALUATION_ID="$(python3 -I -c 'import uuid; print(uuid.uuid4())')"

# Restore trusted public checks before importing candidate code. Candidate edits
# are useful during development; grading always uses the published implementation.
export BASE
run_log timeout --kill-after=10 300 bash -c '
    set -e
    rm -rf /app/eval
    mkdir -p /app/eval
    cp -a /tests/eval/minimax_m3 /app/eval/minimax_m3
    rm -rf /app/tests
    git checkout -q "$BASE" -- tests
    git clean -fdq -- tests
    git apply --whitespace=nowarn --allow-empty /tests/test.patch
' || exit 4

# Every bound below derives from the public operation limits (/tests/eval/minimax_m3/limits.json)
# through the verifier's deadline composition (/tests/infb_verify/deadline_dag.py):
# one source for alarms, clients, workers, subprocesses and group walls.
D() { python3 -I /tests/infb_verify/deadline_dag.py "$@"; }
REFERENCE_BOUND="$(D phase reference)" || exit 4

# The reference tree and its products were built into the verifier image before any candidate
# patch existed. Its component timing and both deployments' draws precede L0, which rebuilds the
# candidate's products.
run_log timeout --kill-after=30 600 "${NB[@]}" "$PY" /tests/preflight.py || exit 3
run_log timeout --kill-after=30 "$REFERENCE_BOUND" "${NB[@]}" "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json || exit 3

# One pytest group under the deadline runner: its budget, its report, its node prefix.
group() {
    local name="$1" report="$2" prefix="$3" root="$4"
    shift 4
    local budget outer
    budget="$(D group "$name")" && outer="$(D outer "$name")" || return 4
    local record_args=()
    if [[ "$name" == bf16kv || "$name" == fp8kv ]]; then
        record_args=(--record "/logs/verifier/efficiency/test_$name/efficiency.json")
    fi
    run_log timeout --kill-after=30 "$outer" "${NB[@]}" env PYTHONPATH=/tests:/tests/eval "$PY" -I /tests/infb_efficiency/deadline.py run \
        --group "$name" --budget-s "$budget" --config /tests/config.json \
        --junit "/logs/verifier/junit-$report.xml" \
        --results "/logs/verifier/$name-results.jsonl" \
        --summary "/logs/verifier/$name-deadline-summary.json" \
        "${record_args[@]}" --expect-prefix "$prefix" --cuda-devices 0,1,2,3 --always-reap -- \
        "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir="$root" -q -s \
        "$@" --junitxml="/logs/verifier/junit-$report.xml"
}

# L0: the documented rebuild (it bounds its own subprocesses), the supplied kernels, the registry, the config.
group l0 l0 infb_verify.test_l0. /tests /tests/infb_verify/test_l0.py
# MSA: correctness over the private cases, component efficiency against the reference phase's
# Triton-arm timing, sparsity; GPU 0, no server resident.
group msa msa infb_verify.test_msa. /tests /tests/infb_verify/test_msa.py
# The adapted parser tests (four conjunction nodes).
group parsers parsers infb_verify.test_parsers. /tests /tests/infb_verify/test_parsers.py
# Existing-engine regressions: the fixed L3 files and the Rust crates.
group l3 l3 tests. /app --continue-on-collection-errors \
    $(python3 -I -c 'import json; print(" ".join(json.load(open("/tests/config.json"))["infb"]["l3"]["files"]))')
run_log timeout --kill-after=30 "$(D outer rust)" "${NB[@]}" "$PY" -I /tests/infb_verify/rust_junit.py \
    --config /tests/config.json --budget-s "$(D group rust)" --junit /logs/verifier/junit-rust.xml
# The two deployments, one candidate launch each, twelve nodes in a fixed order.
group bf16kv bf16kv infb_verify.test_bf16kv. /tests /tests/infb_verify/test_bf16kv.py
group fp8kv fp8kv infb_verify.test_fp8kv. /tests /tests/infb_verify/test_fp8kv.py

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
