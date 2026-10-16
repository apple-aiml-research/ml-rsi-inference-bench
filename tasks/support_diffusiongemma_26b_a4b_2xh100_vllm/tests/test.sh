#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }
# An index the container runtime left unreadable is rebuilt from HEAD before anything reads it;
# a valid index is left untouched.
git status >/dev/null 2>&1 || { rm -f .git/index && git read-tree HEAD; }

timeout --kill-after=10 300 python3 -I /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
export PATH="/opt/vllm/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 CARGO_NET_OFFLINE=true PYTHONDONTWRITEBYTECODE=1
PY=/opt/vllm/bin/python
BASE="$(python3 -I -c 'import json; print(json.load(open("/tests/config.json"))["base_commit"])')"
export INFB_EVALUATION_ID="$(python3 -I -c 'import uuid; print(uuid.uuid4())')"

# Restore the trusted evaluation code before importing candidate code, and tests/ from the base
# plus test.patch. A pulled image's index carries stale stats, so it is refreshed before anything
# is applied.
export BASE
run_log timeout --kill-after=10 300 bash -c '
    set -e
    git update-index -q --refresh >/dev/null 2>&1 || true
    rm -rf /app/eval/diffusion_gemma
    mkdir -p /app/eval
    cp -a /tests/eval/diffusion_gemma /app/eval/diffusion_gemma
    rm -rf /app/tests
    git checkout -q "$BASE" -- tests
    git clean -fdq -- tests
    git apply --whitespace=nowarn --allow-empty /tests/test.patch
' || exit 4

# Every bound below derives from the public operation limits (/tests/eval/diffusion_gemma/budgets.json)
# through the verifier's composition (/tests/infb_verify/budgets.py): one source for alarms, clients,
# subprocesses and group walls.
B() { python3 -I /tests/infb_verify/budgets.py "$@"; }
REFERENCE_BOUND="$(B phase reference)" || exit 4

# The reference tree was snapshotted into the verifier image before any candidate patch existed; its
# comparator arm and its fixed-configuration launches precede L0, which may rebuild candidate products.
run_log timeout --kill-after=30 600 "$PY" /tests/preflight.py || exit 3
run_log timeout --kill-after=30 "$REFERENCE_BOUND" "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json || exit 3

run_group() {
    local group="$1" report="$2" prefix="$3" root="$4"
    local budget outer
    budget="$(B group "$group")" && outer="$(B outer "$group")" || return 4
    shift 4
    run_log timeout --kill-after=30 "$outer" env PYTHONPATH=/tests:/tests/eval "$PY" -I /tests/infb_efficiency/deadline.py run \
        --group "$group" --budget-s "$budget" --config /tests/config.json \
        --junit "/logs/verifier/junit-$report.xml" \
        --results "/logs/verifier/$group-results.jsonl" \
        --summary "/logs/verifier/$group-deadline-summary.json" \
        --expect-prefix "$prefix" --cuda-devices 0,1 --always-reap -- \
        env PYTHONPATH=/tests:/tests/eval "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir="$root" -q -s \
        "$@" --junitxml="/logs/verifier/junit-$report.xml"
}

# Continue after candidate failures so every scored obligation has a result.
run_group l0 l0 infb_verify.test_l0. /tests /tests/infb_verify/test_l0.py
run_group attend attend infb_verify.test_attend. /tests /tests/infb_verify/test_attend.py
run_group parsers parsers infb_verify.test_parsers. /tests /tests/infb_verify/test_parsers.py
L3_FILES="$(python3 -I -c 'import json; print(" ".join(json.load(open("/tests/config.json"))["infb"]["l3"]["files"]))')"
L3_K="$(python3 -I -c 'import json; print(json.load(open("/tests/config.json"))["infb"]["l3"]["k"])')"
# The fork's cute tests import flash_attn.cute, and the fork's package __init__ loads its own FA2 extension,
# which this build does not ship (vLLM carries it as _vllm_fa2_C). A shim package holding only the cute link
# imports the tree's vendored sources; its flash-attn 0.0.0 dist-info keeps transformers' probe answering absent.
FA_SHIM=/tmp/infb-fa-shim
mkdir -p "$FA_SHIM/flash_attn" "$FA_SHIM/flash_attn-0.0.0.dist-info"
: > "$FA_SHIM/flash_attn/__init__.py"
ln -sfn /app/third_party/vllm-flash-attn/flash_attn/cute "$FA_SHIM/flash_attn/cute"
printf 'Metadata-Version: 2.1\nName: flash-attn\nVersion: 0.0.0\n' > "$FA_SHIM/flash_attn-0.0.0.dist-info/METADATA"
printf 'flash_attn\n' > "$FA_SHIM/flash_attn-0.0.0.dist-info/top_level.txt"
# shellcheck disable=SC2086
run_log timeout --kill-after=30 "$(B outer l3)" "$PY" -I /tests/infb_efficiency/deadline.py run --group l3 \
    --budget-s "$(B group l3)" --config /tests/config.json --junit /logs/verifier/junit-l3.xml \
    --results /logs/verifier/l3-results.jsonl --summary /logs/verifier/l3-deadline-summary.json \
    --expect-prefix tests. --expect-prefix third_party. --cuda-devices 0,1 --always-reap -- \
    env PYTHONPATH="$FA_SHIM" "$PY" -I /tests/infb_verify/offline_exec.py --record /logs/verifier/l3-network.json -- \
    "$PY" -m pytest -p no:cacheprovider --continue-on-collection-errors --rootdir=/app -q \
    -k "$L3_K" $L3_FILES --junitxml=/logs/verifier/junit-l3.xml
run_group tp1 tp1 infb_verify.test_tp1. /tests /tests/infb_verify/test_tp1.py
run_group tp2 tp2 infb_verify.test_tp2. /tests /tests/infb_verify/test_tp2.py
run_group ar ar infb_verify.test_ar. /tests /tests/infb_verify/test_ar.py

# The reward disposition: a valid candidate failure anywhere leaves the grader's reward; otherwise
# a censored or blocked scored node voids the trial (the -1 sentinel), never a zero.
python3 -I /tests/infb_verify/disposition.py || { log "void: see /logs/verifier/disposition.json; no reward (sentinel -1)"; exit 3; }
# >>> END RUN TESTS <<<

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

mkdir -p /logs/verifier/reports 2>/dev/null
for _f in /logs/verifier/*; do
  case "${_f##*/}" in
    reward.json|reward.txt|ctrf.json|run.log|test-stdout.txt|reports) continue ;;
  esac
  [ -f "$_f" ] && mv -f "$_f" /logs/verifier/reports/ 2>/dev/null
done
