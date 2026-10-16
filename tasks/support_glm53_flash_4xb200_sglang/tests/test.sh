#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

python3 /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
export PATH="/opt/sglang/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SGLANG_RUST_BUILD_MODE=never CARGO_NET_OFFLINE=true SGLANG_BUILD_RUST_EXTS=none
PY=/opt/sglang/bin/python
BASE="$(python3 -c 'import json; print(json.load(open("/tests/config.json"))["base_commit"])')"
bound() { python3 -c 'import json,sys; print("%g" % json.load(open("/tests/config.json"))["infb"]["bounds"]["groups"][sys.argv[1]])' "$1"; }

run_log timeout --kill-after=30 "$(bound preflight)" "$PY" /tests/preflight.py \
    || { log "preflight failed or overran its bound; no reward written (sentinel -1)"; exit 3; }

run_log timeout --kill-after=30 "$(bound reference)" "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json \
    || { log "the reference could not be served or measured on this host, or overran its bound; no reward (sentinel -1)"; exit 3; }

git checkout -q "$BASE" -- python/sglang/test test eval 2>>"$RUN_LOG" || { log "evaluation-materials reset failed"; exit 4; }
git clean -fdq -- python/sglang/test test eval 2>>"$RUN_LOG" || true
git apply --whitespace=nowarn --allow-empty /tests/test.patch 2>>"$RUN_LOG" || { log "test.patch re-apply failed"; exit 4; }
git status --porcelain -- python/sglang/test test eval >> "$RUN_LOG" 2>&1 || true

run_log "$PY" -I /tests/infb_efficiency/deadline.py run --group L0 --budget-s "$(bound L0)" \
    --junit /logs/verifier/junit-l0.xml --results /logs/verifier/l0-results.jsonl \
    --config /tests/config.json --expect-prefix "infb_verify.test_l0." -- \
    "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q /tests/infb_verify/test_l0.py \
    --junitxml=/logs/verifier/junit-l0.xml

cd /app/test/registered || { log "no /app/test/registered"; exit 5; }
run_log "$PY" -I /tests/infb_efficiency/deadline.py run --group L3 --budget-s "$(bound L3)" \
    --junit /logs/verifier/junit-l3.xml --results /logs/verifier/l3-results.jsonl \
    --config /tests/config.json --expect-prefix "registered.unit." -- \
    "$PY" -m pytest -p no:cacheprovider --continue-on-collection-errors --rootdir=/app/test -q \
    unit/model_executor/test_pool_configurator.py unit/parser/test_jinja_template_utils.py \
    unit/spec/test_eagle_worker_v2_topk1_fastpath.py \
    --junitxml=/logs/verifier/junit-l3.xml
cd /app

cp /app/eval/glm53_flash/tests/data/gsm8k/test.jsonl /tmp/test.jsonl
cd /app/test || { log "no /app/test"; exit 5; }
for _cls in TestHiddenHighThroughput TestHiddenLowLatency; do
  _wb="$(bound "witness:$_cls")"
  log "witness launch $_cls (bound ${_wb}s)"
  mkdir -p "/logs/verifier/witness/$_cls"
  PYTHONPATH=/tests:/app/test run_log "$PY" -I /tests/infb_efficiency/deadline.py run \
      --group "witness-$_cls" --budget-s "$_wb" \
      --junit "/logs/verifier/witness/$_cls/runner.xml" --results "/logs/verifier/witness/$_cls/runner.jsonl" \
      --cuda-devices 0,1,2,3 --always-reap -- \
      "$PY" -I /tests/infb_verify/witness/driver.py --class "$_cls"
  _budget="$(bound "graded:$_cls")"
  log "L2 group $_cls (budget ${_budget}s)"
  mkdir -p "/logs/verifier/efficiency/$_cls"
  PYTHONPATH=/tests:/app/test run_log "$PY" -I /tests/infb_efficiency/deadline.py run \
      --group "$_cls" --budget-s "$_budget" \
      --junit "/logs/verifier/junit-l2-$_cls.xml" \
      --results "/logs/verifier/efficiency/$_cls/l2-results.jsonl" \
      --config /tests/config.json \
      --expect-prefix "infb_verify.test_hidden.$_cls." \
      --record "/logs/verifier/efficiency/$_cls/efficiency.json" \
      --cuda-devices 0,1,2,3 --always-reap -- \
      "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s \
      "/tests/infb_verify/test_hidden.py::$_cls" --junitxml="/logs/verifier/junit-l2-$_cls.xml"
done
cd /app
pkill -f "sglang serve" 2>/dev/null || true
run_log timeout --kill-after=30 "$(bound disposition)" "$PY" -I /tests/infb_verify/disposition.py --config /tests/config.json
_disp=$?
[ "$_disp" -eq 3 ] && { log "void: no valid candidate failure and a censored/blocked/faulted scored node (void.json); no reward (sentinel -1)"; exit 3; }
[ "$_disp" -ne 0 ] && { log "the disposition step failed (rc $_disp); no reward (sentinel -1)"; exit 3; }
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

python3 /tests/grader.py grade
log "reward.json=$(cat /logs/verifier/reward.json 2>/dev/null)"

mkdir -p /logs/verifier/reports 2>/dev/null
for _f in /logs/verifier/*; do
  case "${_f##*/}" in
    reward.json|reward.txt|ctrf.json|run.log|test-stdout.txt|reports) continue ;;
  esac
  [ -f "$_f" ] && mv -f "$_f" /logs/verifier/reports/ 2>/dev/null
done
