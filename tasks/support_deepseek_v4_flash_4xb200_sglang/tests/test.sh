#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

PREP_T0=$(date -u +%s.%N)
timeout --kill-after=60 300 python3 /tests/grader.py prepare || exit $?  # PREPARE_S = 300 + 60 (generated below)
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
export PATH="/opt/sglang/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CARGO_NET_OFFLINE=true
PY=/opt/sglang/bin/python
BASE="$(python3 -c 'import json; print(json.load(open("/tests/config.json"))["base_commit"])')"
CAND_CACHE=/tmp/infb-candidate-cache
mkdir -p $CAND_CACHE/triton $CAND_CACHE/torch-extensions $CAND_CACHE/tilelang $CAND_CACHE/tilelang-tmp $CAND_CACHE/tvm-ffi
export TRITON_CACHE_DIR=$CAND_CACHE/triton TORCH_EXTENSIONS_DIR=$CAND_CACHE/torch-extensions \
       TILELANG_CACHE_DIR=$CAND_CACHE/tilelang TILELANG_TMP_DIR=$CAND_CACHE/tilelang-tmp TVM_FFI_CACHE_DIR=$CAND_CACHE/tvm-ffi

# every step's wall is generated from the verifier's DAG
# (config.json infb.deadline_dag, tests/infb_verify/deadline_dag.py, which also checks these literals), and every
# operation inside a step keeps its public limit (eval/deepseek_v4_flash/limits.json). Each step's start and end are
# stamped into run.log.
PREPARE_S=360
PREFLIGHT_S=360
REFERENCE_KERNELS_S=1860
REFERENCE_S=1860
RESTORE_S=360
L0_BUILD_S=3660
L0_REGISTRY_S=1860
KERNELS_S=11220
KERNEL_NODE_S=(1860 1860 1860 1860 1860 1860)
L3_S=1860
L2_BUDGET_S=33040
GRADE_S=360
lim() { python3 -c "import json,sys; print(int(json.load(open('/tests/eval/deepseek_v4_flash/limits.json'))['operations'][sys.argv[1]]['limit_s']))" "$1"; }
ACTION_LIMIT_S=$(lim action); FILE_LIMIT_S=$(lim unit_test_file); KERNEL_BUILD_LIMIT_S=$(lim kernel_build)
stamp() { echo "[verifier] step $1 $2 $(date -u +%s.%N)" >> "$RUN_LOG" 2>/dev/null; }
echo "[verifier] step prepare start $PREP_T0" >> "$RUN_LOG" 2>/dev/null
stamp prepare end

# 0. Preflight: the environment is what the task declares (GPUs, checkpoint, base' tree, kernels,
#    no stale server). On failure exit WITHOUT a reward: the frame's trap then writes the -1
#    sentinel, so an environment fault is never scored as a candidate defect.
stamp preflight start
run_log timeout --kill-after=$((PREFLIGHT_S - ACTION_LIMIT_S)) "$ACTION_LIMIT_S" "$PY" /tests/preflight.py \
    || { log "preflight failed or exceeded its $ACTION_LIMIT_S s limit; no reward written (sentinel -1)"; exit 3; }
stamp preflight end

stamp reference_kernels start
run_log timeout --kill-after=30 $((REFERENCE_KERNELS_S - 30)) "$PY" -I /tests/infb_verify/kernel_phase.py reference \
    || { log "the reference did not pass the component cases or could not be measured on this host; no reward written (sentinel -1)"; exit 3; }
stamp reference_kernels end
pkill -9 -f "/tests/infb_verify/kernel_worker.py" 2>/dev/null || true   # no worker outlives its phase

stamp reference start
run_log timeout --kill-after=30 $((REFERENCE_S - 30)) "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json \
    || { log "the reference could not be served or measured on this host; no reward written (sentinel -1)"; exit 3; }
stamp reference end

stamp restore start
timeout --kill-after=$((RESTORE_S - ACTION_LIMIT_S)) "$ACTION_LIMIT_S" "$PY" -I -S /tests/infb_verify/restore_paths.py /tests/config.json 2>&1 | tee -a "$RUN_LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { log "restoring the evaluation paths failed; no reward written (sentinel -1)"; exit 4; }
stamp restore end
git apply --whitespace=nowarn --allow-empty /tests/test.patch 2>>"$RUN_LOG" || { log "test.patch re-apply failed"; exit 4; }
git status --porcelain -- python/sglang/test test eval >> "$RUN_LOG" 2>&1 || true

stamp l0_build start
run_log timeout --kill-after=$((L0_BUILD_S - KERNEL_BUILD_LIMIT_S)) "$KERNEL_BUILD_LIMIT_S" "$PY" -m pytest -p no:cacheprovider \
    --rootdir=/tests -q "/tests/infb_verify/test_l0.py::test_build_editable_install" --junitxml=/logs/verifier/junit-l0-build.xml
rc=$?; [ "$rc" = 124 ] || [ "$rc" = 137 ] && log "the L0 build exceeded its $KERNEL_BUILD_LIMIT_S s limit (rc $rc): a candidate resource failure; its id counts as failed"
stamp l0_build end
stamp l0_registry start
run_log timeout --kill-after=$((L0_REGISTRY_S - FILE_LIMIT_S)) "$FILE_LIMIT_S" "$PY" -m pytest -p no:cacheprovider \
    --rootdir=/tests -q "/tests/infb_verify/test_l0.py::test_registry_resolves_native_class" --junitxml=/logs/verifier/junit-l0-registry.xml
rc=$?; [ "$rc" = 124 ] || [ "$rc" = 137 ] && log "L0's registry node exceeded its $FILE_LIMIT_S s limit (rc $rc): a candidate resource failure; its id counts as failed"
stamp l0_registry end
python3 /tests/infb_verify/junit_merge.py /logs/verifier l0 2>&1 | tee -a "$RUN_LOG"

KERNEL_NODES=(test_sparse_bf16_correctness test_sparse_paged_correctness test_compression_state_correctness test_cache_transform_correctness test_kernel_prefill_efficiency test_kernel_decode_efficiency)
stamp kernels start
for i in "${!KERNEL_NODES[@]}"; do
  n=${KERNEL_NODES[$i]}
  run_log timeout --kill-after=10 "${KERNEL_NODE_S[$i]}" "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q \
      "/tests/infb_verify/test_kernels.py::$n" --junitxml=/logs/verifier/junit-kernels-$i.xml
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    log "kernel node $n: its wall (${KERNEL_NODE_S[$i]} s) expired with its operations within their limits: censored"
    "$PY" /tests/infb_verify/kernel_phase.py fault "infb_verify.test_kernels.$n" \
        "censored: the node's wall expired with its operations within their limits" 2>&1 | tee -a "$RUN_LOG"
  fi
done
stamp kernels end
pkill -9 -f "/tests/infb_verify/kernel_worker.py" 2>/dev/null || true   # no worker outlives its node
python3 /tests/infb_verify/junit_merge.py /logs/verifier kernels 2>&1 | tee -a "$RUN_LOG"

cd /app/test || { log "no /app/test"; exit 5; }
L3_FILES=(registered/unit/mem_cache/test_swa_unittest.py registered/unit/mem_cache/test_swa_eviction_boundary.py
          registered/unit/managers/test_prefill_adder.py registered/unit/managers/test_hisparse_unit.py
          registered/unit/utils/test_weight_checker.py registered/unit/entrypoints/openai/test_serving_chat.py)
stamp l3 start
L3_T0=$(date +%s); L3_CENSORED=""; L3_NOT_RUN=""
for i in "${!L3_FILES[@]}"; do
  f=${L3_FILES[$i]}
  left=$(( L3_S - 60 - ($(date +%s) - L3_T0) ))
  if [ "$left" -le 0 ]; then L3_NOT_RUN="$L3_NOT_RUN $f"; continue; fi
  cap=$FILE_LIMIT_S; walled=0
  if [ "$left" -lt "$cap" ]; then cap=$left; walled=1; fi
  PYTHONPATH=/app/test run_log timeout --kill-after=10 "$cap" "$PY" -m pytest -p no:cacheprovider --continue-on-collection-errors \
      --rootdir=/app/test -q "$f" --junitxml=/logs/verifier/junit-l3-$i.xml
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    if [ "$walled" = 1 ]; then L3_CENSORED="$f"; log "L3 $f stopped by the step's wall within its limit: censored"
    else log "L3 $f exceeded its $FILE_LIMIT_S s limit (rc $rc): a candidate resource failure"; fi
  fi
done
stamp l3 end
python3 /tests/infb_verify/l3_merge.py /logs/verifier /tests/config.json "$L3_CENSORED" $L3_NOT_RUN 2>&1 | tee -a "$RUN_LOG"
cd /app

cp /app/eval/deepseek_v4_flash/tests/fixtures/gsm8k_test.jsonl /tmp/test.jsonl
cd /app/test || { log "no /app/test"; exit 5; }
stamp l2 start
PYTHONPATH=/tests:/app/test run_log "$PY" -I /tests/infb_efficiency/deadline.py run --group TestHiddenDSV4 \
    --budget-s "$L2_BUDGET_S" --junit /logs/verifier/junit-l2.xml \
    --results /logs/verifier/efficiency/TestHiddenDSV4/l2-results.jsonl --config /tests/config.json \
    --expect-prefix infb_verify.test_hidden.TestHiddenDSV4. \
    --record /logs/verifier/efficiency/TestHiddenDSV4/efficiency.json --cuda-devices 0,1,2,3 --always-reap -- \
    "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s \
    /tests/infb_verify/test_hidden.py \
    --junitxml=/logs/verifier/junit-l2.xml
stamp l2 end
cd /app
pkill -f "sglang.launch_server" 2>/dev/null || true
pkill -f "sglang serve" 2>/dev/null || true
python3 /tests/infb_verify/disposition.py /tests/config.json 2>&1 | tee -a "$RUN_LOG"
if [ "${PIPESTATUS[0]}" = 10 ]; then log "void: a scored result was censored (void.json); no reward written (sentinel -1)"; VOID=1; fi
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

[ "${VOID:-0}" = 1 ] && { log "void.json: $(cat /logs/verifier/void.json 2>/dev/null)"; exit 7; }
stamp grade start
timeout --kill-after=$((GRADE_S - ACTION_LIMIT_S)) "$ACTION_LIMIT_S" python3 /tests/grader.py grade
stamp grade end
log "reward.json=$(cat /logs/verifier/reward.json 2>/dev/null)"

mkdir -p /logs/verifier/reports 2>/dev/null
for _f in /logs/verifier/*; do
  case "${_f##*/}" in
    reward.json|reward.txt|ctrf.json|run.log|test-stdout.txt|reports) continue ;;
  esac
  [ -f "$_f" ] && mv -f "$_f" /logs/verifier/reports/ 2>/dev/null
done
