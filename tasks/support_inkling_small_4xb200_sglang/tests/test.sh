#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Verifier entrypoint (shared frame).
# Patching and grading live in tests/grader.py. This script owns the
# task-specific part: run the suites, write reports under /logs/verifier/,
# and apply any report fixups before grading.
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

PREP_T0=$(date -u +%s.%N)
timeout --kill-after=60 300 python3 /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
# support_inkling_small_4xb200_sglang: prepare, preflight, the reference phase (the E2
# arm's component timing and the reference's whole-serving measurement, before any candidate import), the restore of
# the evaluation paths, L0, the operator group, L3, the production launch and the deterministic launch, disposition,
# grade. Pier's exec carries no image PATH: name the environment explicitly.
export PATH="/opt/sglang/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CARGO_NET_OFFLINE=true
export TESTS_DIR=/tests APP_DIR=/app INFB_RECORD_DIR=/logs/verifier/records
export INKLING_SMALL_TEST_MODEL_PATH=/model/thinkingmachines/Inkling-Small-NVFP4
export INFB_E2_MEASURE=/logs/verifier/records/e2-measure.json INFB_BINDING_IDS=/logs/verifier/records/binding.json
export INFB_WITNESS_DIR=/logs/verifier/witness INFB_CANDIDATE_CACHE=/tmp/infb-candidate-cache
PY=/opt/sglang/bin/python
mkdir -p /logs/verifier/records

# Step walls: each step's wall is larger than the operations inside it, which keep their public
# limits (eval/inkling_small/limits.json).
PREFLIGHT_S=360
E2_S=1860
REFERENCE_S=4500
RESTORE_S=360
L0_S=1860
REL_S=5100
L3_S=1860
L2P_BUDGET_S=33960
L2D_BUDGET_S=10500
GRADE_S=360
lim() { python3 -c "import json,sys; print(int(json.load(open('/tests/eval/inkling_small/limits.json'))['operations'][sys.argv[1]]['limit_s']))" "$1"; }
ACTION_LIMIT_S=$(lim action)
stamp() { echo "[verifier] step $1 $2 $(date -u +%s.%N)" >> "$RUN_LOG" 2>/dev/null; }
echo "[verifier] step prepare start $PREP_T0" >> "$RUN_LOG" 2>/dev/null
stamp prepare end

# 0. Preflight: GPUs, checkpoint, base tree, no stale server. A failure exits without a reward (the -1 sentinel).
stamp preflight start
run_log timeout --kill-after=$((PREFLIGHT_S - ACTION_LIMIT_S)) "$ACTION_LIMIT_S" "$PY" /tests/preflight.py \
    || { log "preflight failed or exceeded its $ACTION_LIMIT_S s limit; no reward written (sentinel -1)"; exit 3; }
stamp preflight end

stamp reference-e2 start
SEED=$(python3 -c "import json; print(json.load(open('/tests/hidden/efficiency_seed.json'))['seed'])")
REF_TREE=/opt/infb-reference/tree
# The comparator per family: E2 (the landing's score-mod on plain FA4, over the reference tree) for decode_full,
# prefill_full and prefill_local; E1 (the reference's own served kernel, its adapter in the reference tree) for
# decode_local, where plain FA4 is faster on a 512-key window call.
comparator() {  # <adapter> <family> <out>
  ( cd /tmp && env -u PYTHONPATH PYTHONPATH="$REF_TREE/python:/tests/infb_verify" CUDA_VISIBLE_DEVICES=0 \
      TRITON_CACHE_DIR=/tmp/infb-e2-cache/triton CUTE_DSL_CACHE_DIR=/tmp/infb-e2-cache/cute_dsl \
      FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/tmp/infb-e2-cache/fa4 XDG_CACHE_HOME=/tmp/infb-e2-cache/xdg \
      SGLANG_CACHE_DIR=/tmp/infb-e2-cache/sglang TORCH_EXTENSIONS_DIR=/tmp/infb-e2-cache/torch_extensions \
      timeout --kill-after=30 "$(lim invocation)" "$PY" /tests/eval/inkling_small/rel_attention/measure.py \
        --adapter "$1" --family "$2" --seed "$SEED" --output "$3" ) >> "$RUN_LOG" 2>&1
}
for fam in decode_full prefill_full prefill_local; do
  comparator /tests/infb_verify/rel_attention_e2_arm.py "$fam" "/logs/verifier/records/comparator-$fam.json" \
    || { log "the E2 arm could not be timed on this host ($fam); no reward written (sentinel -1)"; exit 3; }
done
comparator "$REF_TREE/inkling_rel_attention.py" decode_local /logs/verifier/records/comparator-decode_local.json \
  || { log "the E1 comparator could not be timed on this host; no reward written (sentinel -1)"; exit 3; }
python3 - "$INFB_E2_MEASURE" <<'EOF' || { log "the comparator record is incomplete; no reward written (sentinel -1)"; exit 3; }
import json, sys
arms = {"decode_full": "E2", "prefill_full": "E2", "prefill_local": "E2", "decode_local": "E1"}
rows = []
for fam, arm in arms.items():
    r = json.load(open(f"/logs/verifier/records/comparator-{fam}.json"))["results"][0]
    rows.append({**r, "comparator": arm})
json.dump({"rule": "E1 for decode_local, E2 otherwise", "results": rows}, open(sys.argv[1], "w"), indent=1)
EOF
stamp reference-e2 end
stamp reference start
run_log timeout --kill-after=30 $((REFERENCE_S - 30)) "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json \
    || { log "the reference could not be served or measured on this host; no reward written (sentinel -1)"; exit 3; }
stamp reference end

# 2. The tests come from the verifier image, never from the agent's tree: restore_paths.py writes eval/inkling_small,
#    python/sglang/test and test from the prepared base's pinned trees, then test.patch is re-applied.
stamp restore start
timeout --kill-after=$((RESTORE_S - ACTION_LIMIT_S)) "$ACTION_LIMIT_S" "$PY" -I -S /tests/infb_verify/restore_paths.py /tests/config.json 2>&1 | tee -a "$RUN_LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { log "restoring the evaluation paths failed; no reward written (sentinel -1)"; exit 4; }
stamp restore end
git apply --whitespace=nowarn --allow-empty /tests/test.patch 2>>"$RUN_LOG" || { log "test.patch re-apply failed"; exit 4; }
cp /app/eval/inkling_small/fixtures/gsm8k_test.jsonl /tmp/test.jsonl

# 3. L0: install, registry, configuration.
stamp l0 start
run_log timeout --kill-after=60 $((L0_S - 60)) "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q /tests/infb_verify/test_l0.py \
    --junitxml=/logs/verifier/junit-l0.xml
stamp l0 end

stamp rel start
run_log timeout --kill-after=60 $((REL_S - 60)) "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q /tests/infb_verify/test_rel_attention.py \
    --junitxml=/logs/verifier/junit-rel.xml
stamp rel end

# 5. L3: pass-to-pass. The BASE versions of the files holding the p2p ids, again with the network blocked
#    (config.json infb.l3_files and p2p_node_ids; infb.l3_excluded names the one id excluded as environment_unavailable),
#    restored by step 2, run WITH the GPUs as they were differentiated, before any server starts.
#    Each file is one unit-test-file operation (limits.json unit_test_file, 1,800 s) in its own pytest process inside
#    the step's wall L3_S = max(3 R, the file limit + 60 s), R the reference's measured step
#    with the network blocked. A file stopped at its own limit is a candidate resource failure; a file stopped
#    because the wall ran out within its limit is censored and the files after it are blocked (l3-wall.json, read by
#    disposition.py). The reports are merged into junit-l3.xml; the grader counts a missing id as failed. The step
#    imposes the task's allow_internet = false itself: the proxy variables removed, and
#    infb_verify/offline/sitecustomize.py first on PYTHONPATH, which refuses every non-loopback connection.
FILE_LIMIT_S=$(lim unit_test_file)
cd /app/test || { log "no /app/test"; exit 5; }
mapfile -t L3_FILES < <(python3 -c "import json; print('\n'.join(json.load(open('/tests/config.json'))['infb']['l3_files']))")
stamp l3 start
L3_T0=$(date +%s); L3_CENSORED=""; L3_NOT_RUN=""
for i in "${!L3_FILES[@]}"; do
  f=${L3_FILES[$i]}
  left=$(( L3_S - 60 - ($(date +%s) - L3_T0) ))
  if [ "$left" -le 0 ]; then L3_NOT_RUN="$L3_NOT_RUN $f"; continue; fi
  cap=$FILE_LIMIT_S; walled=0
  if [ "$left" -lt "$cap" ]; then cap=$left; walled=1; fi
  run_log env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY \
      PYTHONPATH=/tests/infb_verify/offline:/app/test timeout --kill-after=10 "$cap" "$PY" -m pytest -p no:cacheprovider \
      --continue-on-collection-errors --rootdir=/app/test -q "$f" --junitxml=/logs/verifier/junit-l3-$i.xml
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    if [ "$walled" = 1 ]; then L3_CENSORED="$f"; log "L3 $f stopped by the step's wall within its limit: censored"
    else log "L3 $f exceeded its $FILE_LIMIT_S s limit (rc $rc): a candidate resource failure"; fi
  fi
done
stamp l3 end
python3 /tests/infb_verify/l3_merge.py /logs/verifier /tests/config.json "$L3_CENSORED" $L3_NOT_RUN 2>&1 | tee -a "$RUN_LOG"
cd /app

# 6-7. L2: the production launch, then the deterministic launch, each one group under the deadline runner.
cd /app/test || { log "no /app/test"; exit 5; }
stamp l2p start
PYTHONPATH=/tests:/app/test run_log "$PY" -I /tests/infb_efficiency/deadline.py run --group TestHiddenInklingProduction \
    --budget-s "$L2P_BUDGET_S" --junit /logs/verifier/junit-l2p.xml \
    --results /logs/verifier/efficiency/TestHiddenInklingProduction/l2-results.jsonl --config /tests/config.json \
    --expect-prefix infb_verify.test_hidden.TestHiddenInklingProduction. \
    --record /logs/verifier/efficiency/TestHiddenInklingProduction/efficiency.json --cuda-devices 0,1,2,3 --always-reap -- \
    "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s \
    "/tests/infb_verify/test_hidden.py::TestHiddenInklingProduction" --junitxml=/logs/verifier/junit-l2p.xml
stamp l2p end
stamp l2d start
PYTHONPATH=/tests:/app/test run_log "$PY" -I /tests/infb_efficiency/deadline.py run --group TestHiddenInklingDeterministic \
    --budget-s "$L2D_BUDGET_S" --junit /logs/verifier/junit-l2d.xml \
    --results /logs/verifier/efficiency/TestHiddenInklingDeterministic/l2-results.jsonl --config /tests/config.json \
    --expect-prefix infb_verify.test_hidden.TestHiddenInklingDeterministic. --cuda-devices 0,1,2,3 --always-reap -- \
    "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s \
    "/tests/infb_verify/test_hidden.py::TestHiddenInklingDeterministic" --junitxml=/logs/verifier/junit-l2d.xml
stamp l2d end
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
