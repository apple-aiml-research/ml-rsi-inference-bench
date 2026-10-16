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

python3 /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

# Canonical raw-output log. The task middle SHOULD send every suite's combined
# stdout+stderr here so the reason a test failed is never lost -- use run_log,
# or pipe through `tee -a "$RUN_LOG"` when feeding a reporter. Never 2>/dev/null
# a test run.
export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }

# >>> RUN TESTS (task-specific) <<<
# The shared frame's final grade command must not execute candidate site.py,
# sitecustomize.py or editable-package startup hooks after submitted code runs.
# Keep the frame unchanged; all its later python3 invocations use only stdlib.
python3() { /usr/bin/python3 -I -S "$@"; }
# Restore complete protected trees, including added files and symlinks, before
# any candidate Python can run. The system interpreter skips site startup.
/usr/bin/python3 -I -S /tests/protect_source.py || {
  log "submitted capture or protected-source reconstruction failed"
  /usr/bin/python3 -I -S /tests/grader.py grade --apply-failed
  exit 0
}
# L0 structural, L3 pass-to-pass by differential, L2 the registered acceptance
# test on a hidden permutation seed against a measured floor, the fidelity node
# against the model author's own implementation, a second fidelity statistic,
# and the served nodes (the recipe, the reasoning effort the chat endpoint
# renders, the final channel as the chat content, and the decode path against
# the server's own prefill). Every step outside the node deadlines is its own
# process group, so no step can run the verifier into the harness's cap.
# The harness's exec carries no image PATH: name the environment explicitly.
export PATH="/opt/sglang/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_OFFLINE=1
export SGLANG_RUST_BUILD_MODE=never CARGO_NET_OFFLINE=true
# The registered test's own retry must NOT fire: CustomTestCase._callTestMethod retries
# SGLANG_TEST_MAX_RETRY times, defaulting to 1 when SGLANG_IS_IN_CI is set. A bar is never retried
# to green, so the default is not relied on -- it is set.
unset SGLANG_IS_IN_CI SGLANG_AMD_CI
export SGLANG_TEST_MAX_RETRY=0
PY=/opt/sglang/bin/python
cfg() { /usr/bin/python3 -I -S -c 'import json,sys; v=json.load(open("/tests/config.json"))
for k in sys.argv[1].split("."): v=v[k]
print(v)' "$1"; }
BASE="$(cfg base_commit)"
# A unit-test file process is one operation with its own 1,800 s cap, so its expiry is the
# candidate's resource failure (its junit is absent and its ids grade as failed). A composite step
# (a class sharing one server, the component phase, the efficiency group) is bounded by the
# composition of its nodes' bounds from the reference's measured walls (infb.deadline_dag): its
# expiry while every operation stayed within its own cap is a verifier fault, so the trial is
# censored -- no reward, the -1 sentinel -- never scored.
UNIT_FILE_CAP="$(cfg infb.components.operation_limits_sec.component_invocation)"
PREFLIGHT_BOUND="$(cfg infb.deadlines_sec.preflight)"
REFERENCE_BOUND="$(cfg infb.deadlines_sec.reference)"
FIDELITY_BOUND="$(cfg infb.deadlines_sec.fidelity)"
SERVED_BOUND="$(cfg infb.deadlines_sec.served)"
COMPONENT_BOUND="$(cfg infb.deadlines_sec.components)"
EFFICIENCY_BOUND="$(cfg infb.deadlines_sec.efficiency)"
STEP_WALLS=/logs/verifier/step-walls.jsonl

# Every step's wall is recorded, so reference runs supply the measured walls the composition uses.
step_wall() {  # name bound rc start end
  printf '{"step": "%s", "bound_s": %s, "rc": %s, "start": %s, "end": %s, "elapsed_s": %s}\n' \
    "$1" "$2" "$3" "$4" "$5" "$(( $5 - $4 ))" >> "$STEP_WALLS" 2>/dev/null || true
}

# A generation-driving step runs in its OWN PROCESS GROUP under a wall-clock bound, and the group
# is killed on expiry. An in-process alarm cannot end this one: the evaluator's sampler retries every
# non-400 error forever with exponential backoff over an httpx client whose timeout is 3,600 s, in a
# 198-thread pool, inside a ThreadPoolExecutor whose exit joins its worker.
deadline_run() {
  local name=$1 secs=$2; shift 2
  echo "+ [deadline ${secs}s] $*" >> "$RUN_LOG" 2>/dev/null
  local t0; t0=$(date +%s)
  setsid timeout --kill-after=30 --signal=TERM "$secs" "$@" 2>&1 | tee -a "$RUN_LOG"
  local rc=${PIPESTATUS[0]}
  step_wall "$name" "$secs" "$rc" "$t0" "$(date +%s)"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    log "step $name exceeded its ${secs}s bound and was killed"
  fi
  # Whatever the step left behind is stopped before the next one.
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "sglang::" 2>/dev/null || true
  pkill -f "infb_components.probes.observe_launch" 2>/dev/null || true
  return "$rc"
}

# A class that shares one server runs under the group runner, bounded as a whole by its composed bound:
# per-node results are recorded as they arrive, and a node the bound cuts off is censored in its junit and
# in the group's summary. The disposition step before grading decides what a censored node means.
group_run() {  # name bound junit expect-prefix pytest-target
  local name=$1 secs=$2 junit=$3 prefix=$4 target=$5
  local t0; t0=$(date +%s)
  mkdir -p "/logs/verifier/$name"
  PYTHONPATH=/tests:/app/eval:/app/test run_log "$PY" -I /tests/infb_efficiency/deadline.py run \
      --group "$name" --budget-s "$secs" --junit "$junit" \
      --results "/logs/verifier/$name/results.jsonl" --config /tests/config.json \
      --expect-prefix "$prefix" --no-node-alarms --summary "/logs/verifier/$name-deadline-summary.json" \
      --cuda-devices 0,1,2,3 --always-reap -- \
      "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s "$target" \
      --junitxml="$junit"
  local rc=$?
  step_wall "$name" "$secs" "$rc" "$t0" "$(date +%s)"
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "infb_components.probes.observe_launch" 2>/dev/null || true
  return "$rc"
}

# A step that runs no candidate code (preflight, the reference), under its own process-group bound. On
# expiry, as on any failure of these steps, the caller exits 3 without a reward: the host could not do its
# part, so the frame writes the -1 sentinel, never a score.
host_bounded() {
  local name=$1 secs=$2; shift 2
  echo "+ [bound ${secs}s] $*" >> "$RUN_LOG" 2>/dev/null
  local t0; t0=$(date +%s)
  setsid timeout --kill-after=30 --signal=TERM "$secs" "$@" 2>&1 | tee -a "$RUN_LOG"
  local rc=${PIPESTATUS[0]}
  step_wall "$name" "$secs" "$rc" "$t0" "$(date +%s)"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    log "step $name exceeded its ${secs}s bound and was killed (a host fault)"
  fi
  return "$rc"
}

# 0. Preflight: the environment is what the task declares (GPUs, checkpoint, base tree, kernels,
#    no stale server). On failure exit WITHOUT a reward: the frame's trap then writes the -1
#    sentinel, so an environment fault is never scored as a candidate defect.
host_bounded preflight "$PREFLIGHT_BOUND" "$PY" -I -S /tests/trusted_python.py /tests/preflight.py || { log "preflight failed; no reward written (sentinel -1)"; exit 3; }

# 0.5 The REFERENCE, served and measured before any candidate code runs. The tree it serves was built into
#     this image by the snapshot step, so the candidate's patch cannot have reached it. Any failure here is
#     the HOST's: exit 3 without a reward. `-I` and not `-m`: test.sh works from /app, which carries the
#     candidate's patch, and `python -m` would put that directory ahead of the standard library.
host_bounded reference "$REFERENCE_BOUND" "$PY" -I -S /tests/trusted_python.py /tests/infb_efficiency/reference.py --config /tests/config.json \
    || { log "the reference could not be served or measured on this host; no reward (sentinel -1)"; exit 3; }

# 2. L0: the candidate tree installs (editable, offline) and the engine's own registry resolves the
#    checkpoint's architecture to a native model class. One test-file process, one operation.
deadline_run l0 "$UNIT_FILE_CAP" "$PY" -m pytest -p no:cacheprovider --rootdir=/tests -q /tests/infb_verify/test_l0.py \
    --junitxml=/logs/verifier/junit-l0.xml

# 3. L3: pass-to-pass regression on the two subsystems the solution rewrites, at the base versions
#    (step 1 restored them), so these ids are upstream's tests and not the candidate's. Chosen by
#    differential: ids passing at the base tree and at the closure tree. Each test file is its own process
#    and its own 1,800 s operation, rooted at /app so the junit ids carry the `test.srt.` and
#    `python.sglang.test.` prefixes config.json whitelists. BEFORE L2, so every GPU allocation these kernel
#    tests make is freed before L2's server takes --mem-fraction-static 0.93.
deadline_run p2p_fused_moe "$UNIT_FILE_CAP" "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q \
    test/srt/test_fused_moe.py --junitxml=/logs/verifier/junit-l3-fused-moe.xml
deadline_run p2p_triton_attention "$UNIT_FILE_CAP" "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q \
    test/srt/test_triton_attention_kernels.py --junitxml=/logs/verifier/junit-l3-triton-attention.xml
deadline_run p2p_activation "$UNIT_FILE_CAP" "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q \
    python/sglang/test/test_activation.py --junitxml=/logs/verifier/junit-l3-activation.xml

# 4. L2: the registered acceptance test, verbatim apart from the launch limit test.patch normalizes to
#    900 s, at the recipe. Its own launch serves the candidate; the evaluator reads the GPQA question set
#    from /tests/_upstream/gpqa_diamond.csv, so the step needs no network. One test-file process, one
#    operation. Rooted at /app/test so the junit node id is the one config.json whitelists.
cd /app/test || { log "no /app/test"; exit 5; }
deadline_run l2 "$UNIT_FILE_CAP" "$PY" -m pytest -p no:cacheprovider --rootdir=/app/test -q -s \
    srt/test_gpt_oss_4gpu.py -k test_mxfp4_120b \
    --junitxml=/logs/verifier/junit-l2.xml
cd /app
# 5. Fidelity: the served model's own arithmetic against the model author's implementation, two statistics,
#    per band, with their bounded re-measurements. One launch, both nodes.
group_run fidelity "$FIDELITY_BOUND" /logs/verifier/junit-fidelity.xml \
    infb_verify.test_fidelity.TestFidelity. /tests/infb_verify/test_fidelity.py
# 5.5 The served nodes: the recipe, the reasoning effort the chat endpoint renders, the final channel as the
#     chat content, tool calls, decode against prefill, and the graph-coverage workloads, on one server
#     launched through the private production observer.
group_run served "$SERVED_BOUND" /logs/verifier/junit-served.xml \
    infb_verify.test_served.TestServed. /tests/infb_verify/test_served.py
# Complete native experts, packed residency/workspace, ordinary attention, complete Harmony and native
# production binding. The parent uses only private checker code; engine children name explicit
# source/kernel roots. A reference, held-out-data or host failure exits 3.
deadline_run components "$COMPONENT_BOUND" "$PY" -I -S /tests/trusted_python.py \
    /tests/infb_components/runner.py --config /tests/config.json \
    --candidate /app --reference /opt/infb-reference/tree --output /logs/verifier/components
component_rc=$?
if [ "$component_rc" -eq 3 ]; then
  log "component reference, held-out data or host failed; no candidate reward"
  exit 3
fi
if [ "$component_rc" -eq 124 ] || [ "$component_rc" -eq 137 ]; then
  log "the component phase hit its composed bound; its unreported nodes are censored (the disposition decides)"
fi
# 6. The two scored efficiency nodes, as their own group: one class, its own launch, its own process group
#    under the composed bound. Each client invocation has the 1,800 s measurement allowance; a node bound
#    or the group bound expiring is censored (the runner exits 124 or 125), never scored.
_EFF_CLS=TestHiddenAcceptance
mkdir -p "/logs/verifier/efficiency/$_EFF_CLS"
_eff_t0=$(date +%s)
PYTHONPATH=/tests:/app/eval:/app/test run_log "$PY" -I /tests/infb_efficiency/deadline.py run \
    --group "$_EFF_CLS" --budget-s "$EFFICIENCY_BOUND" \
    --junit /logs/verifier/junit-efficiency.xml \
    --results "/logs/verifier/efficiency/$_EFF_CLS/results.jsonl" \
    --config /tests/config.json \
    --expect-prefix "infb_verify.test_hidden.$_EFF_CLS." \
    --record "/logs/verifier/efficiency/$_EFF_CLS/efficiency.json" \
    --summary /logs/verifier/efficiency-deadline-summary.json \
    --cuda-devices 0,1,2,3 --always-reap -- \
    "$PY" -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s \
    "/tests/infb_verify/test_hidden.py::$_EFF_CLS" --junitxml=/logs/verifier/junit-efficiency.xml
efficiency_rc=$?
step_wall efficiency "$EFFICIENCY_BOUND" "$efficiency_rc" "$_eff_t0" "$(date +%s)"
# Whatever the class left behind is stopped before grading.
pkill -f "sglang.launch_server" 2>/dev/null || true
# The reward disposition: a valid candidate failure anywhere leaves the grader's
# reward; otherwise a censored or blocked scored node voids the trial (the -1 sentinel), never a zero.
/usr/bin/python3 -I -S /tests/disposition.py || { log "void: see /logs/verifier/disposition.json; no reward (sentinel -1)"; exit 3; }
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

python3 /tests/grader.py grade
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
