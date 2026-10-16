#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Verifier entrypoint (shared frame; the task-specific part is the RUN TESTS block).
# Patching and grading live in tests/grader.py. This script owns the task-specific part:
# run the suites, write reports under /logs/verifier/, and apply any report fixups before grading.
set -uo pipefail
trap 'if [ ! -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward.txt ]; then mkdir -p /logs/verifier; echo -1 > /logs/verifier/reward.txt; fi' EXIT
log() { echo "[verifier] $*"; }
cd /app || { mkdir -p /logs/verifier; exit 6; }

python3 /tests/grader.py prepare || exit $?
[ -f /logs/verifier/reward.json ] && exit 0   # model.patch didn't apply -> graded 0

export RUN_LOG=/logs/verifier/run.log
: > "$RUN_LOG" 2>/dev/null || true
run_log() { echo "+ $*" >> "$RUN_LOG" 2>/dev/null; local from rc; from=$(( $(stat -c %s "$RUN_LOG" 2>/dev/null || echo 0) + 1 )); "$@" >> "$RUN_LOG" 2>&1; rc=$?; tail -c "+$from" "$RUN_LOG" 2>/dev/null; return "$rc"; }

# >>> RUN TESTS (task-specific) <<<
# Pier's exec carries no image PATH: name the environment explicitly.
export PATH="/opt/vllm/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
PY=/opt/vllm/bin/python
BASE="$(python3 -c 'import json; print(json.load(open("/tests/config.json"))["base_commit"])')"
# Every step that runs the candidate's code is bounded as a group, each bound from
# config.json (infb.efficiency.deadlines.group_bounds_s, with its measured basis). A group that overruns is killed
# and loses its junit, so its every id grades as absent.
# The bound ends the step's WHOLE process tree. `timeout --kill-after=10` signalled
# only its own process group, and upstream's fork-per-test decorator moves pytest out of that group
# (os.setpgrp()) before it forks the test body, so a hanging body could outlive the bound.
# tests/bounded_step.py keeps every process of the step as its descendant (it is a child
# subreaper). At the bound it sends TERM to all of them, KILL 10 s later, and waits up to 20 s for the tree to
# be gone. It exits 124, or 137 if KILL was needed, as timeout did. A step that ends by itself has whatever it
# left running ended the same way, so nothing of one step reaches the next. A group's effective bound is its
# bound plus those 10 s and 20 s. Each step logs one INFB_BOUNDED line.
bound() { python3 -c 'import json, sys; print(json.load(open("/tests/config.json"))["infb"]["efficiency"]["deadlines"]["group_bounds_s"][sys.argv[1]])' "$1"; }
bounded() { local group=$1; shift; python3 -I /tests/bounded_step.py --group "$group" --bound-s "$(bound "$group")" --grace-s 10 --reap-s 20 -- "$@"; }
SESSION_BUDGET="$(python3 -c 'import json; print(json.load(open("/tests/config.json"))["infb"]["efficiency"]["deadlines"]["session_budget_s"])')"

# R. The protected trees come from the verifier image, never from the agent's tree, and are restored BEFORE
#    anything imports, launches or checks them: the evaluation tree eval/ (the supplied
#    protocols under eval/kimi_linear/, of which the grader runs its byte-identical copy under /tests/eval)
#    and upstream's test package tests/. grader.py prepare applied the candidate's model.patch (which may
#    touch anything) and then test.patch; each tree is restored to the base blobs, with anything the
#    candidate added there dropped -- files, symlinks, ignored files and a replaced directory alike -- and
#    test.patch re-applied. The candidate's /app/kda-verification.json is outside both trees and stays.
# `git checkout <base> -- <tree>` alone is not a reset: it rewrites the paths that EXIST at the base and
# leaves behind every file the candidate's patch ADDED there. Removing the tree first makes the
# restore exact, and it depends on nothing about how the grader applied the patch.
#
# WHAT tests/ CARRIES, and why it is load-bearing here: the landing's three-line tests/models/registry.py
# entry, which the base itself now carries. The scored
# upstream node is test_can_initialize_large_subset[KimiLinearForCausalLM], and that parametrization EXISTS
# ONLY IF the registry names the architecture (the arch list is built from HF_EXAMPLE_MODELS). So without this
# reset the candidate's own registry edit would decide whether its own scored node exists.
for tree in eval tests; do
  rm -rf -- "$tree" 2>>"$RUN_LOG"
  git checkout -q "$BASE" -- "$tree" 2>>"$RUN_LOG" || { log "$tree reset failed"; exit 4; }
  git clean -fdqx -- "$tree" 2>>"$RUN_LOG" || true
done
git apply --whitespace=nowarn --allow-empty /tests/test.patch 2>>"$RUN_LOG" || { log "test.patch re-apply failed"; exit 4; }
git status --porcelain --ignored -- eval tests >> "$RUN_LOG" 2>&1 || true
if [ -n "$(git status --porcelain --ignored -- eval)" ] || ! git diff --quiet "$BASE" -- eval \
   || ! python3 /tests/eval_identity.py /app/eval/kimi_linear /tests/eval/kimi_linear >> "$RUN_LOG" 2>&1; then
  log "the restored evaluation tree is not the base's, or differs from the grader's copy"; exit 4
fi

# 0. Preflight: the environment is what the task declares (GPUs, checkpoint, base tree, the supplied
#    kernels, the datasets, no stale server), after the restore above. On failure exit WITHOUT a reward: the frame's trap then
#    writes the -1 sentinel, so an environment fault is never scored as a candidate defect.
run_log "$PY" /tests/preflight.py || { log "preflight failed; no reward written (sentinel -1)"; exit 3; }

# 0b. The efficiency REFERENCE is served and measured before any candidate code runs (this
#     preflight imports only installed packages, so the step goes right after it).
#     `-I` keeps /app, which already holds the candidate's patch, off the reference step's path. A
#     failure is the host's, never the candidate's: exit without a reward, the -1 sentinel.
run_log bounded reference "$PY" -I /tests/infb_efficiency/reference.py --config /tests/config.json \
    || { log "the reference could not be served or measured on this host; no reward written (sentinel -1)"; exit 3; }

run_log bounded l0 "$PY" -P -m pytest -p no:cacheprovider --rootdir=/tests -q -s /tests/infb_verify/test_l0.py \
    --junitxml=/logs/verifier/junit-l0.xml

# 3. The maintainers' own mandatory test, applied by test.patch and run from /app so its node id is
#    the upstream one (pytest's junit classname is the node id relative to the rootdir, so a wrong
#    rootdir makes the grader read the id as ABSENT rather than failed). The node is named EXACTLY
#    rather than selected with -k: an exact id that does not exist grades as absent, which is the
#    right verdict for a tree whose registry does not know the model.
#    It constructs the model with dummy weights on one GPU under hf_overrides (num_hidden_layers=1),
#    which at this checkpoint's 1-indexed kda_layers makes layer 0 a KDA layer with a dense MLP -- so
#    it exercises KDA-layer construction and the recurrent-state spec, not MLA.
cd /app || { log "no /app"; exit 5; }
run_log bounded l2upstream "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q \
    "tests/models/test_initialization.py::test_can_initialize_large_subset[KimiLinearForCausalLM]" \
    --junitxml=/logs/verifier/junit-l2upstream.xml

# 3b. L3, pass-to-pass: upstream's own tests beside the files the landing touches,
#     run from /app so their node ids are upstream's. The scored ids are config.json's p2p_node_ids. The same run carries the
#     maintainers' registry-import node for this model, which is fail-to-pass. Whole files are run
#     so that a missing id is visible in the report rather than silently unselected;
#     tests/compile/test_config.py is not run: it fails to collect on every tree.
# Each file is one existing-engine test-file process, bounded by the
#     published 1,800 s file limit (the group bound), so a file that exhausts it is the candidate's resource
#     failure; each writes its own report, so per-test attribution is kept.
run_log bounded l3-registry "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q tests/models/test_registry.py \
    --junitxml=/logs/verifier/junit-l3-registry.xml
run_log bounded l3-kv-cache "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q tests/v1/core/test_kv_cache_utils.py \
    --junitxml=/logs/verifier/junit-l3-kv-cache.xml
run_log bounded l3-config "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q tests/test_config.py \
    --junitxml=/logs/verifier/junit-l3-config.xml
run_log bounded l3-model-runner "$PY" -m pytest -p no:cacheprovider --rootdir=/app -q tests/v1/worker/test_gpu_model_runner.py \
    --junitxml=/logs/verifier/junit-l3-model-runner.xml
run_log bounded l3-state-lifecycle env PYTHONPATH=/tests "$PY" -P -m pytest -p no:cacheprovider --rootdir=/tests -q \
    /tests/infb_verify/test_state_lifecycle_preservation.py --junitxml=/logs/verifier/junit-l3-state-lifecycle.xml

# Shared GDN numerics require no model support, so they remain independently scoreable
# on the unsolved base. Run before loading the full model to avoid a second CUDA context
# competing with its allocated cache. The reference ran the identical public mathematics.
run_log bounded gdn env PYTHONPATH=/tests "$PY" -P -m pytest \
    -p no:cacheprovider --rootdir=/tests -q -s /tests/infb_verify/test_gdn.py \
    --junitxml=/logs/verifier/junit-gdn.xml

run_log bounded served env PYTHONPATH=/tests "$PY" -I /tests/infb_efficiency/deadline.py run --group test_hidden \
    --budget-s "$SESSION_BUDGET" --config /tests/config.json \
    --junit /logs/verifier/junit-hidden.xml \
    --results /logs/verifier/efficiency/test_hidden/session-results.jsonl \
    --record /logs/verifier/efficiency/test_hidden/efficiency.json \
    --expect-prefix infb_verify.test_hidden. --cuda-devices 0,1,2,3 --always-reap -- \
    "$PY" -P -m pytest -p infb_efficiency.deadline -p no:cacheprovider --rootdir=/tests -q -s \
    /tests/infb_verify/test_hidden.py /tests/infb_verify/test_fidelity.py \
    --junitxml=/logs/verifier/junit-hidden.xml

# Binding instruments a fresh server before graph capture and is never used for speed
# measurement. Normal served tests, including KDA timing, have already torn down above.
run_log bounded binding env PYTHONPATH=/tests \
    INFB_DEADLINE_CONFIG=/tests/config.json "$PY" -P -m pytest -p infb_efficiency.deadline \
    -p no:cacheprovider --rootdir=/tests -q -s /tests/infb_verify/test_binding.py \
    --junitxml=/logs/verifier/junit-binding.xml
# Whatever the module left behind (a server that outlived its test) is stopped before grading; a vLLM
# server holds its GPUs until it is killed.
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true

run_log bounded lifecycle-tp4 env PYTHONPATH=/tests "$PY" -P -m infb_verify.state_lifecycle --session tp4
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
run_log bounded lifecycle-dp2 env PYTHONPATH=/tests "$PY" -P -m infb_verify.state_lifecycle --session dp2
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
run_log bounded lifecycle-apc env PYTHONPATH=/tests "$PY" -P -m infb_verify.state_lifecycle --session apc
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
run_log bounded lifecycle-pressure env PYTHONPATH=/tests "$PY" -P -m infb_verify.state_lifecycle --session pressure
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
run_log bounded lifecycle-reference "$PY" -I /tests/infb_verify/state_lifecycle.py --session reference
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
run_log bounded state-lifecycle env PYTHONPATH=/tests "$PY" -P -m pytest -p no:cacheprovider --rootdir=/tests -q -s \
    /tests/infb_verify/test_state_lifecycle.py --junitxml=/logs/verifier/junit-state-lifecycle.xml
# >>> END RUN TESTS <<<

# Surface raw suite output into our stdout (the harness captures it into test-stdout.txt) so failures
# are debuggable even when the framework report omits the reason. Reasons-per-test come from grade.
_seen=""
for _rl in "$RUN_LOG" /logs/verifier/*_run.log /logs/verifier/*-run.log /logs/verifier/*.log /logs/verifier/*.out; do
  [ -f "$_rl" ] && [ -s "$_rl" ] || continue
  case " $_seen " in *" $_rl "*) continue ;; esac
  case "${_rl##*/}" in *convert*.log|ctrf*.log|junit*.log) continue ;; esac
  _seen="$_seen $_rl"
  echo "===== raw suite output: ${_rl##*/} ====="
  cat "$_rl"
done 2>/dev/null
echo "===== grade ====="

# The disposition. tests/attribution.py attributes every scored id
# that did not pass and writes /logs/verifier/attribution.json: a valid candidate failure leaves the grader's reward
# as it is; otherwise a scored node censored by a wall that expired while every operation kept within its limit,
# blocked behind such a node, or failed by the reference, the environment or the verifier voids the trial. The
# grader grades either way, so ctrf.json and the junit reports keep every case (a censored node's failing case names
# it censored). A void trial has no reward: the grader's reward.json is kept as reward.void.json and the frame's
# trap writes the -1 sentinel, which attribution.json distinguishes from the sentinel's other uses. An attribution
# that cannot run is never a clean grade either.
python3 -I /tests/attribution.py
attribution=$?
python3 /tests/grader.py grade
if [ "$attribution" -ne 0 ]; then
  mv -f /logs/verifier/reward.json /logs/verifier/reward.void.json 2>/dev/null
  if [ "$attribution" -eq 3 ]; then
    log "void: no candidate failure, and a scored node was censored, blocked or failed by a non-candidate cause; no reward written (sentinel -1), see attribution.json"
  else
    log "void: the attribution could not run (exit $attribution); no reward written (sentinel -1)"
  fi
fi
log "reward.json=$(cat /logs/verifier/reward.json 2>/dev/null)"

# Uniform top level: keep only the canonical artifacts at /logs/verifier and tuck every
# framework-native report/log under reports/ (full provenance, no data dropped -- just moved).
mkdir -p /logs/verifier/reports 2>/dev/null
for _f in /logs/verifier/*; do
  case "${_f##*/}" in
    reward.json|reward.txt|reward.void.json|attribution.json|ctrf.json|run.log|test-stdout.txt|reports) continue ;;
  esac
  [ -f "$_f" ] && mv -f "$_f" /logs/verifier/reports/ 2>/dev/null
done
[ "${attribution:-0}" -eq 0 ] || exit 3
