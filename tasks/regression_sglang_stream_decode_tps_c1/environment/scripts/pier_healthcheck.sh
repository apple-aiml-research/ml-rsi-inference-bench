#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -uo pipefail

_HC_LOG=/logs/artifacts/pier_healthcheck.log
mkdir -p /logs/artifacts 2>/dev/null || true
printf '=== healthcheck attempt at %s ===\n' "$(date -u +%FT%TZ 2>/dev/null || echo unknown)" \
  >> "$_HC_LOG" 2>/dev/null || true

log() {
  printf '[healthcheck] %s\n' "$*"
  printf '[healthcheck] %s\n' "$*" >> "$_HC_LOG" 2>/dev/null || true
}

workspace="${INFB_WORKSPACE:-/app}"

postconditions() {
  local mode="${1:-report}"
  local fail=0
  _pc_log() { [[ "$mode" == report ]] && log "$@"; return 0; }

  if [[ -d "${workspace}/repo/.git" ]]; then
    _pc_log "workspace: ${workspace}/repo/.git present"
  else
    _pc_log "FATAL: ${workspace}/repo/.git absent -- there is nothing for the agent to edit"
    fail=1
  fi
  if git -C "${workspace}/repo" rev-parse -q --verify inference_base >/dev/null 2>&1; then
    _pc_log "diff base: inference_base present"
  else
    _pc_log "FATAL: the inference_base tag is absent -- the verifier has nothing to diff against"
    fail=1
  fi
  if [[ -e "${workspace}/model" ]]; then
    _pc_log "model: ${workspace}/model -> $(readlink -f "${workspace}/model" 2>/dev/null || printf '?')"
  elif [[ -n "${INFB_MODEL_PATH:-}" && -d "${INFB_MODEL_PATH}" ]]; then
    _pc_log "model: INFB_MODEL_PATH=${INFB_MODEL_PATH} (no ${workspace}/model link; this task resolves weights by env)"
  elif [[ -f /app/task.toml ]] && ! grep -q '^hf_model_id' /app/task.toml; then
    _pc_log "model: no hf_model_id declared, so no weights are staged for this task -- postcondition not applicable"
  else
    _pc_log "FATAL: preparation resolved no weights, so ${workspace}/model is absent. Provide them by ONE of:"
    _pc_log "  * INFB_MODEL_PATH=<dir>            -- bind-mount the weights, point this at the mount"
    _pc_log "  * HF_HOME=<dir>                    -- an ordinary Hugging Face hub cache"
    _pc_log "  * INFB_ARTIFACT_CACHE=<dir>        -- containing hf/hub/<model>/snapshots"
    fail=1
  fi
  return "$fail"
}

workspace_is_already_built() {
  postconditions probe || return 1
  [[ -n "$(find "${workspace}/repo" -mindepth 1 -maxdepth 1 -not -name .git -print -quit 2>/dev/null)" ]]
}

if bash /app/scripts/start_agent_timer.sh; then
  log "timer ready"
else
  log "WARNING: start_agent_timer.sh failed (rc=$?) -- the agent will not see its remaining budget"
fi

nsys_rc=0
nsys_out=$(timeout 20 nsys --version 2>&1) || nsys_rc=$?
nsys_probe=$(printf '%s\n' "$nsys_out" | head -2 | tr '\n' ' ')
case "$nsys_rc" in
  0)   log "profiler ready: ${nsys_probe:-nsys (no version banner)}" ;;
  124) log "WARNING: \`nsys --version\` did not return in 20s -- the on-PATH nsys is looping, not profiling."
       log "WARNING: agents should use the in-process torch profiler (--profiler-config.profiler=torch)." ;;
  127) log "WARNING: no nsys on PATH -- external profiling is unavailable; use the in-process torch profiler." ;;
  *)   log "WARNING: \`nsys --version\` exited ${nsys_rc} -- ${nsys_probe:-no output}" ;;
esac

prepare=/app/scripts/prepare_target_env.sh
if [[ ! -x "$prepare" && ! -f "$prepare" ]]; then
  log "FATAL: ${prepare} is missing from the image"
  exit 1
fi
# PROBE BEFORE PREPARING. prepare_target_env.sh calls itself idempotent, but its only mechanism is
# destroy-and-rebuild -- `rm -rf "${target_dir}"` followed by a fresh materialization -- and it has more
# than one caller. A runtime adapter may have already run it once by the time this healthcheck
# starts, and healthchecks are retried, so running it unconditionally tore a working workspace down and
# rebuilt it once per attempt.
#
# That is not merely wasteful. On the four tasks that stage a whole FA source tree, preparation ends in
# `git add -f` + `git commit --amend`, which fires a DETACHED `git gc --auto` -- gc.autoDetach defaults
# to true, so the gc outlives the script. It is still rewriting .git/objects/ when the NEXT invocation's
# `rm -rf` walks the tree, and the rm dies with
#     rm: cannot remove '<workspace>/repo/.git': Directory not empty
# which is fatal here and kills the run before agent setup, after the environment build has already
# been paid for. Reproduced 8/8 with that exact message, and 0/6
# once gc.auto=0 -- so the race is the cause, not a symptom.
#
# The public path is unchanged by this: with no adapter, nothing has prepared anything, the probe fails,
# and preparation runs exactly as it did before. Only the redundant second and later runs are skipped.
if workspace_is_already_built; then
  log "workspace already prepared -- skipping preparation (it is destroy-and-rebuild, not a no-op)"
else
log "preparing the target env ..."
_prep_out="$(mktemp 2>/dev/null || printf '/tmp/_hc_prep.%s' "$$")"
bash "$prepare" >"$_prep_out" 2>&1
rc=$?
cat "$_prep_out" 2>/dev/null || true
cat "$_prep_out" >> "$_HC_LOG" 2>/dev/null || true
rm -f "$_prep_out" 2>/dev/null || true
if [[ "$rc" -ne 0 ]]; then
  log "FATAL: prepare_target_env.sh failed (rc=${rc}); the agent would start with no workspace"
  exit "$rc"
fi
fi

postconditions report
exit $?

