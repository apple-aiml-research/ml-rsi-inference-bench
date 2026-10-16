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


# Reject incomplete or altered model inputs before charging the agent budget.
if ! python3 /app/scripts/model_assets.py verify --root /opt/infb/models \
    --receipt /logs/artifacts/model-assets.json; then
  log "FATAL: prepare the pinned SD3 and CLIP assets and mount them read-only at /opt/infb/models; see README.md"
  exit 1
fi

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


postconditions report || exit $?

# Validate the same default commands that a fresh agent shell will execute.
_entrypoint_helper="$(dirname "${BASH_SOURCE[0]}")/python_entrypoints.py"
if [[ ! -f "$workspace/python-entrypoints.json" ]]; then
  log "FATAL: Python entrypoint receipt is missing"
  exit 1
fi
if ! python3 "$_entrypoint_helper" check --workspace "$workspace"; then
  log "FATAL: Python environment or candidate imports are not ready"
  exit 1
fi
log "default python/python3 and DiffGenerator imports ready"
exit 0

