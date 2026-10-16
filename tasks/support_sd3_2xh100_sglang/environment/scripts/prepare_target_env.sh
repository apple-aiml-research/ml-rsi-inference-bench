#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

















workspace="${INFB_WORKSPACE:-$PWD}"
log() { printf '[prepare_target_env:sd3] %s\n' "$*" >&2; }

ART="${INFB_VLLM_TARGET_ENV_ARTIFACT:-${INFB_SGLANG_TARGET_ENV_ARTIFACT:-}}"












init_candidate_repo() {
  local repo="$1"
  local git_id=(-c user.email='infb@example.com' -c user.name='infb')
  printf '%s\n' '.venv/' '__pycache__/' '*.pyc' '*.so' >> "${repo}/.gitignore"
  if [[ -d "${repo}/.git" ]]; then
    git -C "${repo}" checkout -q -B agent_work
  else
    git -C "${repo}" init -q -b agent_work










    git -C "${repo}" config gc.auto 0
  fi
  git -C "${repo}" add -A
  git -C "${repo}" "${git_id[@]}" commit -q --allow-empty \
      -m 'pristine baseline (inference_base)'
  git -C "${repo}" tag -f inference_base HEAD >/dev/null
  log "candidate repo ${repo}: branch=agent_work tag=inference_base head=$(git -C "${repo}" rev-parse --short HEAD)"
}

_infb_image_src="${SGLANG_SOURCE_DIR:-${INFB_SGLANG_SOURCE_DIR:-}}"
if [[ -n "$ART" && -f "$ART/manifest.json" ]] || [[ -d "${_infb_image_src}/python/sglang" ]]; then
  if [[ -n "$ART" && -f "$ART/manifest.json" && -z "${_infb_image_src}" ]]; then





    _infb_route="artifact"
    log "ARTIFACT mode: $ART"
    eval "$(python3 - "$ART/manifest.json" "$ART" <<'PY'
import json, sys, shlex
d = json.load(open(sys.argv[1])); art = sys.argv[2]
print("SRC=" + shlex.quote(f"{art}/source"))
print("COMPAT=" + shlex.quote(f"{art}/compat"))
print("GOLDEN=" + shlex.quote(f"{art}/golden"))
PY
)"
  else









    SRC="${_infb_image_src}"




    COMPAT=""
    GOLDEN=""
    _infb_route="image"
    log "IMAGE mode: $SRC (pinned baseline source)"
  fi



  [[ -d "$SRC/python/sglang" ]] || { log "ERROR: source/ has no sglang tree ($SRC)"; exit 2; }









  _infb_scrub="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scrub_agent_surfaces.py"
  _infb_scrub_receipt="$workspace/.infb_agent_surface_scrub.json"
  [[ -f "$_infb_scrub" ]] || { log "FATAL: $_infb_scrub is missing; cannot remove the agent auto-injection surfaces"; exit 3; }
  if ! python3 "$_infb_scrub" scrub --tree "$SRC" --label engine_source --receipt "$_infb_scrub_receipt" \
        --route "${_infb_route:-unknown}"; then
    log "FATAL: the agent-surface scrub of $SRC failed; refusing to build a baseline that may still carry them"
    exit 3
  fi


  log "rsync source -> $workspace/repo"
  mkdir -p "$workspace/repo"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude '.claude/skills' "$SRC/" "$workspace/repo/"
  else rm -rf "$workspace/repo"; cp -a "$SRC" "$workspace/repo"; fi

















  find "$workspace/repo" -type d -path '*/.claude/skills' -prune -exec rm -rf {} + 2>/dev/null || true



  _infb_n_skills=$(find "$workspace/repo" -type d -path '*/.claude/skills' -not -path '*/.git/*' 2>/dev/null | wc -l || true)
  if [[ "$_infb_n_skills" -ne 0 ]]; then
    log "FATAL: $_infb_n_skills .claude/skills dir(s) survived the sweep under $workspace/repo"
    find "$workspace/repo" -type d -path '*/.claude/skills' -not -path '*/.git/*' 2>/dev/null | sed 's/^/  /' >&2
    exit 3
  fi



  if ! python3 "$_infb_scrub" scrub --tree "$workspace/repo" --label candidate_repo --receipt "$_infb_scrub_receipt" \
        --route "${_infb_route:-unknown}"; then
    log "FATAL: the agent-surface scrub of $workspace/repo failed; refusing to build the baseline"
    exit 3
  fi







  _sd3_sentinel="$workspace/repo/python/sglang/multimodal_gen/runtime/pipelines/stable_diffusion_3.py"
  if [[ -e "$_sd3_sentinel" ]]; then
    log "ERROR: baseline is contaminated — stable_diffusion_3.py present in the staged source"
    exit 2
  fi
  log "baseline verified: stable_diffusion_3.py absent from the staged tree"





















  rm -rf "$workspace/repo/.git"
  log "severed upstream history: repo/.git removed before candidate initialization"


  if [[ -e "$workspace/repo/.git" ]]; then
    log "FATAL: repo/.git still present after sever"
    exit 3
  fi
  _n_hist_skills=$(find "$workspace/repo" -path '*/.claude/skills/*' -type f 2>/dev/null | wc -l)
  if [[ "$_n_hist_skills" -ne 0 ]]; then
    log "FATAL: $_n_hist_skills .claude/skills file(s) present after sever"
    exit 3
  fi




  if [[ -z "${ART:-}" || ! -f "${ART}/manifest.json" || -n "${_infb_image_src}" ]]; then
    INNER_PY="${INFB_SGLANG_PYTHON:-${INFB_VLLM_PYTHON:-}}"
    if [[ -z "$INNER_PY" && -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
      INNER_PY="${VIRTUAL_ENV}/bin/python3"
    fi
    [[ -n "$INNER_PY" ]] || INNER_PY="$(command -v python3 || true)"
    SITE="$("$INNER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
    log "inner python (IMAGE route): $INNER_PY ($("$INNER_PY" -V 2>&1)) | site-packages=$SITE"


    rm -f "$SITE"/__editable__*sglang*.pth "$SITE"/__editable___sglang*.pth 2>/dev/null || true
  else









  SITE="$(ls -d "$ART"/venv/lib/python3.*/site-packages 2>/dev/null | head -1)"



  _venv_pyver="$(basename "$(dirname "$SITE")" 2>/dev/null)"   # e.g. python3.12
  INNER_PY="$(command -v "$_venv_pyver" || true)"
  if [[ -z "$INNER_PY" ]]; then
    for _c in "/opt/conda/bin/$_venv_pyver" "/usr/bin/$_venv_pyver" "/usr/local/bin/$_venv_pyver"; do
      [[ -x "$_c" ]] && { INNER_PY="$_c"; break; }
    done
  fi
  if [[ -z "$INNER_PY" ]]; then
    log "FATAL: artifact venv is $_venv_pyver but no matching interpreter on this host."
    log "  Falling back to a different minor version yields cp-ABI mismatch and"
    log "  ModuleNotFoundError: sglang in every downstream gate. Install $_venv_pyver"
    log "  (e.g. uv python install ${_venv_pyver#python}) or rebuild the artifact venv."
    log "  candidates seen: $(command -v python3 || true) $(python3 -V 2>&1)"
    exit 3
  fi
  log "inner python resolved: $INNER_PY ($("$INNER_PY" -V 2>&1)) matching venv $_venv_pyver"


  rm -f "$SITE"/__editable__*sglang*.pth "$SITE"/__editable___sglang*.pth 2>/dev/null || true
  fi
  [[ -n "${COMPAT:-}" && -d "$COMPAT" ]] && export LD_LIBRARY_PATH="$COMPAT:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$workspace/repo/python:$SITE:${PYTHONPATH:-}"

  # Persist the selected interpreter beyond this preparation process.
  INNER_PY="$("$INNER_PY" "$(dirname "${BASH_SOURCE[0]}")/python_entrypoints.py" install \
    --workspace "$workspace" --site-packages "$SITE" \
    --bin-dir "${INFB_PYTHON_BIN_DIR:-$workspace/.infb_bin}")"



















  INFB_SHIM_BIN="$workspace/.infb_bin"
  mkdir -p "$INFB_SHIM_BIN"
  if [[ -x "$ART/venv/bin/ninja" ]]; then
    ln -sf "$ART/venv/bin/ninja" "$INFB_SHIM_BIN/ninja"
  fi
  export PATH="$INFB_SHIM_BIN:${PATH:-}"
  _infb_ninja="$("$INNER_PY" -c "import shutil;print(shutil.which('ninja') or '')" 2>/dev/null || true)"
  if [[ -z "$_infb_ninja" ]]; then
    log "FATAL: 'ninja' is not resolvable by the INNER interpreter ($INNER_PY) even with"
    log "  PATH=$INFB_SHIM_BIN prepended. flashinfer's JIT would fail inside TextEncodingStage and"
    log "  the candidate would produce no image at all, scoring 0 for an environment reason."
    log "  source: $ART/venv/bin/ninja (exists: $([[ -x "$ART/venv/bin/ninja" ]] && echo yes || echo no))"
    exit 6
  fi


  _infb_py3="$(command -v python3 || true)"
  if [[ "$_infb_py3" == "$ART/venv/bin/"* ]]; then
    log "FATAL: python3 now resolves to the artifact venv ($_infb_py3); that interpreter cannot"
    log "  bootstrap in this container (no stdlib at its build prefix) and every bare python3 call"
    log "  downstream would die with ModuleNotFoundError: encodings."
    exit 6
  fi
  log "ninja -> $_infb_ninja via shim $INFB_SHIM_BIN (python3 still $_infb_py3)"
  log "overlay: python=$INNER_PY ($("$INNER_PY" -V 2>&1)) | site-packages=$SITE"


  "$INNER_PY" - <<'PY' 2>&1 | sed 's/^/[prepare_target_env:sd3]   /' | tail -4 || true
import torch; print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
PY
  if "$INNER_PY" -c 'import sglang.multimodal_gen.runtime.models.dits.stablediffusion3' 2>/dev/null; then
    log "WARNING: SD3 still importable after baseline (expected absent)"
  else
    log "baseline OK: SD3 module absent (DiffGenerator cannot resolve stable-diffusion-3-*)"
  fi











  if [ -z "${CUDA_HOME:-}" ]; then
    for _c in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/lib/cuda; do
      [ -d "$_c" ] && { CUDA_HOME="$_c"; break; }
    done
  fi
  if [ -z "${CUDA_HOME:-}" ]; then

    _nvdir="$("$INNER_PY" -c 'import os,nvidia;print(os.path.dirname(nvidia.__file__))' 2>/dev/null || true)"
    [ -n "$_nvdir" ] && [ -d "$_nvdir" ] && CUDA_HOME="$_nvdir"
  fi
  export CUDA_HOME
  log "CUDA_HOME=${CUDA_HOME:-<unset>} (deep_gemm import-time assert guard)"




  "$INNER_PY" - "$workspace/effective_target_env.json" "$INNER_PY" "$workspace/repo" "$SITE" "$GOLDEN" "$COMPAT" "${CUDA_HOME:-}" "$INFB_SHIM_BIN" "${_infb_route:-unknown}" "$_infb_scrub_receipt" <<'PY'
import json, sys
out, py, repo, site, golden, compat, cuda_home, venv_bin, route, scrub_receipt = sys.argv[1:]
json.dump({
    "python_executable": py, "sglang_src": repo, "repo": f"{repo}/python",
    "pythonpath": f"{repo}/python:{site}", "site_packages": site,
    "golden_dir": golden, "compat_dir": compat,

    "venv_bin": venv_bin,

    "cuda_home": cuda_home,


    "mode": f"{route}_inner_overlay",

    "agent_surface_scrub_receipt": scrub_receipt,
}, open(out, "w"), indent=2)
PY
  log "baseline ready (${_infb_route:-unknown} mode, inner-python overlay) at $workspace/repo; golden_dir=$GOLDEN"
  log "agent must reproduce the full SD3 support (model-specific files + shared-infra edits)."













  rm -rf "$workspace/repo/.git"
  log "dropped upstream git history from the candidate repo (answer containment)"

  init_candidate_repo "$workspace/repo"
















  model_link="${workspace}/model"
  unset model_path
  hub_dir_name="models--stabilityai--stable-diffusion-3-medium-diffusers"
  flat_dir_name="stable-diffusion-3-medium-diffusers"

  resolve_hub_snapshot() {
    local hub_root="$1"
    local sdir="${hub_root}/${hub_dir_name}/snapshots"
    if [[ -d "${sdir}" ]]; then

      ls -1d "${sdir}/"*/ 2>/dev/null | tail -1 | sed 's|/$||'
    fi
  }

  if [[ -n "${INFB_MODEL_PATH:-}" && -d "${INFB_MODEL_PATH}" ]]; then
    model_path="${INFB_MODEL_PATH}"
  fi
  if [[ -z "${model_path:-}" ]]; then
    _hf_home_probe="${HF_HOME:-${HOME:-/root}/.cache/huggingface}"
    candidate="$(resolve_hub_snapshot "${_hf_home_probe}/hub")"
    [[ -n "${candidate}" && -d "${candidate}" ]] && model_path="${candidate}"
  fi
  if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" ]]; then
    candidate="$(resolve_hub_snapshot "${INFB_ARTIFACT_CACHE}/hf/hub")"
    [[ -n "${candidate}" && -d "${candidate}" ]] && model_path="${candidate}"
  fi
  if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" \
        && -d "${INFB_ARTIFACT_CACHE}/models/${flat_dir_name}" ]]; then
    model_path="${INFB_ARTIFACT_CACHE}/models/${flat_dir_name}"
  fi
  if [[ -z "${model_path:-}" && -n "${ART:-}" && -d "${ART}/model" ]]; then
    model_path="${ART}/model"
  fi

  if [[ -n "${model_path:-}" && -d "${model_path}" ]]; then
    ln -sfn "${model_path}" "${model_link}"
    log "model symlink: ${model_link} -> ${model_path}"
  else
    log "WARNING: could not resolve weights for stabilityai/stable-diffusion-3-medium-diffusers (a GATED repo:"
    log "  accept the licence at https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers and"
    log "  download it on the host, then set INFB_MODEL_PATH or HF_HOME)"
  fi
  exit 0
fi


log "ERROR: no prepared SGLang source or target environment is available."
log "Build and run the task's environment/Dockerfile, which provides /opt/engine-src and /opt/venv."
log "For a prebuilt environment, set INFB_SGLANG_SOURCE_DIR or INFB_SGLANG_TARGET_ENV_ARTIFACT."
exit 2
