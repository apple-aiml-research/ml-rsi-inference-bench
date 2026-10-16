#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Prepare an SGLang source tree under workspace/repo/. Idempotent.
#
#   * FLATTEN layout: SGLang's package root is `source/python/sglang`, not
#     `source/sglang`. We rsync `source/python/` -> `workspace/repo/` so that
#     `repo/sglang` exists and PYTHONPATH=workspace/repo resolves
#     `import sglang` (the verifier sets PYTHONPATH=<repo> directly).
#   * NO compiled-extension symlinks: sglang/srt/** is pure Python; the
#     native bits (sgl_kernel, flashinfer) resolve from the venv directly,
#     so there are NO `vllm/*.so` symlinks and NO `.gitignore` vllm rules.
set -euo pipefail

workspace="${INFB_WORKSPACE:-$PWD}"
target_env_artifact="${INFB_VLLM_TARGET_ENV_ARTIFACT:-${INFB_SGLANG_TARGET_ENV_ARTIFACT:-}}"

source_kind="direct_source"
if [[ -n "${target_env_artifact}" && -z "${SGLANG_SOURCE_DIR:-}" && -z "${INFB_SGLANG_SOURCE_DIR:-}" ]]; then
    source_dir="${target_env_artifact}/source"
    source_kind="target_env_artifact"
else
    source_dir="${SGLANG_SOURCE_DIR:-${INFB_SGLANG_SOURCE_DIR:-/opt/engine-src}}"
fi
target_dir="${workspace}/repo"
pinned_commit="${INFB_SGLANG_PINNED_COMMIT:-v0.5.18}"

src_python="${source_dir}/python"
if [[ ! -d "${src_python}/sglang" || ! -f "${src_python}/pyproject.toml" ]]; then
    echo "SGLang source tree not found at ${source_dir}" >&2
    echo "Expected ${src_python}/sglang and ${src_python}/pyproject.toml, via one of:" >&2
    echo "  - INFB_VLLM_TARGET_ENV_ARTIFACT/source/ (serving env artifact)" >&2
    echo "  - SGLANG_SOURCE_DIR / INFB_SGLANG_SOURCE_DIR" >&2
    echo "  - /opt/engine-src (the source baked into the task image)" >&2
    exit 2
fi

rm -rf "${target_dir}"
mkdir -p "${target_dir}"

if command -v rsync >/dev/null 2>&1; then
    rsync -a \
        --exclude .git \
        --exclude .venv \
        --exclude __pycache__ \
        --exclude '*.pyc' \
        --exclude '.claude/skills' \
        "${src_python}/" "${target_dir}/"
else
    cp -a "${src_python}/." "${target_dir}/"
    rm -rf "${target_dir}/.git" "${target_dir}/.venv"
    find "${target_dir}" -type d -name __pycache__ -prune -exec rm -rf {} +
    find "${target_dir}" -type f -name '*.pyc' -delete
fi

# ── STRIP upstream's Claude Code skills (BENCHMARK-INTEGRITY, pre-agent) ──────
# Upstream SGLang ships agent-facing skills under .claude/skills/**. At the
# pinned tree that was 6 skills / 20 files under python/sglang/multimodal_gen/
# (sglang-diffusion-{add-model,performance,cuda-kernel,triton-kernel,
# benchmark-profile,ako4all-kernel}) — HOW-TO guides carrying kernel templates,
# per-GPU tuning tables, nsight recipes and ready-made benchmark scripts, i.e.
# large parts of what these tasks are supposed to MEASURE. They must never reach
# the candidate repo. The rsync --exclude above keeps them out on the rsync path;
# this sweep additionally covers (a) the `cp -a` fallback branch and (b) an
# already-populated ${target_dir}. Runs BEFORE the baseline commit below, so the
# files are absent from inference_base and their removal is NOT part of the
# agent's scored diff. `.claude/` itself is KEPT — only skills/ is dropped — so
# the tree stays otherwise faithful to upstream.
find "${target_dir}" -type d -path '*/.claude/skills' -prune -exec rm -rf {} + 2>/dev/null || true
# FAIL LOUD. The sweep above is best-effort (`|| true`, stderr dropped) because rm can
# legitimately race a read-only mount, but a SILENT failure would ship exactly what this
# block exists to remove. Assert instead: a surviving skills dir means the agent would see
# upstream HOW-TOs for the thing it is being measured on, so refuse to build the baseline.
_infb_n_skills=$(find "${target_dir}" -type d -path '*/.claude/skills' 2>/dev/null | wc -l || true)
if [[ "${_infb_n_skills}" -ne 0 ]]; then
    echo "[prepare_target_env] FATAL: ${_infb_n_skills} .claude/skills dir(s) survived the sweep under ${target_dir}" >&2
    find "${target_dir}" -type d -path '*/.claude/skills' 2>/dev/null | sed 's/^/  /' >&2
    exit 3
fi

# Initialize a clean git repo at the pinned-commit baseline so:
#  (a) the runner's candidate-artifact capture can `git diff` to extract
#      the agent's patch
#  (b) the verifier can `git reset --hard inference_base` during the
#      pristine baseline recheck, then restore via `git reset --hard <HEAD>`
#  (c) the agent has access to history-style `git log` for orientation
git -C "${target_dir}" init -q -b agent_work
git -C "${target_dir}" config gc.auto 0
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
    config commit.gpgsign false


cat > "${target_dir}/sitecustomize.py" <<'PYSITE'
"""inference-bench sglang: re-add the artifact venv's site-packages.

A `python -m venv`/`uv venv` shipped in the target-env artifact can, in the inner
container, resolve sys.prefix to the absent build-host base — leaving its own
site-packages (sglang deps incl. IPython/torch/sgl_kernel/flashinfer) off sys.path.
Fires at every venv-python startup (site runs under `-P`; this module is on
PYTHONPATH=<repo>). Re-adds the real site-packages so `import sglang` resolves.

Robustness notes:
  * Derive the venv from sys.executable WITHOUT realpath() — the venv's bin/python
    is typically a symlink to the (uv/system) base interpreter; realpath would land
    in the base, whose site-packages LACKS the artifact's installed deps.
  * Also try the artifact-root env vars, and validate that a candidate
    site-packages actually contains IPython/torch before trusting it.
"""
import glob
import os
import site
import sys

_roots = []
def _add_root(p):
    if p and p not in _roots:
        _roots.append(p)

# venv = parent-of-bin-dir of the (NON-realpath'd) interpreter path.
_add_root(os.path.dirname(os.path.dirname(sys.executable)))
for _ev in ("INFB_VLLM_TARGET_ENV_ARTIFACT", "INFB_SGLANG_TARGET_ENV_ARTIFACT"):
    _b = os.environ.get(_ev)
    if _b:
        _add_root(os.path.join(_b, "venv"))

_matched = []
for _root in _roots:
    for _sub in ("lib", "lib64"):
        for _sp in glob.glob(os.path.join(_root, _sub, "python*", "site-packages")):
            if os.path.isdir(_sp) and _sp not in _matched:
                _matched.append(_sp)

# Prefer dirs that actually hold the deps; otherwise add all matches.
_validated = [d for d in _matched
              if os.path.isdir(os.path.join(d, "IPython"))
              or os.path.isdir(os.path.join(d, "torch"))]
for _sp in (_validated or _matched):
    if _sp not in sys.path:
        site.addsitedir(_sp)
PYSITE

git -C "${target_dir}" add -A
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
    commit -q -m "pristine SGLang ${pinned_commit} baseline (inference_base)"
git -C "${target_dir}" tag inference_base

if [[ "${INFB_EDIT_POLICY_CSRC_SGLKERNEL:-}" =~ ^(1|true|yes)$ ]]; then
    sglk_src=""
    for _sglk_cand in "${source_dir}/sgl-kernel" "${source_dir}/python/sglang/kernels/aot"; do
        if [[ -d "${_sglk_cand}" ]]; then
            sglk_src="${_sglk_cand}"
            break
        fi
    done
    if [[ -z "${sglk_src}" ]]; then
        sglk_src="${source_dir}/sgl-kernel"   # keep the original path in the FATAL below
    fi
    if [[ -d "${sglk_src}" && ! -e "${target_dir}/sgl-kernel" ]]; then
        echo "[prepare_target_env] staging sgl-kernel source EDITABLE (INFB_EDIT_POLICY_CSRC_SGLKERNEL=1) from ${sglk_src}"
        if command -v rsync >/dev/null 2>&1; then
            rsync -a \
                --exclude '.git' \
                --exclude '**/.git' \
                --exclude '__pycache__' \
                --exclude '*.pyc' \
                --exclude 'build/' \
                --exclude 'dist/' \
                --exclude '*.so' \
                --exclude '.claude/skills' \
                "${sglk_src}/" "${target_dir}/sgl-kernel/"
        else
            cp -a "${sglk_src}" "${target_dir}/sgl-kernel"
            find "${target_dir}/sgl-kernel" -type d -name .git -prune -exec rm -rf {} + 2>/dev/null || true
            find "${target_dir}/sgl-kernel" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
            find "${target_dir}/sgl-kernel" -type f \( -name '*.pyc' -o -name '*.so' \) -delete 2>/dev/null || true
        fi
        {
            printf '/sgl-kernel/build/\n'
            printf '/sgl-kernel/dist/\n'
            printf '/sgl-kernel/**/*.so\n'
            printf '/sgl-kernel/**/__pycache__/\n'
            printf '/sgl-kernel/**/*.pyc\n'
            printf '/sgl-kernel/3rdparty/\n'
            printf '/.sglkredit/\n'

            _infb_deps_root="${INFB_CSRC_DEPS_DIR:-/opt/csrc-deps}"
            if [[ -s "${_infb_deps_root}/knobs.tsv" ]]; then
                _infb_editsrc="${target_dir}/sgl-kernel/_editsrc"
                if [[ -d "${_infb_editsrc}" && -n "$(ls -A "${_infb_editsrc}" 2>/dev/null)" ]]; then
                    echo "[prepare_target_env] sgl-kernel dep trees already staged (idempotent)"
                else
                    mkdir -p "${_infb_editsrc}"
                    _infb_copied=0
                    while IFS=$'\t' read -r _n _k _s _d; do
                        [[ -n "${_n}" && -n "${_d}" && "${_d}" != "-" ]] || continue
                        if [[ ! -d "${_infb_deps_root}/${_n}" ]]; then
                            echo "[prepare_target_env]   dep ${_n}: not in this image — FetchContent will try to fetch it at configure time" >&2
                            continue
                        fi
                        mkdir -p "${_infb_editsrc}/${_d}"
                        if cp -a "${_infb_deps_root}/${_n}/." "${_infb_editsrc}/${_d}/" 2>/dev/null; then
                            _infb_copied=$((_infb_copied + 1))
                        else
                            rm -rf "${_infb_editsrc}/${_d}"
                            echo "[prepare_target_env]   WARN: copying dep ${_n} failed; it will not be editable" >&2
                        fi
                    done < "${_infb_deps_root}/knobs.tsv"
                    if [[ "${_infb_copied}" -gt 0 ]]; then
                        find "${_infb_editsrc}" -maxdepth 2 -type d -name .git -prune -exec rm -rf {} + 2>/dev/null || true
                        find "${_infb_editsrc}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
                        find "${_infb_editsrc}" -type f -name '*.pyc' -delete 2>/dev/null || true
                        echo "[prepare_target_env] staged ${_infb_copied} editable kernel dep tree(s) under repo/sgl-kernel/_editsrc"
                    else
                        rm -rf "${_infb_editsrc}"
                        echo "[prepare_target_env] WARN: the image's dep table named no tree that exists here; repo/sgl-kernel/_editsrc not created." >&2
                    fi
                fi
            else
                echo "[prepare_target_env] WARN: this task declares the sgl-kernel surface open but the image staged no configure-time dep table at ${_infb_deps_root}/knobs.tsv. The agent-editable dep trees under repo/sgl-kernel/_editsrc will NOT exist, and a kernel rebuild may need network at configure time — which trials do not have. Refresh the image's configure-time dep table or stop baking the flag for this environment." >&2
            fi
        } >> "${target_dir}/.gitignore"
        find "${target_dir}/sgl-kernel" -type d -path '*/.claude/skills' -prune -exec rm -rf {} + 2>/dev/null || true
        _infb_n_sglk=$(find "${target_dir}/sgl-kernel" -type d -path '*/.claude/skills' 2>/dev/null | wc -l || true)
        if [[ "${_infb_n_sglk}" -ne 0 ]]; then
            echo "[prepare_target_env] FATAL: ${_infb_n_sglk} .claude/skills dir(s) survived under sgl-kernel/" >&2
            exit 3
        fi
        git -C "${target_dir}" add sgl-kernel .gitignore
        git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
            -c gc.auto=0 commit -q --amend --no-edit
        git -C "${target_dir}" tag -f inference_base >/dev/null
        n_cu="$(find "${target_dir}/sgl-kernel" -name '*.cu' -o -name '*.cuh' 2>/dev/null | wc -l)"
        echo "[prepare_target_env] sgl-kernel staged EDITABLE at repo/sgl-kernel (${n_cu} .cu/.cuh) + folded into inference_base"
    elif [[ ! -d "${sglk_src}" ]]; then
        echo "[prepare_target_env] FATAL: INFB_EDIT_POLICY_CSRC_SGLKERNEL=1 but ${sglk_src} not found — sgl-kernel cannot be opened. (This task's source_dir should be the image's /opt/engine-src monorepo ROOT; a python/-only tree cannot open it.) Refusing to continue." >&2
        exit 95
    fi
fi

venv_dir=""
if [[ -n "${target_env_artifact:-}" && -d "${target_env_artifact}/venv" ]]; then
    venv_dir="${target_env_artifact}/venv"
elif [[ -n "${INFB_VENV:-}" ]]; then
    venv_dir="${INFB_VENV}"
fi
if [[ -z "${venv_dir}" ]]; then
    helper_dir="$(dirname "${BASH_SOURCE[0]}")"
    if [[ -f "${helper_dir}/_target_env.py" ]]; then
        venv_python=$(python3 -c "
import sys
sys.path.insert(0, '${helper_dir}')
try:
    from _target_env import resolve_venv_python
    print(resolve_venv_python())
except Exception:
    pass
" 2>/dev/null || true)
        if [[ -n "${venv_python}" && -x "${venv_python}" ]]; then
            venv_dir="$(dirname "$(dirname "${venv_python}")")"
        fi
    fi
fi

# MANDATORY editable-overlay strip (HIGHEST-SEVERITY for the PYTHONPATH overlay
# to work). The artifact venv was `pip install -e source/python`, leaving a
# PEP660 finder + `.pth` in site-packages that points `import sglang` at the
# BUILD-TIME source path. With that finder present the PYTHONPATH=<repo>
# overlay loses and the agent's edits NEVER load. Remove both forms across
# every python*/site-packages dir in the venv.
if [[ -n "${venv_dir:-}" ]]; then
    stripped=0
    for site in "${venv_dir}"/lib/python*/site-packages; do
        [[ -d "${site}" ]] || continue
        for f in "${site}"/__editable__*sglang*.pth \
                 "${site}"/__editable___sglang*.pth \
                 "${site}"/__editable___sglang_*_finder.py \
                 "${site}"/__editable__.sglang*_finder.py; do
            if [[ -e "${f}" ]]; then
                if rm -f "${f}" 2>/dev/null; then
                    echo "[prepare_target_env] removed editable overlay: ${f}"
                    stripped=1
                else
                    echo "[prepare_target_env] NOTE: ${f} present but venv read-only (expected — stripped at build); PYTHONPATH overlay still wins"
                fi
            fi
        done
    done
    if [[ "${stripped}" -eq 0 ]]; then
        echo "[prepare_target_env] no editable sglang .pth/finder found (already stripped or non-editable install)"
    fi
fi

flashinfer_overlay="${workspace}/.flashinfer_overlay"
if [[ -n "${venv_dir:-}" ]]; then
    src_pkg=$(echo "${venv_dir}"/lib/python*/site-packages/flashinfer_cubin 2>/dev/null | head -1)
    if [[ -d "${src_pkg}" ]]; then
        rm -rf "${flashinfer_overlay}"
        mkdir -p "${flashinfer_overlay}"
        cp -aL "${src_pkg}" "${flashinfer_overlay}/flashinfer_cubin" 2>/dev/null \
            || cp -a "${src_pkg}" "${flashinfer_overlay}/flashinfer_cubin"
        chmod -R u+w "${flashinfer_overlay}"
        : > "${flashinfer_overlay}/.ready"
        echo "[prepare_target_env] flashinfer_cubin writable overlay: ${flashinfer_overlay}"
    fi
fi

model_link="${workspace}/model"
unset model_path

resolve_dsv4_flash_dspark_snapshot() {
    local hub_root="$1"
    local sdir="${hub_root}/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots"
    if [[ -d "${sdir}" ]]; then
        # shellcheck disable=SC2012
        ls -1d "${sdir}/"*/ 2>/dev/null | tail -1 | sed 's|/$||'
    fi
}

if [[ -n "${INFB_MODEL_PATH:-}" && -d "${INFB_MODEL_PATH}" ]]; then
    model_path="${INFB_MODEL_PATH}"
fi
if [[ -z "${model_path:-}" ]]; then
    hf_home="${HF_HOME:-${HOME:-/root}/.cache/huggingface}"
    candidate=$(resolve_dsv4_flash_dspark_snapshot "${hf_home}/hub")
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
        model_path="${candidate}"
    fi
fi
if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" ]]; then
    candidate=$(resolve_dsv4_flash_dspark_snapshot "${INFB_ARTIFACT_CACHE}/hf/hub")
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
        model_path="${candidate}"
    fi
fi
if [[ -z "${model_path:-}" && -n "${target_env_artifact:-}" \
      && -d "${target_env_artifact}/model" ]]; then
    model_path="${target_env_artifact}/model"
fi

if [[ -n "${model_path:-}" && -d "${model_path}" ]]; then
    ln -sfn "${model_path}" "${model_link}"
    echo "[prepare_target_env] model symlink: ${model_link} -> ${model_path}"
else
    echo "[prepare_target_env] WARNING: could not resolve model path; set INFB_MODEL_PATH" >&2
fi

# Stamp the source provenance so the verifier knows what to diff against.
SOURCE_KIND="${source_kind}" \
SOURCE_DIR="${source_dir}" \
TARGET_DIR="${target_dir}" \
TARGET_ENV_ARTIFACT="${target_env_artifact}" \
PINNED_COMMIT="${pinned_commit}" \
python3 - <<'PY' > "${workspace}/real_engine_source.json"
import json
import os

print(
    json.dumps(
        {
            "engine": "sglang",
            "source_kind": os.environ["SOURCE_KIND"],
            "source_dir": os.environ["SOURCE_DIR"],
            "target_dir": os.environ["TARGET_DIR"],
            "target_env_artifact": os.environ["TARGET_ENV_ARTIFACT"] or None,
            "pinned_commit": os.environ["PINNED_COMMIT"],
            "git_initialized": True,
            "base_branch": "agent_work",
            "base_tag": "inference_base",
        },
        indent=2,
        sort_keys=True,
    )
)
PY

echo "[prepare_target_env] SGLang source ready at ${target_dir}"
echo "[prepare_target_env] Git base: branch=agent_work tag=inference_base"
echo "[prepare_target_env] Source kind: ${source_kind}"
