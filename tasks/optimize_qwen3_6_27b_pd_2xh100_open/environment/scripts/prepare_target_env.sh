#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -euo pipefail

workspace="${INFB_WORKSPACE:-$PWD}"
target_env_artifact="${INFB_VLLM_TARGET_ENV_ARTIFACT:-}"

source_kind="direct_source"
if [[ -n "${target_env_artifact}" && -z "${VLLM_SOURCE_DIR:-}" && -z "${INFB_VLLM_SOURCE_DIR:-}" ]]; then
    source_dir="${target_env_artifact}/source"
    source_kind="target_env_artifact"
else
    source_dir="${VLLM_SOURCE_DIR:-${INFB_VLLM_SOURCE_DIR:-/opt/engine-src}}"
fi
target_dir="${workspace}/repo"
pinned_commit="${INFB_VLLM_PINNED_COMMIT:-v0.20.1}"

if [[ ! -d "${source_dir}/vllm" || ! -f "${source_dir}/pyproject.toml" ]]; then
    echo "vLLM source tree not found at ${source_dir}" >&2
    echo "Expected either:" >&2
    echo "  - INFB_VLLM_TARGET_ENV_ARTIFACT/source/" >&2
    echo "  - VLLM_SOURCE_DIR / INFB_VLLM_SOURCE_DIR" >&2
    echo "  - /opt/engine-src (the engine source baked into the image)" >&2
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
        "${source_dir}/" "${target_dir}/"
else
    cp -a "${source_dir}/." "${target_dir}/"
    rm -rf "${target_dir}/.git" "${target_dir}/.venv"
    find "${target_dir}" -type d -name __pycache__ -prune -exec rm -rf {} +
    find "${target_dir}" -type f -name '*.pyc' -delete
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

cat >> "${target_dir}/.gitignore" <<'EOF'

# inference-bench: venv-side compiled extensions exposed via symlink
# (see scripts/prepare_target_env.sh)
/vllm/*.abi3.so
/vllm/*.so
/vllm/_version.py
/vllm/third_party/deep_gemm
/vllm/third_party/deep_gemm/
/vllm/vllm_flash_attn/_vllm_fa*_C.abi3.so
/vllm/vllm_flash_attn/cute
EOF

git -C "${target_dir}" add -A
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
    commit -q -m "pristine vLLM ${pinned_commit} baseline (inference_base)"
git -C "${target_dir}" tag inference_base

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
if [[ -n "${venv_dir}" ]]; then
    venv_vllm=$(echo "${venv_dir}"/lib/python*/site-packages/vllm 2>/dev/null | head -1)
    if [[ -d "${venv_vllm}" ]]; then
        for f in "${venv_vllm}"/*.so "${venv_vllm}"/_version.py; do
            [[ -e "${f}" ]] || continue
            ln -sfn "${f}" "${target_dir}/vllm/$(basename "${f}")"
        done
        for sub in third_party/deep_gemm third_party/flashmla; do
            if [[ -d "${venv_vllm}/${sub}" ]]; then
                mkdir -p "${target_dir}/vllm/${sub}"
                for f in "${venv_vllm}/${sub}"/*.py "${venv_vllm}/${sub}"/*.so; do
                    [[ -e "${f}" ]] || continue
                    bn=$(basename "${f}")
                    if [[ ! -e "${target_dir}/vllm/${sub}/${bn}" ]]; then
                        ln -sfn "${f}" "${target_dir}/vllm/${sub}/${bn}"
                    fi
                done
                for d in "${venv_vllm}/${sub}"/*/; do
                    [[ -d "${d}" ]] || continue
                    bn=$(basename "${d}")
                    [[ "${bn}" == "__pycache__" ]] && continue
                    if [[ ! -e "${target_dir}/vllm/${sub}/${bn}" ]]; then
                        ln -sfn "${d%/}" "${target_dir}/vllm/${sub}/${bn}"
                    fi
                done
            fi
        done
        if [[ -d "${venv_vllm}/vllm_flash_attn" && -d "${target_dir}/vllm/vllm_flash_attn" ]]; then
            for f in "${venv_vllm}"/vllm_flash_attn/*.so; do
                [[ -e "${f}" ]] || continue
                ln -sfn "${f}" "${target_dir}/vllm/vllm_flash_attn/$(basename "${f}")"
            done
            _fa_overlaid=""
            for d in "${venv_vllm}"/vllm_flash_attn/*/; do
                [[ -d "${d}" ]] || continue
                _bn="$(basename "${d%/}")"
                [[ "${_bn}" == "__pycache__" ]] && continue
                if [[ ! -e "${target_dir}/vllm/vllm_flash_attn/${_bn}" ]]; then
                    ln -sfn "${d%/}" "${target_dir}/vllm/vllm_flash_attn/${_bn}"
                    _fa_overlaid="${_fa_overlaid} ${_bn}"
                fi
            done
            echo "[prepare_target_env] vllm_flash_attn wheel-only subpackages overlaid:${_fa_overlaid:- none}"
        fi
        echo "[prepare_target_env] overlaid compiled extensions from ${venv_vllm}"
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

resolve_qwen3_6_27b_snapshot() {
    local hub_root="$1"
    local sdir="${hub_root}/models--Qwen--Qwen3.6-27B/snapshots"
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
    candidate=$(resolve_qwen3_6_27b_snapshot "${hf_home}/hub")
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
        model_path="${candidate}"
    fi
fi
if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" ]]; then
    candidate=$(resolve_qwen3_6_27b_snapshot "${INFB_ARTIFACT_CACHE}/hf/hub")
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
        model_path="${candidate}"
    fi
fi
if [[ -z "${model_path:-}" && -n "${INFB_VLLM_TARGET_ENV_ARTIFACT:-}" \
      && -d "${INFB_VLLM_TARGET_ENV_ARTIFACT}/model" ]]; then
    model_path="${INFB_VLLM_TARGET_ENV_ARTIFACT}/model"
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
python - <<'PY' > "${workspace}/real_engine_source.json"
import json
import os

print(
    json.dumps(
        {
            "engine": "vllm",
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

echo "[prepare_target_env] vLLM source ready at ${target_dir}"
echo "[prepare_target_env] Git base: branch=agent_work tag=inference_base"
echo "[prepare_target_env] Source kind: ${source_kind}"

_infb_stager=""
for _c in "/app/harness/scripts/stage_editable_pkgs.sh" \
          "$(dirname "${BASH_SOURCE[0]}")/stage_editable_pkgs.sh"; do
    [[ -f "${_c}" ]] && { _infb_stager="${_c}"; break; }
done
if [[ -n "${_infb_stager}" ]]; then
    echo "[prepare_target_env] sourcing editable-pkg stager: ${_infb_stager}"
    source "${_infb_stager}"
    infb_stage_editable_pkgs "${target_dir}" "${venv_dir}"
else
    echo "[prepare_target_env] WARN: stage_editable_pkgs.sh not found (INFB_EDIT_POLICY_* will NOT open)" >&2
fi
