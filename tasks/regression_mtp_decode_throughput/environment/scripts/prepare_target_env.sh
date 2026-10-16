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
  if [[ "${INFB_RESOURCE_PROFILE:-}" == "harbor" && -z "${VLLM_SOURCE_DIR:-}" && -z "${INFB_VLLM_SOURCE_DIR:-}" ]]; then
    echo "Harbor/container runs must set INFB_VLLM_TARGET_ENV_ARTIFACT, or run an image whose" >&2
    echo "engine stage bakes INFB_VLLM_SOURCE_DIR (the current shape for this task)." >&2
    exit 2
  fi
  source_dir="${VLLM_SOURCE_DIR:-${INFB_VLLM_SOURCE_DIR:-/opt/engine-src}}"
fi
target_dir="${workspace}/repo"

if [[ ! -d "${source_dir}/vllm" || ! -f "${source_dir}/pyproject.toml" ]]; then
  echo "vLLM source tree not found at ${source_dir}" >&2
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

git -C "${target_dir}" init -q -b agent_work
git -C "${target_dir}" config gc.auto 0
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
  config commit.gpgsign false

cat >> "${target_dir}/.gitignore" <<'EOF'

# inference-bench: venv-side compiled extensions exposed via symlink
/vllm/*.abi3.so
/vllm/*.so
/vllm/_version.py
/vllm/vllm_flash_attn/_vllm_fa*_C.abi3.so
# EVERY subpackage of vllm_flash_attn, not `cute` by name. At this pin the built wheel ships
# `layers/` and `ops/` too, and all of them arrive as overlay symlinks rather than as files the
# agent authored -- so listing one by name would leave the others inside candidate.patch.
/vllm/vllm_flash_attn/*/
# At this pin `vllm/vllm_flash_attn/` holds only `.gitkeep` -- every `.py` that appears there is a
# symlink this script overlays from the built venv, never a file the agent authored, so keeping
# them out of candidate.patch is the same rule as the `.so` lines above.
/vllm/vllm_flash_attn/*.py
EOF

git -C "${target_dir}" add -A
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
  commit -q -m "engine source baseline"
git -C "${target_dir}" tag inference_base

venv_dir=""
if [[ -n "${INFB_VENV:-}" && -d "${INFB_VENV}" ]]; then
  venv_dir="${INFB_VENV}"
elif [[ -n "${target_env_artifact:-}" && -d "${target_env_artifact}/venv" ]]; then
  venv_dir="${target_env_artifact}/venv"
fi

if [[ -n "${venv_dir}" ]]; then
  venv_vllm=/opt/infb/engine-ext/vllm
  if [[ ! -d "${venv_vllm}" ]]; then
    venv_vllm=$(echo "${venv_dir}"/lib/python*/site-packages/vllm 2>/dev/null | head -1)
    echo "[prepare_target_env] WARNING: no harvested extensions at /opt/infb/engine-ext/vllm;" \
         "falling back to the venv copy at ${venv_vllm}, which carries a second," \
         "readable engine tree." >&2
  fi
  if [[ -d "${venv_vllm}" ]]; then
    for f in "${venv_vllm}"/*.so "${venv_vllm}"/_version.py; do
      [[ -e "${f}" ]] || continue
      ln -sfn "${f}" "${target_dir}/vllm/$(basename "${f}")"
    done
    if [[ -d "${venv_vllm}/vllm_flash_attn" && -d "${target_dir}/vllm/vllm_flash_attn" ]]; then
      for f in "${venv_vllm}"/vllm_flash_attn/*.so "${venv_vllm}"/vllm_flash_attn/*.py; do
        [[ -e "${f}" ]] || continue
        base=$(basename "${f}")
        if [[ "${base}" == *.py && -f "${target_dir}/vllm/vllm_flash_attn/${base}" ]]; then
          continue
        fi
        ln -sfn "${f}" "${target_dir}/vllm/vllm_flash_attn/${base}"
      done
        for d in "${venv_vllm}"/vllm_flash_attn/*/; do
          [[ -d "${d}" ]] || continue
          base=$(basename "${d}")
          [[ "${base}" == "__pycache__" ]] && continue
          [[ -e "${target_dir}/vllm/vllm_flash_attn/${base}" ]] && continue
          ln -sfn "${d%/}" "${target_dir}/vllm/vllm_flash_attn/${base}"
        done
        missing_fa=""
        for src in "${venv_vllm}"/vllm_flash_attn/*; do
          [[ -e "${src}" ]] || continue
          base=$(basename "${src}")
          [[ "${base}" == "__pycache__" ]] && continue
          [[ -e "${target_dir}/vllm/vllm_flash_attn/${base}" ]] || missing_fa="${missing_fa} ${base}"
        done
        if [[ -n "${missing_fa}" ]]; then
          echo "[prepare_target_env] FATAL: vllm/vllm_flash_attn is missing after the overlay:${missing_fa}" >&2
          echo "  The venv at ${venv_vllm} has them and the staged tree does not; the model registry" >&2
          echo "  would die on the first import that needs one." >&2
          exit 2
        fi
    fi
    echo "[prepare_target_env] overlaid compiled extensions from ${venv_vllm}"
  fi
fi

model_link="${workspace}/model"
unset model_path

resolve_hf_snapshot() {
  local hub_root="$1"
  local hf_repo_dir="$2"
  local flat_name="$3"
  local sdir="${hub_root}/${hf_repo_dir}/snapshots"
  if [[ -d "${sdir}" ]]; then
    ls -1d "${sdir}/"*/ 2>/dev/null | tail -1 | sed 's|/$||'
    return 0
  fi
  if [[ -d "${hub_root}/${flat_name}" ]]; then
    printf '%s\n' "${hub_root}/${flat_name}"
    return 0
  fi
}

if [[ -n "${INFB_MODEL_PATH:-}" && -d "${INFB_MODEL_PATH}" ]]; then
  model_path="${INFB_MODEL_PATH}"
fi
if [[ -z "${model_path:-}" ]]; then
  hf_home="${HF_HOME:-${HOME:-/root}/.cache/huggingface}"
  candidate=$(resolve_hf_snapshot "${hf_home}/hub" \
    "models--Qwen--Qwen3-Next-80B-A3B-Instruct" "Qwen3-Next-80B-A3B-Instruct")
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
fi
if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" ]]; then
  candidate=$(resolve_hf_snapshot "${INFB_ARTIFACT_CACHE}/hf/hub" \
    "models--Qwen--Qwen3-Next-80B-A3B-Instruct" "Qwen3-Next-80B-A3B-Instruct")
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
  if [[ -z "${model_path:-}" && -d "${INFB_ARTIFACT_CACHE}/models/Qwen3-Next-80B-A3B-Instruct" ]]; then
    model_path="${INFB_ARTIFACT_CACHE}/models/Qwen3-Next-80B-A3B-Instruct"
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
  echo "[prepare_target_env] WARNING: could not resolve Qwen3-Next-80B-A3B-Instruct model path; set INFB_MODEL_PATH" >&2
fi

SOURCE_KIND="${source_kind}" \
SOURCE_DIR="${source_dir}" \
TARGET_DIR="${target_dir}" \
TARGET_ENV_ARTIFACT="${target_env_artifact}" \
VENV_DIR="${venv_dir}" \
MODEL_PATH="${model_path:-}" \
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
            "venv_dir": os.environ["VENV_DIR"] or None,
            "model_path": os.environ["MODEL_PATH"] or None,
        },
        indent=2,
        sort_keys=True,
    )
)
PY

echo "[prepare_target_env] vLLM source ready at ${target_dir}"
echo "[prepare_target_env] venv:  ${venv_dir:-<none>}"
echo "[prepare_target_env] model: ${model_path:-<none>}"
