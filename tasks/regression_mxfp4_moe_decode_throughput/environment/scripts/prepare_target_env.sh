#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

workspace="${INFB_WORKSPACE:-$PWD}"
target_env_artifact="${INFB_VLLM_TARGET_ENV_ARTIFACT:-}"

source_kind="direct_source"
if [[ -n "${target_env_artifact}" && -z "${VLLM_SOURCE_DIR:-}" && -z "${INFB_VLLM_SOURCE_DIR:-}" && -z "${INFB_SGLANG_SOURCE_DIR:-}" ]]; then
  source_dir="${target_env_artifact}/source"
  source_kind="target_env_artifact"
else
  source_dir="${SGLANG_SOURCE_DIR:-${INFB_SGLANG_SOURCE_DIR:-${VLLM_SOURCE_DIR:-${INFB_VLLM_SOURCE_DIR:-/opt/engine-src}}}}"
fi
target_dir="${workspace}/repo"

if [[ ! -d "${source_dir}/python/sglang" || ! -f "${source_dir}/python/pyproject.toml" ]]; then
  echo "SGLang source tree not found at ${source_dir}" >&2
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

chown -R "$(id -u):$(id -g)" "${target_dir}" 2>/dev/null || true

git -C "${target_dir}" init -q -b agent_work
git -C "${target_dir}" config gc.auto 0
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
  config commit.gpgsign false

cat >> "${target_dir}/.gitignore" <<'EOF'

# inference-bench: the engine's compiled half, linked in from the image's harvest dir.
# See scripts/stage_engine_extensions.sh — these files are build output, not engine source,
# and the pinned tree has never carried them.
/python/sglang/**/*.so
/python/sglang/_version.py
EOF

git -C "${target_dir}" add -A
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
  commit -q -m "engine source baseline"
git -C "${target_dir}" tag inference_base

stage="$(dirname "${BASH_SOURCE[0]}")/stage_engine_extensions.sh"
[[ -f "${stage}" ]] || stage=/app/scripts/stage_engine_extensions.sh
engine_ext="${INFB_VLLM_ENGINE_EXT:-/opt/inference-bench/engine-ext}"
engine_pkg="$(basename "$(find "${engine_ext}" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)")"
if [[ -n "${engine_pkg}" ]]; then
  bash "${stage}" overlay "${target_dir}" "${engine_ext}"
else
  echo "[prepare_target_env] FATAL: no harvested engine artefacts at ${engine_ext}." >&2
  echo "  Build the image from environment/Dockerfile, or run" >&2
  echo "    bash ${stage} harvest <venv>" >&2
  echo "  against the venv you are bind-mounting. Do NOT fall back to linking them out of an" >&2
  echo "  installed engine package: this task's engine Python must exist in the image exactly once." >&2
  exit 3
fi

venv_dir="${INFB_VENV:-}"
if [[ -z "${venv_dir}" && -n "${target_env_artifact:-}" && -d "${target_env_artifact}/venv" ]]; then
  venv_dir="${target_env_artifact}/venv"
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
    "models--openai--gpt-oss-20b" "gpt-oss-20b")
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
fi
if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" ]]; then
  candidate=$(resolve_hf_snapshot "${INFB_ARTIFACT_CACHE}/hf/hub" \
    "models--openai--gpt-oss-20b" "gpt-oss-20b")
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
  if [[ -z "${model_path:-}" && -d "${INFB_ARTIFACT_CACHE}/models/gpt-oss-20b" ]]; then
    model_path="${INFB_ARTIFACT_CACHE}/models/gpt-oss-20b"
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
  echo "[prepare_target_env] WARNING: could not resolve Llama-3-8B-Instruct model path; set INFB_MODEL_PATH" >&2
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
