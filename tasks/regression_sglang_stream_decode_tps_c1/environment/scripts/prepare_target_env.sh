#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

workspace="${INFB_WORKSPACE:-$PWD}"
target_env_artifact="${INFB_SGLANG_TARGET_ENV_ARTIFACT:-}"
target_dir="${workspace}/repo"
archive="${INFB_ENGINE_SRC_ARCHIVE:-/opt/inference-bench/engine-src.tar.gz}"

source_kind=""
src_python=""
extracted=""
cleanup() { [[ -n "${extracted}" && -d "${extracted}" ]] && rm -rf "${extracted}"; }
trap cleanup EXIT

for candidate in \
    "${SGLANG_SOURCE_DIR:-}" \
    "${INFB_SGLANG_SOURCE_DIR:-}" \
    "${target_env_artifact:+${target_env_artifact}/source}"; do
  [[ -n "${candidate}" ]] || continue
  if [[ -d "${candidate}/python/sglang" && -f "${candidate}/python/pyproject.toml" ]]; then
    source_kind="runtime_mount"
    src_python="${candidate}/python"
    break
  fi
done

if [[ -z "${src_python}" ]]; then
  if [[ -f "${archive}" ]]; then
    if [[ -f "${archive}.sha256" ]]; then
      want="$(awk 'NR==1{print $1}' "${archive}.sha256")"
      got="$(gzip -dc "${archive}" | sha256sum | awk '{print $1}')"
      if [[ "${want}" != "${got}" ]]; then
        echo "[prepare_target_env] FATAL: ${archive} does not match the digest this image" >&2
        echo "  recorded for it (want ${want}, have ${got}); refusing to stage it." >&2
        exit 5
      fi
    fi
    extracted="$(mktemp -d)"
    tar xzf "${archive}" -C "${extracted}"
    root="${extracted}/engine-src"
    if [[ -d "${root}/python/sglang" && -f "${root}/python/pyproject.toml" ]]; then
      source_kind="image_archive"
      src_python="${root}/python"
    fi
  fi
fi

if [[ -z "${src_python}" ]]; then
  echo "engine source not found" >&2
  echo "  need either the MONOREPO ROOT (a directory containing python/sglang and" >&2
  echo "  python/pyproject.toml) via one of:" >&2
  echo "    - INFB_SGLANG_SOURCE_DIR=<dir>            (or SGLANG_SOURCE_DIR)" >&2
  echo "    - INFB_SGLANG_TARGET_ENV_ARTIFACT=<dir>   (uses <dir>/source)" >&2
  echo "  or the image's own archive at ${archive} (INFB_ENGINE_SRC_ARCHIVE)." >&2
  exit 2
fi

echo "[prepare_target_env] engine source route: ${source_kind} (${src_python})"

rm -rf "${target_dir}"
mkdir -p "${target_dir}"

if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude .git \
    --exclude .venv \
    --exclude __pycache__ \
    --exclude '*.pyc' \
    "${src_python}/" "${target_dir}/"
else
  cp -a "${src_python}/." "${target_dir}/"
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

# inference-bench: build output, not engine source. `_version.py` is written by setuptools-scm
# during a build and the upstream tree gitignores it; the grpc extension module is compiled by
# setuptools-rust. Neither exists in a clean checkout, and neither is part of a repair.
/sglang/_version.py
/sglang/srt/grpc/_core*.so
EOF

git -C "${target_dir}" add -A
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
  commit -q -m "engine source baseline"
git -C "${target_dir}" tag inference_base

venv_python="${INFB_VLLM_PYTHON:-}"
if [[ -z "${venv_python}" && -n "${INFB_VENV:-}" && -x "${INFB_VENV}/bin/python" ]]; then
  venv_python="${INFB_VENV}/bin/python"
fi
if [[ -n "${venv_python}" && -x "${venv_python}" ]]; then
  if PYTHONPATH= "${venv_python}" -c 'import sglang' >/dev/null 2>&1; then
    echo "[prepare_target_env] FATAL: ${venv_python} can import the engine with an empty" >&2
    echo "  PYTHONPATH, so a second copy exists and the overlay at ${target_dir} may never load." >&2
    exit 4
  fi
  echo "[prepare_target_env] one-copy check: the venv cannot import an engine of its own"
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
  candidate=$(resolve_hf_snapshot "${hf_home}/hub" "models--openai--gpt-oss-120b" "gpt-oss-120b") || true
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
fi
if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" ]]; then
  candidate=$(resolve_hf_snapshot "${INFB_ARTIFACT_CACHE}/hf/hub" "models--openai--gpt-oss-120b" "gpt-oss-120b") || true
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
  if [[ -z "${model_path:-}" && -d "${INFB_ARTIFACT_CACHE}/models/gpt-oss-120b" ]]; then
    model_path="${INFB_ARTIFACT_CACHE}/models/gpt-oss-120b"
  fi
fi
if [[ -z "${model_path:-}" && -n "${target_env_artifact}" && -d "${target_env_artifact}/model" ]]; then
  model_path="${target_env_artifact}/model"
fi

if [[ -n "${model_path:-}" && -d "${model_path}" ]]; then
  ln -sfn "${model_path}" "${model_link}"
  echo "[prepare_target_env] model symlink: ${model_link} -> ${model_path}"
else
  echo "[prepare_target_env] WARNING: could not resolve the model weights; set INFB_MODEL_PATH" >&2
fi

SOURCE_KIND="${source_kind}" \
SOURCE_DIR="$( [[ "${source_kind}" == "runtime_mount" ]] && printf '%s' "${src_python}" || true )" \
SOURCE_ARCHIVE="$( [[ -f "${archive}" ]] && printf '%s' "${archive}" || true )" \
TARGET_DIR="${target_dir}" \
TARGET_ENV_ARTIFACT="${target_env_artifact}" \
VENV_DIR="${INFB_VENV:-}" \
MODEL_PATH="${model_path:-}" \
python3 - <<'PY' > "${workspace}/real_engine_source.json"
import json
import os

print(
    json.dumps(
        {
            "engine": "sglang",
            "source_kind": os.environ["SOURCE_KIND"],
            # The FLATTENED path (<mount>/python) when a runtime mount won, else null. Recording
            # <mount> instead would make the baseline copy a monorepo root while the candidate is a
            # package root, and every bench of the baseline would fail to import.
            "source_dir": os.environ["SOURCE_DIR"] or None,
            "source_archive": os.environ["SOURCE_ARCHIVE"] or None,
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

echo "[prepare_target_env] engine source ready at ${target_dir} (${source_kind})"
echo "[prepare_target_env] venv:  ${INFB_VENV:-<none>}"
echo "[prepare_target_env] model: ${model_path:-<none>}"
