#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Prepare a real vLLM source tree + model symlink + venv overlay.
#
# Idempotent: safe to re-run.
set -euo pipefail

workspace="${INFB_WORKSPACE:-$PWD}"
target_env_artifact="${INFB_VLLM_TARGET_ENV_ARTIFACT:-}"

# ---- Resolve vLLM source tree ----------------------------------------------
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
  # `/opt/engine-src` is where the engine stage puts the source tree, and this task's image bakes
  # INFB_VLLM_SOURCE_DIR pointing at it -- so this default is only reached if something unset that
  # var.
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

# ---- Initialize git repo -------------
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
/vllm/vllm_flash_attn/cute
# At this pin `vllm/vllm_flash_attn/` holds only `.gitkeep` -- every `.py` that appears there is a
# symlink this script overlays from the built venv, never a file the agent authored, so keeping
# them out of candidate.patch is the same rule as the `.so` lines above.
/vllm/vllm_flash_attn/*.py
# Same rule for the vendored third-party packages this script overlays from the built venv.
/vllm/third_party/flashmla/*.py
/vllm/third_party/triton_kernels/*.py
EOF

git -C "${target_dir}" add -A
git -C "${target_dir}" -c user.email='infb@example.com' -c user.name='infb' \
  commit -q -m "engine source baseline"
git -C "${target_dir}" tag inference_base

# ---- Resolve vLLM venv (provides .so extensions for PYTHONPATH overlay) ----
venv_dir=""
if [[ -n "${INFB_VENV:-}" && -d "${INFB_VENV}" ]]; then
  venv_dir="${INFB_VENV}"
elif [[ -n "${target_env_artifact:-}" && -d "${target_env_artifact}/venv" ]]; then
  venv_dir="${target_env_artifact}/venv"
fi

# NOT gated on ${venv_dir}: the harvest is the primary source and exists without one.
if true; then
  # The compiled half is HARVESTED out of the venv during the image build and the engine package is
  # then removed outright, so the image holds exactly one readable engine tree -- the one the agent
  # edits. Resolve the harvest first. The venv path below is reachable only on an image built before
  # that step landed, and it SAYS SO rather than silently restoring the second tree.
  venv_vllm=/opt/infb/engine-ext/vllm
  if [[ ! -d "${venv_vllm}" ]]; then
    venv_vllm=$(echo "${venv_dir}"/lib/python*/site-packages/vllm 2>/dev/null | head -1)
    echo "[prepare_target_env] WARNING: no harvested extensions at /opt/infb/engine-ext/vllm;" \
         "falling back to the venv copy at ${venv_vllm}, which carries a second," \
         "readable engine tree." >&2
  fi
  # FATAL, NOT SKIPPED. A missing source directory here has to stop the run: without the compiled
  # extensions the staged tree imports a Python-only engine, which runs, produces numbers, and
  # prints no error anywhere.
  if [[ ! -d "${venv_vllm}" ]]; then
    echo "[prepare_target_env] FATAL: no compiled engine extensions at /opt/infb/engine-ext/vllm," >&2
    echo "  and none under ${venv_dir:-<no venv resolved>}. The staged tree would import a" >&2
    echo "  Python-only engine and every downstream gate would fail for the wrong reason." >&2
    exit 2
  fi
  if [[ -d "${venv_vllm}" ]]; then
    for f in "${venv_vllm}"/*.so "${venv_vllm}"/_version.py; do
      [[ -e "${f}" ]] || continue
      ln -sfn "${f}" "${target_dir}/vllm/$(basename "${f}")"
    done
    if [[ -d "${venv_vllm}/vllm_flash_attn" && -d "${target_dir}/vllm/vllm_flash_attn" ]]; then
      # THE PYTHON FILES, NOT JUST THE .so -- and the pin is why. At this pin the whole of
      # `vllm/vllm_flash_attn/` in git is a single `.gitkeep`: that package's Python
      # (`__init__.py`, `flash_attn_interface.py`) is GENERATED at build time by CMake and exists
      # only in the built wheel. Overlay the `.so` alone and the directory stays an empty namespace
      # package, so `v1/attention/backends/fa_utils.py` dies at import with
      #     ImportError: cannot import name 'flash_attn_varlen_func' from
      #                  'vllm.vllm_flash_attn' (unknown location)
      # which takes down model registry inspection and with it the whole bench.
      #
      # The optimize tasks overlay only `*.so` and are CORRECT to: their pins ship the stub Python
      # in git. So the `.py` arm below SKIPS any name the source tree already has -- a pin that does
      # ship stubs keeps its own, and this stays a gap-filler rather than an override.
      for f in "${venv_vllm}"/vllm_flash_attn/*.so "${venv_vllm}"/vllm_flash_attn/*.py; do
        [[ -e "${f}" ]] || continue
        base=$(basename "${f}")
        if [[ "${base}" == *.py && -f "${target_dir}/vllm/vllm_flash_attn/${base}" ]]; then
          continue
        fi
        ln -sfn "${f}" "${target_dir}/vllm/vllm_flash_attn/${base}"
      done
      if [[ -d "${venv_vllm}/vllm_flash_attn/cute" && ! -e "${target_dir}/vllm/vllm_flash_attn/cute" ]]; then
        ln -sfn "${venv_vllm}/vllm_flash_attn/cute" "${target_dir}/vllm/vllm_flash_attn/cute"
      fi
      # Fail LOUD here rather than 20 minutes later inside a model-registry subprocess whose
      # traceback names neither this script nor the overlay.
      if [[ ! -e "${target_dir}/vllm/vllm_flash_attn/__init__.py" ]]; then
        echo "[prepare_target_env] FATAL: vllm/vllm_flash_attn has no __init__.py after the overlay." >&2
        echo "  Source pin ships none and the venv at ${venv_vllm} supplied none." >&2
        exit 2
      fi
    fi
    # SAME GAP, SECOND PACKAGE. `vllm/third_party/` is vendored submodule code: the WHEEL carries
    # `flashmla/flash_mla_interface.py` and the whole `triton_kernels/` package, while the source
    # tree at this pin carries only `third_party/__init__.py`, `flashmla/__init__.py` and
    # `pynvml.py`. Overlay nothing and the engine dies at import with
    #     ModuleNotFoundError: No module named 'vllm.third_party.flashmla.flash_mla_interface'
    # That is the allowlist working as intended: a name nobody thought of surfaces as an ImportError
    # on the first boot instead of failing silently later. The answer is to add the name, not to
    # widen this into a denylist. Skip any name the source tree already ships, so a pin that carries
    # its own stays authoritative.
    # WRITTEN OUT LITERALLY, twice, rather than looped -- a requirement, not a style choice. These
    # names are read STATICALLY by the build-time check that they are all harvested, so a `${sub}` or
    # a `$1` would be reported as unharvested -- correctly: a path only a human can expand is a path
    # no check can verify.
    for f in "${venv_vllm}"/third_party/flashmla/*.py; do
      [[ -e "${f}" ]] || continue
      mkdir -p "${target_dir}/vllm/third_party/flashmla"
      base=$(basename "${f}")
      [[ -f "${target_dir}/vllm/third_party/flashmla/${base}" ]] && continue
      ln -sfn "${f}" "${target_dir}/vllm/third_party/flashmla/${base}"
    done
    for f in "${venv_vllm}"/third_party/triton_kernels/*.py; do
      [[ -e "${f}" ]] || continue
      mkdir -p "${target_dir}/vllm/third_party/triton_kernels"
      base=$(basename "${f}")
      [[ -f "${target_dir}/vllm/third_party/triton_kernels/${base}" ]] && continue
      ln -sfn "${f}" "${target_dir}/vllm/third_party/triton_kernels/${base}"
    done
    if [[ -d "${venv_vllm}/third_party" && ! -e "${target_dir}/vllm/third_party/flashmla/flash_mla_interface.py" ]]; then
      echo "[prepare_target_env] FATAL: vllm/third_party/flashmla has no flash_mla_interface.py after" >&2
      echo "  the overlay. The source pin ships none and the harvest at ${venv_vllm} supplied none;" >&2
      echo "  the engine will die at import in the MLA backend. Add third_party/ to the Dockerfile's" >&2
      echo "  engine-extension harvest." >&2
      exit 2
    fi
    echo "[prepare_target_env] overlaid compiled extensions from ${venv_vllm}"
  fi
fi

# ---- Resolve DeepSeek-R1-0528-NVFP4 model (workspace/model symlink) -----------
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
    "models--nvidia--DeepSeek-R1-0528-NVFP4" "DeepSeek-R1-0528-NVFP4")
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
fi
if [[ -z "${model_path:-}" && -n "${INFB_ARTIFACT_CACHE:-}" ]]; then
  candidate=$(resolve_hf_snapshot "${INFB_ARTIFACT_CACHE}/hf/hub" \
    "models--nvidia--DeepSeek-R1-0528-NVFP4" "DeepSeek-R1-0528-NVFP4")
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    model_path="${candidate}"
  fi
  if [[ -z "${model_path:-}" && -d "${INFB_ARTIFACT_CACHE}/models/DeepSeek-R1-0528-NVFP4" ]]; then
    model_path="${INFB_ARTIFACT_CACHE}/models/DeepSeek-R1-0528-NVFP4"
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
  echo "[prepare_target_env] WARNING: could not resolve DeepSeek-R1-0528-NVFP4 model path; set INFB_MODEL_PATH" >&2
fi

# ---- Stamp provenance ------------------------------------------------------
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
