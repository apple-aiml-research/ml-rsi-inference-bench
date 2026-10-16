#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Oracle solution for regression_fp8_decode_throughput.
#
# Applies upstream's merged fix for vllm-project/vllm#34900, unmodified. Two files, both pure Python
# under vllm/model_executor/, so the staged tree is authoritative and nothing has to be rebuilt.
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/mla_decode_quant_shape_fix.patch"
target_file="${workspace}/repo/vllm/model_executor/custom_op.py"
second_file="${workspace}/repo/vllm/model_executor/layers/attention/mla_attention.py"

if [[ ! -f "${target_file}" ]]; then
  prepare_script="${workspace}/scripts/prepare_target_env.sh"
  if [[ ! -f "${prepare_script}" ]]; then
    prepare_script="/app/scripts/prepare_target_env.sh"
  fi
  if [[ -f "${prepare_script}" ]]; then
    echo "Running prepare_target_env.sh to sync vLLM source tree..."
    bash "${prepare_script}"
  fi
fi

for f in "${target_file}" "${second_file}"; do
  if [[ ! -f "$f" ]]; then
    echo "Oracle target file not found: $f" >&2
    exit 2
  fi
done

cd "${workspace}/repo"

if command -v patch >/dev/null 2>&1; then
  patch -p1 --batch --forward < "${patch_file}"
elif [[ -d .git ]] && command -v git >/dev/null 2>&1; then
  git apply --whitespace=nowarn "${patch_file}"
else
  echo "Neither 'patch' nor 'git' available — cannot apply oracle fix." >&2
  exit 2
fi

for f in "${target_file}" "${second_file}"; do
  if ! grep -q "dynamic_arg_dims" "$f"; then
    echo "Oracle patch reported success but ${f} does not contain the fix" >&2
    exit 3
  fi
done

echo "Oracle fix applied to $(realpath "${target_file}") and $(realpath "${second_file}")"
