#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/mxfp4_moe_split_k_fix.patch"
target_file="${workspace}/repo/python/sglang/srt/layers/quantization/mxfp4.py"

if [[ ! -f "${target_file}" ]]; then
  prepare_script="${workspace}/scripts/prepare_target_env.sh"
  [[ -f "${prepare_script}" ]] || prepare_script="/app/scripts/prepare_target_env.sh"
  if [[ -f "${prepare_script}" ]]; then
    echo "Running prepare_target_env.sh to sync the SGLang source tree..."
    bash "${prepare_script}"
  fi
fi

if [[ ! -f "${target_file}" ]]; then
  echo "Oracle target file not found: ${target_file}" >&2
  exit 2
fi

cd "${workspace}/repo"

if command -v patch >/dev/null 2>&1; then
  patch -p1 --batch --forward < "${patch_file}"
elif [[ -d .git ]] && command -v git >/dev/null 2>&1; then
  git apply --whitespace=nowarn "${patch_file}"
else
  echo "Neither 'patch' nor 'git' available -- cannot apply the oracle fix." >&2
  exit 2
fi

# ASSERT THE EDIT LANDED, rather than trusting `patch --forward`'s exit code. With --forward, an
# already-applied patch is reported as a skip and still exits 0, so a solve.sh that ran against a
# tree someone had already modified would print success over a no-op. The scored arm would then be
# byte-identical to the baseline and the task would look mis-calibrated rather than mis-run.
grep -q '_is_sm90_supported' "${target_file}" \
  || { echo "FATAL: the patch reported success but ${target_file} does not carry the change" >&2; exit 3; }

echo "Oracle fix applied across $(grep -c '^diff --git ' "${patch_file}") engine file(s)"
