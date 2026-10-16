#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/oracle_fix.patch"
target_tree="${workspace}/repo/sglang"

if [[ ! -d "${target_tree}" ]]; then
  prepare_script="${workspace}/scripts/prepare_target_env.sh"
  if [[ ! -f "${prepare_script}" ]]; then
    prepare_script="/app/scripts/prepare_target_env.sh"
  fi
  if [[ -f "${prepare_script}" ]]; then
    echo "Running prepare_target_env.sh to stage the engine source tree..."
    bash "${prepare_script}"
  fi
fi

if [[ ! -d "${target_tree}" ]]; then
  echo "Reference target tree not found: ${target_tree}" >&2
  exit 2
fi

cd "${workspace}/repo"

if command -v patch >/dev/null 2>&1; then
  patch -p2 --batch --forward < "${patch_file}"
elif [[ -d .git ]] && command -v git >/dev/null 2>&1; then
  git apply -p2 --whitespace=nowarn "${patch_file}"
else
  echo "Neither 'patch' nor 'git' available — cannot apply the reference fix." >&2
  exit 2
fi

fixed="${target_tree}/kernels/ops/attention/fla/l2norm.py"
if ! grep -q 'do_not_specialize' "${fixed}"; then
  echo "FATAL: the patch reported success but the repaired file does not carry the change" >&2
  exit 3
fi

echo "Reference fix applied under $(realpath "${target_tree}")"
