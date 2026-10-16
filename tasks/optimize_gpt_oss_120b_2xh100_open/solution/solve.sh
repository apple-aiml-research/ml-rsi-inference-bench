#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -euo pipefail

workspace="${INFB_WORKSPACE:-/app}"
repo="${workspace}/repo"

if [[ ! -d "$repo" ]]; then
    echo "[oracle] $repo missing — running prepare_target_env.sh first" >&2
    if   [[ -f "${workspace}/scripts/prepare_target_env.sh" ]]; then
        bash "${workspace}/scripts/prepare_target_env.sh"
    elif [[ -f "${INFB_TASK_DIR:-}/environment/scripts/prepare_target_env.sh" ]]; then
        bash "${INFB_TASK_DIR}/environment/scripts/prepare_target_env.sh"
    else
        echo "[oracle] cannot find prepare_target_env.sh" >&2
        exit 2
    fi
fi

oracle_patch="${INFB_ORACLE_PATCH:-__vllm_noop__}"

if [[ "${oracle_patch}" == "__vllm_noop__" ]]; then
    cd "${repo}"
    noop_target="$(find . -maxdepth 4 -path '*/vllm/__init__.py' 2>/dev/null | head -1)"
    if [[ -z "${noop_target}" ]]; then
        echo "[oracle] FATAL: no */vllm/__init__.py under ${repo} — the prepare flatten/overlay is broken (a real agent run would fail the same way)." >&2
        find . -maxdepth 3 -name '__init__.py' 2>/dev/null | head -20 >&2
        exit 2
    fi
    printf '\n# inference-bench vllm oracle no-op (baseline reproduce)\n' >> "${noop_target}"
    git -c user.email='infb@example.com' -c user.name='infb' add -A
    git -c user.email='infb@example.com' -c user.name='infb' \
        commit -q -m "oracle: vllm no-op (baseline reproduce)"
    echo "[oracle] vllm no-op applied (1-line comment in ${noop_target}) — baseline reproduce" >&2
    git log --oneline inference_base..HEAD
    exit 0
fi

patch_file="$(dirname "${BASH_SOURCE[0]}")/${oracle_patch}"
if [[ ! -f "${patch_file}" ]]; then
    echo "[oracle] FATAL: ${patch_file} does not exist. For a deliberate no-op use INFB_ORACLE_PATCH=__vllm_noop__." >&2
    exit 3
fi
if ! grep -qE '^(\+\+\+|---) ' "${patch_file}"; then
    echo "[oracle] FATAL: ${patch_file} contains no diff, so git apply would apply nothing and the trial would be scored no_patch. For a deliberate no-op use INFB_ORACLE_PATCH=__vllm_noop__." >&2
    exit 3
fi
echo "[oracle] applying ${oracle_patch} ..."
git -C "${repo}" apply --3way "${patch_file}"

echo "[oracle] done"
