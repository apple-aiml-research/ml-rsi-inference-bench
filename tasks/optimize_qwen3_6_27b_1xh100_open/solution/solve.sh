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
        echo "[oracle] cannot find prepare_target_env.sh (looked under workspace/scripts and INFB_TASK_DIR/environment/scripts)" >&2
        exit 2
    fi
fi

if [[ ! -d "$repo/.git" ]]; then
    echo "[oracle] $repo/.git missing; cannot apply patch as a commit." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/known_good.patch"

if [[ ! -f "$patch_file" ]]; then
    echo "[oracle] $patch_file missing." >&2
    exit 2
fi

# Apply the patch as a real commit so the verifier's git diff against
# inference_base captures it.
cd "$repo"
git apply --3way "$patch_file" 2>&1 || {
    rc=$?
    echo "[oracle] git apply failed (rc=$rc). Trying with -p0 / unidiff fallback..." >&2
    git apply --3way -p0 "$patch_file" 2>&1 || true
    if ! git diff --quiet inference_base 2>/dev/null; then
        echo "[oracle] (some diff produced; continuing with whatever applied)" >&2
    else
        echo "[oracle] no changes applied; oracle may have drifted vs current vLLM." >&2
        exit 3
    fi
}

git -c user.email='infb@example.com' -c user.name='infb' \
    add -A
git -c user.email='infb@example.com' -c user.name='infb' \
    commit -q -m "oracle: no-op placeholder (pipeline integration test)"

echo "[oracle] Applied known-good broad patch:"
git log --oneline inference_base..HEAD
