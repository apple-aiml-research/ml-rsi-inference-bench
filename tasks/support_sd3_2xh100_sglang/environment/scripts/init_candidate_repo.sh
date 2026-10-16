#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#



















set -euo pipefail

repo="${1:?usage: init_candidate_repo.sh <repo_dir> [extra_gitignore_line ...]}"
shift || true
if [[ ! -d "${repo}" ]]; then
  echo "[init_candidate_repo] ERROR: repo dir not found: ${repo}" >&2
  exit 1
fi



{
  printf '%s\n' '.venv/' '__pycache__/' '*.pyc' '*.so'
  for line in "$@"; do printf '%s\n' "${line}"; done
} >> "${repo}/.gitignore"

git_id=(-c user.email='infb@example.com' -c user.name='infb')

if [[ -d "${repo}/.git" ]]; then




  git -C "${repo}" checkout -q -B agent_work
  git -C "${repo}" add -A
  git -C "${repo}" "${git_id[@]}" commit -q --allow-empty -m 'pristine baseline (inference_base)'
  git -C "${repo}" tag -f inference_base HEAD >/dev/null
else


  git -C "${repo}" init -q -b agent_work
  git -C "${repo}" add -A
  git -C "${repo}" "${git_id[@]}" commit -q --allow-empty -m 'pristine baseline (inference_base)'
  git -C "${repo}" tag -f inference_base >/dev/null
fi

echo "[init_candidate_repo] ${repo}: branch=agent_work tag=inference_base head=$(git -C "${repo}" rev-parse --short HEAD)"
