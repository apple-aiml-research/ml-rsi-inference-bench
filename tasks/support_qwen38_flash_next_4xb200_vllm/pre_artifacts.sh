#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Capture the agent's committed work as the submission artifact: the diff
# between the starting commit and the agent's final HEAD.
set -euo pipefail
cd /app
mkdir -p /logs/artifacts
rm -f /logs/artifacts/model.patch
git config --global --add safe.directory /app 2>/dev/null || true
# Keep the capture base aligned with task.toml, config.json and oracle.lock.
BASE=a5e63c9ee859ceb2f299ff55f063c7c32f94f08b
git cat-file -e "$BASE^{commit}"
git merge-base --is-ancestor "$BASE" HEAD
PATCH=$(mktemp /logs/artifacts/model.patch.XXXXXX)
trap 'rm -f "$PATCH"' EXIT
git diff --no-ext-diff --binary "$BASE" HEAD > "$PATCH"
mv "$PATCH" /logs/artifacts/model.patch
echo "[pre_artifacts] captured $(wc -c < /logs/artifacts/model.patch) bytes"
