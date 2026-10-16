#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Capture the agent's committed work as the submission artifact: the diff
# between the starting commit and the agent's final HEAD.
set -uo pipefail
cd /app || exit 0
mkdir -p /logs/artifacts
git config --global --add safe.directory /app 2>/dev/null || true
git diff --no-ext-diff --no-color --binary --full-index acb5b28091e784d9f6f0a0d61338aa31b5f4b86f HEAD > /logs/artifacts/model.patch 2>/dev/null || true
echo "[pre_artifacts] captured $(wc -c < /logs/artifacts/model.patch) bytes"
