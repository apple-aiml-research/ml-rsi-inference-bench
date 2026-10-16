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
git diff --binary 1183049084a198c8d133edcb843a4e55076ed1ab HEAD > /logs/artifacts/model.patch 2>/dev/null || true
echo "[pre_artifacts] captured $(wc -c < /logs/artifacts/model.patch) bytes"
