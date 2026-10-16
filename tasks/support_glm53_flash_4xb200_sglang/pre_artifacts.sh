#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -uo pipefail
cd /app || exit 0
mkdir -p /logs/artifacts
git config --global --add safe.directory /app 2>/dev/null || true
git diff --binary 9f6b624b4b4460f4b62aed3297cd2574dbf02b92 HEAD > /logs/artifacts/model.patch 2>/dev/null || true
echo "[pre_artifacts] captured $(wc -c < /logs/artifacts/model.patch) bytes"
