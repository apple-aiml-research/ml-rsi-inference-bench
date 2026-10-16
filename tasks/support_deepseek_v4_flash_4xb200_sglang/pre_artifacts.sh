#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -uo pipefail
cd /app || exit 0
mkdir -p /logs/artifacts
git config --global --add safe.directory /app 2>/dev/null || true
git diff --binary eac98398e7adc70dd10d8db3ff77226b026000ce HEAD > /logs/artifacts/model.patch 2>/dev/null || true
echo "[pre_artifacts] captured $(wc -c < /logs/artifacts/model.patch) bytes"
