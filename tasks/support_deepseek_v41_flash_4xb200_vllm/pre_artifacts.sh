#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Capture the agent's committed work as the submission artifact: the diff
# between the starting commit (the sparse-indexer preparation commit, atop the evaluation-materials commit)
# and the agent's final HEAD.
set -uo pipefail
cd /app || exit 0
mkdir -p /logs/artifacts
git config --global --add safe.directory /app 2>/dev/null || true
git diff --binary 6ead86daeaed3fb406414cc61db23cb214bd7668 HEAD > /logs/artifacts/model.patch 2>/dev/null || true
echo "[pre_artifacts] captured $(wc -c < /logs/artifacts/model.patch) bytes"
