#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
set -euo pipefail
solution_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$solution_dir/install.py"
