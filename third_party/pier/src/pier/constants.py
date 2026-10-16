#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from pathlib import Path

PYPI_PACKAGE_NAME = "datacurve-pier"
CACHE_DIR = Path("~/.cache/pier").expanduser()
TASK_CACHE_DIR = CACHE_DIR / "tasks"
PACKAGE_CACHE_DIR = CACHE_DIR / "tasks" / "packages"
ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
