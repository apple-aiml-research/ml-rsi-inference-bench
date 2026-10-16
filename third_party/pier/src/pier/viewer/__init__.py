#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
"""Pier Viewer - Web UI for browsing jobs and trajectories."""

import os
from pathlib import Path

from pier.viewer.server import create_app


def create_app_from_env():
    """Factory function for uvicorn reload mode.

    Reads PIER_VIEWER_FOLDER and PIER_VIEWER_MODE from environment and creates the app.
    This is needed because uvicorn reload requires an import string, not an app instance.
    """
    folder = os.environ.get("PIER_VIEWER_FOLDER") or os.environ.get(
        "PIER_VIEWER_JOBS_DIR"
    )
    if not folder:
        raise RuntimeError("PIER_VIEWER_FOLDER environment variable not set")
    mode = os.environ.get("PIER_VIEWER_MODE", "jobs")
    return create_app(Path(folder), mode=mode)


__all__ = ["create_app", "create_app_from_env"]
