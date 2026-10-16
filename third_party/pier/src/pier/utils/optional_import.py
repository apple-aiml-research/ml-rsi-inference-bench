#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
"""Helpers for environment integration dependencies.

The utilities here produce clear, actionable error messages when a user tries
to use a feature that requires a missing provider package.
"""

from pier.constants import PYPI_PACKAGE_NAME


class MissingExtraError(ImportError):
    """Raised when an optional dependency is not installed.

    Parameters
    ----------
    package:
        The PyPI package name that is missing (e.g. ``"daytona"``).
    extra:
        The future ``datacurve-pier`` extra that provides this package
        (e.g. ``"daytona"``).
    """

    def __init__(self, *, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"The '{package}' package is required but not installed. "
            f"Install it with:\n"
            f"  pip install {PYPI_PACKAGE_NAME}\n"
            f"  uv tool install {PYPI_PACKAGE_NAME}"
        )
