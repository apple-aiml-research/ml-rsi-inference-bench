#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
"""Tool call model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call within a step."""

    tool_call_id: str = Field(
        default=...,
        description="Unique identifier for this specific tool call",
    )
    function_name: str = Field(
        default=...,
        description="The name of the function or tool being invoked",
    )
    arguments: dict[str, Any] = Field(
        default=...,
        description="Arguments passed to the function (can be empty dict)",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Custom tool-call-level metadata (e.g., timeout, retry count, tool version). "
            "Added in ATIF-v1.7."
        ),
    )

    model_config = {"extra": "forbid"}
