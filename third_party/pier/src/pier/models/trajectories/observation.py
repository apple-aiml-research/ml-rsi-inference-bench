#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
"""Observation model for ATIF trajectories."""

from pydantic import BaseModel, Field

from pier.models.trajectories.observation_result import ObservationResult


class Observation(BaseModel):
    """Environment feedback/result after actions or system events."""

    results: list[ObservationResult] = Field(
        default=...,
        description="Array of result objects from tool calls or actions",
    )

    model_config = {"extra": "forbid"}
