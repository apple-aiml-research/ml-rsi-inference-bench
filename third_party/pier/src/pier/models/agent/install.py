#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class InstallStep(BaseModel):
    """One install phase (maps to Dockerfile ``USER`` + ``RUN bash -c ...``)."""

    run: str
    user: Literal["root", "agent"] = "agent"
    env: dict[str, str] | None = None


class AgentInstallSpec(BaseModel):
    agent_name: str
    version: str | None = None
    steps: list[InstallStep]
    verification_command: str | None = None
    cache_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_empty(self) -> "AgentInstallSpec":
        if not self.steps:
            raise ValueError("AgentInstallSpec requires at least one step")
        return self

    def fingerprint(self) -> str:
        if self.cache_key:
            return self.cache_key
        steps_json = json.dumps(
            [s.model_dump(mode="python", exclude_none=True) for s in self.steps],
            sort_keys=True,
        )
        payload = "\n".join(
            [
                self.agent_name,
                self.version or "",
                steps_json,
                self.verification_command or "",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
