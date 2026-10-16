#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from pier.models.agent.context import AgentContext
from pier.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
from pier.models.trial.config import AgentConfig, EnvironmentConfig
from pier.models.trial.result import AgentInfo, ExceptionInfo, TimingInfo


class CritiqueConfig(BaseModel):
    """Configuration for one critique run over an existing Pier job."""

    run_name: str
    source_job_dir: Path
    prompt_path: Path
    agent: AgentConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float | None = None
    agent_setup_timeout_multiplier: float | None = None
    environment_build_timeout_multiplier: float | None = None
    n_concurrent: int = 1
    overwrite: bool = False
    trial_names: list[str] | None = None
    limit: int | None = None
    sample_seed: int | None = None
    filter_passing: bool | None = None
    source_agent_names: list[str] | None = None
    source_model_names: list[str] | None = None


class CritiqueItem(BaseModel):
    source_trial_dir: Path
    source_trial_name: str
    task_dir: Path


class CritiqueItemResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_trial_name: str
    critique_trial_name: str
    task_name: str
    task_id: LocalTaskId | GitTaskId | PackageTaskId
    task_checksum: str
    source_trial_uri: str
    critique_trial_uri: str
    agent_info: AgentInfo | None = None
    agent_result: AgentContext | None = None
    critique_result: dict[str, Any] | None = None
    exception_info: ExceptionInfo | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    environment_setup: TimingInfo | None = None
    agent_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None


class CritiqueJobResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_name: str
    source_job_dir: Path
    critique_dir: Path
    config: CritiqueConfig
    started_at: datetime | None = None
    finished_at: datetime | None = None
    item_results: list[CritiqueItemResult] = Field(default_factory=list)
    failed_items: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def n_items(self) -> int:
        return len(self.item_results) + len(self.failed_items)

    @computed_field
    @property
    def n_failed(self) -> int:
        return len(
            [r for r in self.item_results if r.exception_info is not None]
        ) + len(self.failed_items)
