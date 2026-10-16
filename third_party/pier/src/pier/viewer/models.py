#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
"""API response models for the viewer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class EvalSummary(BaseModel):
    """Summary of metrics for an agent/model/dataset combination."""

    metrics: list[dict[str, Any]] = []


class JobSummary(BaseModel):
    """Summary of a job for list views."""

    name: str
    id: UUID | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    finished_at: datetime | None = None
    n_total_trials: int = 0
    n_completed_trials: int = 0
    n_errored_trials: int = 0
    datasets: list[str] = []
    agents: list[str] = []
    providers: list[str] = []
    models: list[str] = []
    environment_type: str | None = None
    evals: dict[str, EvalSummary] = {}
    total_input_tokens: int | None = None
    total_cached_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cost_usd: float | None = None
    total_agent_steps: int | None = None


class TaskSummary(BaseModel):
    """Summary of a task group (agent + model + dataset + task) for list views."""

    task_name: str
    source: str | None = None
    agent_name: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    n_trials: int = 0
    n_completed: int = 0
    n_errors: int = 0
    exception_types: list[str] = []
    avg_reward: float | None = None
    avg_duration_ms: float | None = None
    avg_input_tokens: float | None = None
    avg_cached_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_cost_usd: float | None = None
    avg_peak_context_tokens: float | None = None
    avg_agent_steps: float | None = None


class TrialSummary(BaseModel):
    """Summary of a trial for list views."""

    name: str
    task_name: str
    id: UUID | None = None
    source: str | None = None
    agent_name: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    reward: float | None = None
    error_type: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    peak_context_tokens: int | None = None
    agent_steps: int | None = None


class CritiqueRunSummary(BaseModel):
    """Summary of a critique run stored under a source job."""

    name: str
    id: UUID | None = None
    status: str = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    n_items: int = 0
    n_completed_items: int = 0
    n_running_items: int = 0
    n_pending_items: int = 0
    n_missing_items: int = 0
    n_failed_items: int = 0
    agent_name: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    environment_type: str | None = None
    critique_uri: str | None = None
    has_config: bool = False
    has_result: bool = False


class CritiqueItemSummary(BaseModel):
    """Summary of one source-trial critique item."""

    source_trial_name: str
    critique_trial_name: str | None = None
    task_name: str | None = None
    source: str | None = None
    agent_name: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    source_reward: float | None = None
    source_error_type: str | None = None
    cost_usd: float | None = None
    rating: str | None = None
    tags: list[str] = []
    feedback: str | None = None
    critique_values: dict[str, Any] = {}
    status: str = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_type: str | None = None
    has_metadata: bool = False
    has_result_json: bool = False
    has_result_md: bool = False


class CritiqueRunDetail(BaseModel):
    """Full viewer payload for a critique run."""

    run: CritiqueRunSummary
    config: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    items: list[CritiqueItemSummary] = []


class CritiqueHeatmapRow(BaseModel):
    """A grouped row in the critique heatmap."""

    key: str
    label: str
    kind: str
    value: str | None = None


class CritiqueHeatmapColumn(BaseModel):
    """A grouped column in the critique heatmap."""

    key: str
    label: str
    source: str | None = None
    task_name: str | None = None


class CritiqueHeatmapCell(BaseModel):
    """Aggregated critique stats for one heatmap crossing."""

    row_key: str
    column_key: str
    n_items: int = 0
    n_good: int = 0
    n_bad: int = 0
    n_errors: int = 0
    good_rate: float | None = None
    bad_rate: float | None = None
    rating_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}


class CritiqueHeatmapData(BaseModel):
    """Data for a critique run heatmap."""

    rows: list[CritiqueHeatmapRow]
    columns: list[CritiqueHeatmapColumn]
    cells: dict[str, dict[str, CritiqueHeatmapCell]]


class TrialCritiqueDetail(BaseModel):
    """Critique data applicable to a specific source trial."""

    run_name: str
    status: str = "pending"
    critique_uri: str | None = None
    run_config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    critique_result: dict[str, Any] | None = None
    markdown: str | None = None
    log: str | None = None
    exception_text: str | None = None
    files: list[FileInfo] = []
    manifest: Any | None = None
    has_item_dir: bool = False
    has_artifacts_dir: bool = False
    has_result_json: bool = False
    has_result_md: bool = False


class ModelPricing(BaseModel):
    """Per-token pricing rates for a model, sourced from LiteLLM."""

    model_name: str
    input_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    output_cost_per_token: float | None = None


class FileInfo(BaseModel):
    """Information about a file in a trial directory."""

    path: str  # Relative path from trial dir
    name: str  # File name
    is_dir: bool
    size: int | None = None  # File size in bytes (None for dirs)


class FilterOption(BaseModel):
    """A filter option with a value and count."""

    value: str
    count: int


class JobFilters(BaseModel):
    """Available filter options for jobs list."""

    agents: list[FilterOption]
    providers: list[FilterOption]
    models: list[FilterOption]


class TaskFilters(BaseModel):
    """Available filter options for tasks list within a job."""

    agents: list[FilterOption]
    providers: list[FilterOption]
    models: list[FilterOption]
    sources: list[FilterOption]
    tasks: list[FilterOption]


class CritiqueItemFilters(BaseModel):
    """Available filter options for critique items within a critique run."""

    agents: list[FilterOption]
    providers: list[FilterOption]
    models: list[FilterOption]
    sources: list[FilterOption]
    tasks: list[FilterOption]
    ratings: list[FilterOption]
    tags: list[FilterOption]
    statuses: list[FilterOption]


class TaskDefinitionSummary(BaseModel):
    """Summary of a task definition for list views."""

    name: str
    version: str = "1.0"
    source: str | None = None
    metadata: dict[str, Any] = {}
    has_instruction: bool = False
    has_environment: bool = False
    has_tests: bool = False
    has_solution: bool = False
    agent_timeout_sec: float | None = None
    verifier_timeout_sec: float | None = None
    os: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None


class TaskDefinitionDetail(BaseModel):
    """Full detail of a task definition."""

    name: str
    task_dir: str = ""
    config: dict[str, Any] = {}
    instruction: str | None = None
    has_instruction: bool = False
    has_environment: bool = False
    has_tests: bool = False
    has_solution: bool = False


class TaskDefinitionFilters(BaseModel):
    """Available filter options for task definitions list."""

    difficulties: list[FilterOption] = []
    categories: list[FilterOption] = []
    tags: list[FilterOption] = []


class JobHeatmapRouteParams(BaseModel):
    """Exact route params for drilling into a heatmap cell."""

    job_name: str | None = None
    source: str | None = None
    agent_name: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    task_name: str


class JobHeatmapRow(BaseModel):
    """A grouped row in the job heatmap."""

    key: str
    label: str
    job_name: str | None = None
    agent_name: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    reasoning_effort: str | None = None


class JobHeatmapColumn(BaseModel):
    """A grouped column in the job heatmap."""

    key: str
    label: str
    source: str | None = None
    task_name: str | None = None


class JobHeatmapCell(BaseModel):
    """Aggregated trial stats for one heatmap crossing."""

    row_key: str
    column_key: str
    n_trials: int = 0
    n_completed: int = 0
    n_errors: int = 0
    avg_reward: float | None = None
    avg_duration_ms: float | None = None
    avg_input_tokens: float | None = None
    avg_cached_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_cost_usd: float | None = None
    total_cost_usd: float | None = None
    avg_peak_context_tokens: float | None = None
    avg_agent_steps: float | None = None
    exception_counts: dict[str, int] = {}
    dominant_exception: str | None = None
    route_params: JobHeatmapRouteParams | None = None


class JobHeatmapData(BaseModel):
    """Data for the single-job heatmap view."""

    rows: list[JobHeatmapRow]
    columns: list[JobHeatmapColumn]
    cells: dict[str, dict[str, JobHeatmapCell]]  # row.key -> column.key -> cell
