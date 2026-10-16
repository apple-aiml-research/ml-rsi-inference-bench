//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import type {
  AgentLogs,
  ArtifactsData,
  CritiqueHeatmapColumnBy,
  CritiqueHeatmapData,
  CritiqueHeatmapRatingFilter,
  CritiqueHeatmapRowBy,
  CritiqueHeatmapSourceTrialsFilter,
  CritiqueItemFilters,
  CritiqueItemSummary,
  CritiqueRunDetail,
  CritiqueRunSummary,
  FileInfo,
  JobFilters,
  JobHeatmapColumnBy,
  JobHeatmapData,
  JobHeatmapRowBy,
  JobResult,
  JobSummary,
  ModelPricing,
  PaginatedResponse,
  TaskDefinitionDetail,
  TaskDefinitionFilters,
  TaskDefinitionSummary,
  TaskFilters,
  TaskSummary,
  Trajectory,
  TrialCritiqueDetail,
  TrialResult,
  TrialSummary,
  VerifierOutput,
} from "./types";

// In production (served from same origin): use relative URL
// In dev: use VITE_API_URL environment variable
const API_BASE = import.meta.env.VITE_API_URL ?? "";

export interface ViewerConfig {
  folder: string;
  mode: "jobs" | "tasks";
  /** @deprecated Use folder instead */
  jobs_dir?: string;
}

export async function fetchConfig(): Promise<ViewerConfig> {
  const response = await fetch(`${API_BASE}/api/config`);
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchModelPricing(
  model: string
): Promise<ModelPricing | null> {
  const params = new URLSearchParams({ model });
  const response = await fetch(`${API_BASE}/api/pricing?${params}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch pricing: ${response.statusText}`);
  }
  return response.json();
}

export interface JobListFilters {
  search?: string;
  agents?: string[];
  providers?: string[];
  models?: string[];
  dates?: string[];
}

export async function fetchJobs(
  page: number = 1,
  pageSize: number = 100,
  filters?: JobListFilters
): Promise<PaginatedResponse<JobSummary>> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  if (filters?.search) {
    params.set("q", filters.search);
  }
  if (filters?.agents) {
    for (const agent of filters.agents) {
      params.append("agent", agent);
    }
  }
  if (filters?.providers) {
    for (const provider of filters.providers) {
      params.append("provider", provider);
    }
  }
  if (filters?.models) {
    for (const model of filters.models) {
      params.append("model", model);
    }
  }
  if (filters?.dates) {
    for (const date of filters.dates) {
      params.append("date", date);
    }
  }
  const response = await fetch(`${API_BASE}/api/jobs?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch jobs: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchJobFilters(): Promise<JobFilters> {
  const response = await fetch(`${API_BASE}/api/jobs/filters`);
  if (!response.ok) {
    throw new Error(`Failed to fetch job filters: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchJob(jobName: string): Promise<JobResult> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch job: ${response.statusText}`);
  }
  return response.json();
}

export interface CritiqueRunListFilters {
  search?: string;
  statuses?: string[];
  hasFailures?: boolean;
}

export async function fetchCritiqueRuns(
  jobName: string,
  page: number = 1,
  pageSize: number = 100,
  filters?: CritiqueRunListFilters
): Promise<PaginatedResponse<CritiqueRunSummary>> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  if (filters?.search) {
    params.set("q", filters.search);
  }
  if (filters?.statuses) {
    for (const status of filters.statuses) {
      params.append("status", status);
    }
  }
  if (filters?.hasFailures) {
    params.set("has_failures", "true");
  }
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/critiques?${params}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch critique runs: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchCritiqueRun(
  jobName: string,
  critiqueRunName: string,
  includeItems: boolean = true
): Promise<CritiqueRunDetail> {
  const params = new URLSearchParams();
  if (!includeItems) {
    params.set("include_items", "false");
  }
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/critiques/${encodeURIComponent(critiqueRunName)}?${params}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch critique run: ${response.statusText}`);
  }
  return response.json();
}

export interface CritiqueItemListFilters {
  search?: string;
  agents?: string[];
  providers?: string[];
  models?: string[];
  sources?: string[];
  tasks?: string[];
  ratings?: string[];
  tags?: string[];
  sourceTrials?: CritiqueHeatmapSourceTrialsFilter;
  statuses?: string[];
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export async function fetchCritiqueItems(
  jobName: string,
  critiqueRunName: string,
  page: number = 1,
  pageSize: number = 100,
  filters?: CritiqueItemListFilters
): Promise<PaginatedResponse<CritiqueItemSummary>> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  if (filters?.search) {
    params.set("q", filters.search);
  }
  if (filters?.agents) {
    for (const agent of filters.agents) {
      params.append("agent", agent);
    }
  }
  if (filters?.providers) {
    for (const provider of filters.providers) {
      params.append("provider", provider);
    }
  }
  if (filters?.models) {
    for (const model of filters.models) {
      params.append("model", model);
    }
  }
  if (filters?.sources) {
    for (const source of filters.sources) {
      params.append("source", source);
    }
  }
  if (filters?.tasks) {
    for (const task of filters.tasks) {
      params.append("task", task);
    }
  }
  if (filters?.ratings) {
    for (const rating of filters.ratings) {
      params.append("rating", rating);
    }
  }
  if (filters?.tags) {
    for (const tag of filters.tags) {
      params.append("tag", tag);
    }
  }
  if (filters?.sourceTrials) {
    params.set("source_trials", filters.sourceTrials);
  }
  if (filters?.statuses) {
    for (const status of filters.statuses) {
      params.append("status", status);
    }
  }
  if (filters?.sortBy) {
    params.set("sort_by", filters.sortBy);
  }
  if (filters?.sortOrder) {
    params.set("sort_order", filters.sortOrder);
  }
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/critiques/${encodeURIComponent(critiqueRunName)}/items?${params}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch critique items: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchCritiqueItemFilters(
  jobName: string,
  critiqueRunName: string
): Promise<CritiqueItemFilters> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/critiques/${encodeURIComponent(critiqueRunName)}/items/filters`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch critique item filters: ${response.statusText}`);
  }
  return response.json();
}

export interface CritiqueHeatmapFilters {
  rowBy?: CritiqueHeatmapRowBy;
  columnBy?: CritiqueHeatmapColumnBy;
  sourceTrials?: CritiqueHeatmapSourceTrialsFilter;
  rating?: CritiqueHeatmapRatingFilter;
  agents?: string[];
  providers?: string[];
  models?: string[];
  sources?: string[];
  tasks?: string[];
  ratings?: string[];
  tags?: string[];
  statuses?: string[];
  search?: string;
}

export async function fetchCritiqueHeatmap(
  jobName: string,
  critiqueRunName: string,
  filters?: CritiqueHeatmapFilters
): Promise<CritiqueHeatmapData> {
  const params = new URLSearchParams();
  if (filters?.rowBy) params.set("row_by", filters.rowBy);
  if (filters?.columnBy) params.set("column_by", filters.columnBy);
  if (filters?.sourceTrials) params.set("source_trials", filters.sourceTrials);
  if (filters?.rating) params.set("rating", filters.rating);
  if (filters?.agents) {
    for (const agent of filters.agents) {
      params.append("agent", agent);
    }
  }
  if (filters?.providers) {
    for (const provider of filters.providers) {
      params.append("provider", provider);
    }
  }
  if (filters?.models) {
    for (const model of filters.models) {
      params.append("model", model);
    }
  }
  if (filters?.sources) {
    for (const source of filters.sources) {
      params.append("source", source);
    }
  }
  if (filters?.tasks) {
    for (const task of filters.tasks) {
      params.append("task", task);
    }
  }
  if (filters?.ratings) {
    for (const rating of filters.ratings) {
      params.append("rating_value", rating);
    }
  }
  if (filters?.tags) {
    for (const tag of filters.tags) {
      params.append("tag", tag);
    }
  }
  if (filters?.statuses) {
    for (const status of filters.statuses) {
      params.append("status", status);
    }
  }
  if (filters?.search) params.set("q", filters.search);
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/critiques/${encodeURIComponent(critiqueRunName)}/heatmap?${params}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch critique heatmap: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteJob(jobName: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}`,
    { method: "DELETE" }
  );
  if (!response.ok) {
    throw new Error(`Failed to delete job: ${response.statusText}`);
  }
}

export interface TaskListFilters {
  search?: string;
  agents?: string[];
  providers?: string[];
  models?: string[];
  sources?: string[];
  tasks?: string[];
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export async function fetchTasks(
  jobName: string,
  page: number = 1,
  pageSize: number = 100,
  filters?: TaskListFilters
): Promise<PaginatedResponse<TaskSummary>> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  if (filters?.search) {
    params.set("q", filters.search);
  }
  if (filters?.agents) {
    for (const agent of filters.agents) {
      params.append("agent", agent);
    }
  }
  if (filters?.providers) {
    for (const provider of filters.providers) {
      params.append("provider", provider);
    }
  }
  if (filters?.models) {
    for (const model of filters.models) {
      params.append("model", model);
    }
  }
  if (filters?.sources) {
    for (const source of filters.sources) {
      params.append("source", source);
    }
  }
  if (filters?.tasks) {
    for (const task of filters.tasks) {
      params.append("task", task);
    }
  }
  if (filters?.sortBy) {
    params.set("sort_by", filters.sortBy);
  }
  if (filters?.sortOrder) {
    params.set("sort_order", filters.sortOrder);
  }
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/tasks?${params}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch tasks: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTaskFilters(jobName: string): Promise<TaskFilters> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/tasks/filters`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch task filters: ${response.statusText}`);
  }
  return response.json();
}

export type JobHeatmapTrialsFilter = "all" | "non_errored" | "successful";

export interface JobHeatmapFilters {
  search?: string;
  agents?: string[];
  providers?: string[];
  models?: string[];
  sources?: string[];
  tasks?: string[];
  rowBy?: JobHeatmapRowBy;
  columnBy?: JobHeatmapColumnBy;
  trialsFilter?: JobHeatmapTrialsFilter;
}

function buildJobHeatmapParams(filters?: JobHeatmapFilters): URLSearchParams {
  const params = new URLSearchParams();
  if (filters?.rowBy) {
    params.set("row_by", filters.rowBy);
  }
  if (filters?.columnBy) {
    params.set("column_by", filters.columnBy);
  }
  if (filters?.search) {
    params.set("q", filters.search);
  }
  if (filters?.agents) {
    for (const agent of filters.agents) {
      params.append("agent", agent);
    }
  }
  if (filters?.providers) {
    for (const provider of filters.providers) {
      params.append("provider", provider);
    }
  }
  if (filters?.models) {
    for (const model of filters.models) {
      params.append("model", model);
    }
  }
  if (filters?.sources) {
    for (const source of filters.sources) {
      params.append("source", source);
    }
  }
  if (filters?.tasks) {
    for (const task of filters.tasks) {
      params.append("task", task);
    }
  }
  if (filters?.trialsFilter === "non_errored") {
    params.set("exclude_errored", "true");
  } else if (filters?.trialsFilter === "successful") {
    params.set("only_successful", "true");
  }
  return params;
}

export async function fetchJobHeatmap(
  jobName: string,
  filters?: JobHeatmapFilters
): Promise<JobHeatmapData> {
  const params = buildJobHeatmapParams(filters);
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/heatmap?${params}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch job heatmap: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchComparisonHeatmap(
  jobNames: string[],
  filters?: JobHeatmapFilters
): Promise<JobHeatmapData> {
  const params = buildJobHeatmapParams(filters);
  for (const jobName of jobNames) {
    params.append("job", jobName);
  }
  const response = await fetch(`${API_BASE}/api/compare/heatmap?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch comparison heatmap: ${response.statusText}`);
  }
  return response.json();
}

export interface TrialFilters {
  taskName?: string;
  source?: string;
  agentName?: string;
  modelName?: string;
}

export async function fetchTrials(
  jobName: string,
  page: number = 1,
  pageSize: number = 100,
  filters?: TrialFilters
): Promise<PaginatedResponse<TrialSummary>> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });

  if (filters?.taskName) {
    params.set("task_name", filters.taskName);
  }
  if (filters?.source) {
    params.set("source", filters.source);
  }
  if (filters?.agentName) {
    params.set("agent_name", filters.agentName);
  }
  if (filters?.modelName) {
    params.set("model_name", filters.modelName);
  }

  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials?${params}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch trials: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTrial(
  jobName: string,
  trialName: string
): Promise<TrialResult> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch trial: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTrialCritiques(
  jobName: string,
  trialName: string
): Promise<TrialCritiqueDetail[]> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/critiques`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch trial critiques: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTrialCritique(
  jobName: string,
  trialName: string,
  critiqueRunName: string
): Promise<TrialCritiqueDetail> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/critiques/${encodeURIComponent(critiqueRunName)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch trial critique: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTrialCritiqueTrajectory(
  jobName: string,
  trialName: string,
  critiqueRunName: string
): Promise<Trajectory | null> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/critiques/${encodeURIComponent(critiqueRunName)}/trajectory`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch critique trajectory: ${response.statusText}`);
  }
  return response.json();
}

function stepQuery(step?: string | null): string {
  return step ? `?step=${encodeURIComponent(step)}` : "";
}

export async function fetchTrajectory(
  jobName: string,
  trialName: string,
  step?: string | null
): Promise<Trajectory | null> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/trajectory${stepQuery(step)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch trajectory: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchVerifierOutput(
  jobName: string,
  trialName: string,
  step?: string | null
): Promise<VerifierOutput> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/verifier-output${stepQuery(step)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch verifier output: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTrialFiles(
  jobName: string,
  trialName: string,
  step?: string | null
): Promise<FileInfo[]> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/files${stepQuery(step)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch trial files: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTrialFile(
  jobName: string,
  trialName: string,
  filePath: string,
  step?: string | null
): Promise<string> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/files/${filePath}${stepQuery(step)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch file: ${response.statusText}`);
  }
  return response.text();
}

export async function fetchArtifacts(
  jobName: string,
  trialName: string,
  step?: string | null
): Promise<ArtifactsData> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/artifacts${stepQuery(step)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch artifacts: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchAgentLogs(
  jobName: string,
  trialName: string,
  step?: string | null
): Promise<AgentLogs> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/agent-logs${stepQuery(step)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch agent logs: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchJobSummary(
  jobName: string
): Promise<{ summary: string | null }> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/summary`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch job summary: ${response.statusText}`);
  }
  return response.json();
}

export async function summarizeJob(
  jobName: string,
  model: string = "haiku",
  nConcurrent: number = 32,
  onlyFailed: boolean = true
): Promise<{
  summary: string | null;
  n_trials_summarized: number;
  job_summary_created: boolean;
}> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/summarize`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        n_concurrent: nConcurrent,
        only_failed: onlyFailed,
      }),
    }
  );
  if (!response.ok) {
    throw new Error(`Failed to summarize job: ${response.statusText}`);
  }
  return response.json();
}


export async function summarizeTrial(
  jobName: string,
  trialName: string,
  model: string = "haiku"
): Promise<{ summary: string | null }> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/summarize`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    }
  );
  if (!response.ok) {
    throw new Error(`Failed to summarize trial: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchExceptionText(
  jobName: string,
  trialName: string
): Promise<string | null> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/files/exception.txt`
  );
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch exception: ${response.statusText}`);
  }
  return response.text();
}

export async function fetchTrialLog(
  jobName: string,
  trialName: string
): Promise<string | null> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobName)}/trials/${encodeURIComponent(trialName)}/files/trial.log`
  );
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch trial log: ${response.statusText}`);
  }
  return response.text();
}

// Task definition API functions (task browser mode)

export interface TaskDefinitionListFilters {
  search?: string;
  difficulties?: string[];
  categories?: string[];
  tags?: string[];
}

export async function fetchTaskDefinitions(
  page: number = 1,
  pageSize: number = 100,
  filters?: TaskDefinitionListFilters
): Promise<PaginatedResponse<TaskDefinitionSummary>> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  if (filters?.search) {
    params.set("q", filters.search);
  }
  if (filters?.difficulties) {
    for (const d of filters.difficulties) {
      params.append("difficulty", d);
    }
  }
  if (filters?.categories) {
    for (const c of filters.categories) {
      params.append("category", c);
    }
  }
  if (filters?.tags) {
    for (const t of filters.tags) {
      params.append("tag", t);
    }
  }
  const response = await fetch(`${API_BASE}/api/task-definitions?${params}`);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch task definitions: ${response.statusText}`
    );
  }
  return response.json();
}

export async function fetchTaskDefinitionFilters(): Promise<TaskDefinitionFilters> {
  const response = await fetch(`${API_BASE}/api/task-definitions/filters`);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch task definition filters: ${response.statusText}`
    );
  }
  return response.json();
}

export async function fetchTaskDefinition(
  name: string
): Promise<TaskDefinitionDetail> {
  const response = await fetch(
    `${API_BASE}/api/task-definitions/${encodeURIComponent(name)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch task definition: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchTaskDefinitionFiles(
  name: string
): Promise<FileInfo[]> {
  const response = await fetch(
    `${API_BASE}/api/task-definitions/${encodeURIComponent(name)}/files`
  );
  if (!response.ok) {
    throw new Error(
      `Failed to fetch task definition files: ${response.statusText}`
    );
  }
  return response.json();
}

export async function fetchTaskDefinitionFile(
  name: string,
  filePath: string
): Promise<string> {
  const encodedPath = filePath.split('/').map(encodeURIComponent).join('/');
  const response = await fetch(
    `${API_BASE}/api/task-definitions/${encodeURIComponent(name)}/files/${encodedPath}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch file: ${response.statusText}`);
  }
  return response.text();
}

export async function sendTaskChatMessage(
  taskName: string,
  message: string,
  onDelta: (text: string) => void,
  onDone: () => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/task-definitions/${encodeURIComponent(taskName)}/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal,
    }
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onDone();
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") {
        onDone();
        return;
      }
      try {
        const event = JSON.parse(payload);
        if (event.type === "delta" && event.text) {
          onDelta(event.text);
        }
      } catch {
        // skip malformed lines
      }
    }
  }
  onDone();
}

export async function resetTaskChat(taskName: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/task-definitions/${encodeURIComponent(taskName)}/chat`,
    { method: "DELETE" }
  );
  if (!response.ok) {
    throw new Error(`Failed to reset chat: ${response.statusText}`);
  }
}
