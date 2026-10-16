//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface EvalSummary {
  metrics: Record<string, number | string>[];
}

export interface JobSummary {
  name: string;
  id: string | null;
  started_at: string | null;
  updated_at: string | null;
  finished_at: string | null;
  n_total_trials: number;
  n_completed_trials: number;
  n_errored_trials: number;
  datasets: string[];
  agents: string[];
  providers: string[];
  models: string[];
  environment_type: string | null;
  evals: Record<string, EvalSummary>;
  total_input_tokens: number | null;
  total_cached_input_tokens: number | null;
  total_output_tokens: number | null;
  total_cost_usd: number | null;
  total_agent_steps: number | null;
}

export interface AgentDatasetStats {
  n_trials: number;
  n_errors: number;
  metrics: Record<string, number | string>[];
}

export interface JobStats {
  n_completed_trials: number;
  n_errored_trials: number;
  n_running_trials: number;
  n_pending_trials: number;
  n_cancelled_trials: number;
  n_retries: number;
  evals: Record<string, AgentDatasetStats>;
}

export interface JobResult {
  id: string;
  started_at: string;
  updated_at: string | null;
  finished_at: string | null;
  n_total_trials: number;
  stats: JobStats;
  job_uri: string;
}

export interface TaskSummary {
  task_name: string;
  source: string | null;
  agent_name: string | null;
  model_provider: string | null;
  model_name: string | null;
  n_trials: number;
  n_completed: number;
  n_errors: number;
  exception_types: string[];
  avg_reward: number | null;
  avg_duration_ms: number | null;
  avg_input_tokens: number | null;
  avg_cached_input_tokens: number | null;
  avg_output_tokens: number | null;
  avg_cost_usd: number | null;
  avg_peak_context_tokens: number | null;
  avg_agent_steps: number | null;
}

export interface TrialSummary {
  name: string;
  task_name: string;
  id: string | null;
  source: string | null;
  agent_name: string | null;
  model_provider: string | null;
  model_name: string | null;
  reward: number | null;
  error_type: string | null;
  started_at: string | null;
  finished_at: string | null;
  input_tokens: number | null;
  cached_input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  peak_context_tokens: number | null;
  agent_steps: number | null;
}

export interface CritiqueRunSummary {
  name: string;
  id: string | null;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  n_items: number;
  n_completed_items: number;
  n_running_items: number;
  n_pending_items: number;
  n_missing_items: number;
  n_failed_items: number;
  agent_name: string | null;
  model_provider: string | null;
  model_name: string | null;
  environment_type: string | null;
  critique_uri: string | null;
  has_config: boolean;
  has_result: boolean;
}

export interface CritiqueItemSummary {
  source_trial_name: string;
  critique_trial_name: string | null;
  task_name: string | null;
  source: string | null;
  agent_name: string | null;
  model_provider: string | null;
  model_name: string | null;
  source_reward: number | null;
  source_error_type: string | null;
  cost_usd: number | null;
  rating: string | null;
  tags: string[];
  feedback: string | null;
  critique_values: Record<string, string | number | boolean | null>;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  error_type: string | null;
  has_metadata: boolean;
  has_result_json: boolean;
  has_result_md: boolean;
}

export interface CritiqueRunDetail {
  run: CritiqueRunSummary;
  config: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  items: CritiqueItemSummary[];
}

export type CritiqueHeatmapRowBy = "rating" | "tag";
export type CritiqueHeatmapColumnBy = "task" | "dataset";
export type CritiqueHeatmapSourceTrialsFilter =
  | "all"
  | "non_errored"
  | "errored"
  | "successful";
export type CritiqueHeatmapRatingFilter = "all" | "good" | "bad";

export interface CritiqueHeatmapRow {
  key: string;
  label: string;
  kind: string;
  value: string | null;
}

export interface CritiqueHeatmapColumn {
  key: string;
  label: string;
  source: string | null;
  task_name: string | null;
}

export interface CritiqueHeatmapCell {
  row_key: string;
  column_key: string;
  n_items: number;
  n_good: number;
  n_bad: number;
  n_errors: number;
  good_rate: number | null;
  bad_rate: number | null;
  rating_counts: Record<string, number>;
  tag_counts: Record<string, number>;
}

export interface CritiqueHeatmapData {
  rows: CritiqueHeatmapRow[];
  columns: CritiqueHeatmapColumn[];
  cells: Record<string, Record<string, CritiqueHeatmapCell>>;
}

export interface TrialCritiqueDetail {
  run_name: string;
  status: string;
  critique_uri: string | null;
  run_config: Record<string, unknown> | null;
  metadata: Record<string, unknown> | null;
  critique_result: Record<string, unknown> | null;
  markdown: string | null;
  log: string | null;
  exception_text: string | null;
  files: FileInfo[];
  manifest: unknown | null;
  has_item_dir: boolean;
  has_artifacts_dir: boolean;
  has_result_json: boolean;
  has_result_md: boolean;
}

export interface TimingInfo {
  started_at: string | null;
  finished_at: string | null;
}

export interface ExceptionInfo {
  exception_type: string;
  exception_message: string;
  exception_traceback: string;
  occurred_at: string;
}

export interface ModelInfo {
  name: string;
  provider: string;
}

export interface AgentInfo {
  name: string;
  version: string;
  model_info: ModelInfo | null;
}

export interface VerifierResult {
  rewards: Record<string, number> | null;
}

export interface StepResult {
  step_name: string;
  verifier_result: VerifierResult | null;
  exception_info: ExceptionInfo | null;
  agent_execution: TimingInfo | null;
  verifier: TimingInfo | null;
}

export interface TrialResult {
  id: string;
  task_name: string;
  trial_name: string;
  trial_uri: string;
  source: string | null;
  agent_info: AgentInfo;
  verifier_result: VerifierResult | null;
  exception_info: ExceptionInfo | null;
  started_at: string | null;
  finished_at: string | null;
  environment_setup: TimingInfo | null;
  agent_setup: TimingInfo | null;
  agent_execution: TimingInfo | null;
  verifier: TimingInfo | null;
  step_results: StepResult[] | null;
}

// Trajectory types (ATIF format)

// Multimodal content types (ATIF v1.6)
export interface ImageSource {
  media_type: string;
  path: string;
}

export interface ContentPart {
  type: "text" | "image";
  text?: string;
  source?: ImageSource;
}

export type MessageContent = string | ContentPart[];
export type ObservationContent = string | ContentPart[] | null;

export interface ToolCall {
  tool_call_id: string;
  function_name: string;
  arguments: Record<string, unknown>;
}

export interface SubagentTrajectoryRef {
  trajectory_id?: string | null;
  session_id?: string | null;
  trajectory_path?: string | null;
  extra?: Record<string, unknown> | null;
}

export interface ObservationResult {
  source_call_id: string | null;
  content: ObservationContent;
  subagent_trajectory_ref?: SubagentTrajectoryRef[] | null;
  extra?: Record<string, unknown> | null;
}

export interface Observation {
  results: ObservationResult[];
}

export interface StepMetrics {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  cached_tokens: number | null;
  cost_usd: number | null;
  extra?: Record<string, unknown> | null;
}

export interface Step {
  step_id: number;
  timestamp: string | null;
  source: "system" | "user" | "agent";
  model_name: string | null;
  reasoning_effort?: string | number | null;
  message: MessageContent;
  reasoning_content: string | null;
  tool_calls: ToolCall[] | null;
  observation: Observation | null;
  metrics: StepMetrics | null;
  is_copied_context?: boolean | null;
  llm_call_count?: number | null;
  extra?: Record<string, unknown> | null;
}

export interface TrajectoryAgent {
  name: string;
  version: string;
  model_name: string | null;
}

export interface FinalMetrics {
  total_prompt_tokens: number | null;
  total_completion_tokens: number | null;
  total_cached_tokens: number | null;
  total_cost_usd: number | null;
  total_steps: number | null;
  extra?: Record<string, unknown> | null;
}

export interface Trajectory {
  schema_version: string;
  session_id: string | null;
  trajectory_id?: string | null;
  agent: TrajectoryAgent;
  steps: Step[];
  notes: string | null;
  final_metrics: FinalMetrics | null;
  continued_trajectory_ref?: string | null;
  subagent_trajectories?: Trajectory[] | null;
  extra?: Record<string, unknown> | null;
}

export interface RewardCriterion {
  name: string;
  value: number;
  raw: boolean | number | string;
  weight: number;
  description?: string;
  reasoning?: string;
  error?: string | null;
}

export interface RewardJudge {
  model?: string;
  agent?: string;
  timeout?: number;
  reasoning_effort?: string;
  files?: string[];
  cwd?: string;
  isolated?: boolean;
  atif_trajectory?: string;
  reference?: string;
}

export interface RewardDetail {
  score: number;
  kind: "programmatic" | "llm" | "agent";
  criteria: RewardCriterion[];
  judge?: RewardJudge;
  judge_output?: string;
  warnings?: string[];
}

export type RewardDetails = Record<string, RewardDetail | RewardDetail[]>;

export interface VerifierOutput {
  stdout: string | null;
  stderr: string | null;
  ctrf: string | null;
  reward: Record<string, number> | null;
  reward_details: RewardDetails | null;
}

export interface FileInfo {
  path: string;
  name: string;
  is_dir: boolean;
  size: number | null;
}

export interface ModelPricing {
  model_name: string;
  input_cost_per_token: number | null;
  cache_read_input_token_cost: number | null;
  output_cost_per_token: number | null;
}

export interface CommandLog {
  index: number;
  content: string;
}

export interface AgentLogs {
  oracle: string | null;
  setup: string | null;
  commands: CommandLog[];
  summary: string | null;
}

export interface ArtifactManifestEntry {
  source: string;
  destination: string;
  type: "file" | "directory";
  status?: string;
}

export interface ArtifactsData {
  files: FileInfo[];
  manifest: ArtifactManifestEntry[] | null;
}

export interface FilterOption {
  value: string;
  count: number;
}

export interface JobFilters {
  agents: FilterOption[];
  providers: FilterOption[];
  models: FilterOption[];
}

export interface TaskFilters {
  agents: FilterOption[];
  providers: FilterOption[];
  models: FilterOption[];
  sources: FilterOption[];
  tasks: FilterOption[];
}

export interface CritiqueItemFilters {
  agents: FilterOption[];
  providers: FilterOption[];
  models: FilterOption[];
  sources: FilterOption[];
  tasks: FilterOption[];
  ratings: FilterOption[];
  tags: FilterOption[];
  statuses: FilterOption[];
}

export type JobHeatmapRowBy = "config" | "agent" | "model";
export type JobHeatmapColumnBy = "task" | "dataset";

export interface JobHeatmapRouteParams {
  job_name: string | null;
  source: string | null;
  agent_name: string | null;
  model_provider: string | null;
  model_name: string | null;
  task_name: string;
}

export interface JobHeatmapRow {
  key: string;
  label: string;
  job_name: string | null;
  agent_name: string | null;
  model_provider: string | null;
  model_name: string | null;
  reasoning_effort?: string | null;
}

export interface JobHeatmapColumn {
  key: string;
  label: string;
  source: string | null;
  task_name: string | null;
}

export interface JobHeatmapCell {
  row_key: string;
  column_key: string;
  n_trials: number;
  n_completed: number;
  n_errors: number;
  avg_reward: number | null;
  avg_duration_ms: number | null;
  avg_input_tokens: number | null;
  avg_cached_input_tokens: number | null;
  avg_output_tokens: number | null;
  avg_cost_usd: number | null;
  total_cost_usd: number | null;
  avg_peak_context_tokens: number | null;
  avg_agent_steps: number | null;
  exception_counts: Record<string, number>;
  dominant_exception: string | null;
  route_params: JobHeatmapRouteParams | null;
}

export interface JobHeatmapData {
  rows: JobHeatmapRow[];
  columns: JobHeatmapColumn[];
  cells: Record<string, Record<string, JobHeatmapCell>>;
}

// Task definition types (for task browser mode)

export interface TaskDefinitionSummary {
  name: string;
  version: string;
  source: string | null;
  metadata: Record<string, unknown>;
  has_instruction: boolean;
  has_environment: boolean;
  has_tests: boolean;
  has_solution: boolean;
  agent_timeout_sec: number | null;
  verifier_timeout_sec: number | null;
  os: string | null;
  cpus: number | null;
  memory_mb: number | null;
  storage_mb: number | null;
  gpus: number | null;
}

export interface TaskDefinitionDetail {
  name: string;
  task_dir: string;
  config: Record<string, unknown>;
  instruction: string | null;
  has_instruction: boolean;
  has_environment: boolean;
  has_tests: boolean;
  has_solution: boolean;
}

export interface TaskDefinitionFilters {
  difficulties: FilterOption[];
  categories: FilterOption[];
  tags: FilterOption[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
}
