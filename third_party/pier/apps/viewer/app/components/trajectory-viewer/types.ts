//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * View-layer types for the trajectory viewer.
 *
 * Design tenet: stay true to ATIF. We do NOT reshape the trajectory —
 * one ATIF Step = one ViewStep, with a thin sidecar of view-derived
 * fields (parsed timestamps, normalized content parts, pre-resolved
 * sub-agent refs, pretty-printed tool args, etc.). The original ATIF
 * objects are kept untouched on `step` / `trajectory` / `result` so any
 * inspector can drill back to the canonical payload.
 *
 * Field naming: ATIF fields stay snake_case (we read them via `.step.*`
 * etc.); only view-derived sidecar fields are camelCase, by convention.
 */
import type { ContentPart, ObservationResult, Step, ToolCall, Trajectory } from "~/lib/types";

export interface ResolvedToolCall {
  /** Original ATIF tool call, untouched. */
  call: ToolCall;
  /** Pretty-printed JSON of `call.arguments` for the input panel. */
  argsJson: string;
}

export interface ResolvedToolResult {
  /** Original ATIF observation result, untouched. */
  result: ObservationResult;
  /** Tool-call id this result pairs with. Falls back to a positional id
   *  when `source_call_id` is absent (some agents — mini-swe-agent in
   *  particular — emit observations without ids when call/result are 1:1
   *  positional). Empty string when no pairing is possible. */
  toolCallId: string;
  /** Concatenated text portion of `result.content`. */
  text: string;
  imageParts: Array<{ path: string; mediaType: string }>;
  isError: boolean;
  /** Pre-resolved sub-agent trajectory (when `subagent_trajectory_ref`
   *  resolves into the parent's `subagent_trajectories` array). */
  subagent?: ViewTrajectory;
}

export interface ViewStep {
  /** Original ATIF step, untouched. */
  step: Step;
  /** 0-based ordinal in `trajectory.steps` (preserved across
   *  preamble/turn split for things like flash + scroll). */
  index: number;
  /** Parsed `step.timestamp` as epoch ms, when present. */
  epochMs?: number;
  /** Normalized message content. A bare string in ATIF becomes
   *  `[{type: "text", text}]` so the renderer doesn't have to branch. */
  parts: ContentPart[];
  /** ATIF tool calls + view-derived fields (or `[]`). */
  toolCalls: ResolvedToolCall[];
  /** ATIF observation results + view-derived fields (or `[]`). */
  results: ResolvedToolResult[];
  /** `tool_call_id` → resolved result, for inline pairing in the row. */
  resultsByCallId: Map<string, ResolvedToolResult>;
  /** Results whose `source_call_id` doesn't match any tool call on this
   *  step. Rendered after the inline tool-call list so no data is hidden. */
  orphanResults: ResolvedToolResult[];
}

export interface ViewTrajectory {
  /** Original ATIF trajectory, untouched. */
  trajectory: Trajectory;
  agent: {
    name: string;
    version?: string;
    model?: string;
    /** "name · model" if model present, else "name". */
    displayName: string;
  };
  sessionId?: string;
  steps: ViewStep[];
  totalCostUsd?: number;
  totalSteps?: number;
}
