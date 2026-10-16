//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * Build a `ViewTrajectory` sidecar over an ATIF trajectory.
 *
 * One ATIF step → one `ViewStep` (no splitting, no synthetic step kinds).
 * We only compute view-derived fields (`epochMs`, normalized content
 * parts, pretty-printed tool args, resolved sub-agent refs) and leave
 * the original ATIF objects on `step` / `trajectory` / `result` so any
 * downstream inspector can drill back to the untouched payload.
 */
import type {
  ContentPart,
  MessageContent,
  ObservationContent,
  ObservationResult,
  Step,
  ToolCall,
  Trajectory,
} from "~/lib/types";
import type { ResolvedToolCall, ResolvedToolResult, ViewStep, ViewTrajectory } from "./types";

/** ATIF content (string | ContentPart[] | null) → ContentPart[]. */
function toParts(content: MessageContent | ObservationContent): ContentPart[] {
  if (content == null) return [];
  if (typeof content === "string") {
    return content ? [{ type: "text", text: content }] : [];
  }
  // ATIF allows a missing `media_type`; default to png so <img> still tries.
  return content.map((p): ContentPart => {
    if (p.type === "image" && p.source) {
      return {
        type: "image",
        source: { ...p.source, media_type: p.source.media_type || "image/png" },
      };
    }
    return { type: "text", text: p.text ?? "" };
  });
}

function textOfParts(parts: ContentPart[]): string {
  return parts
    .filter((p) => p.type === "text")
    .map((p) => p.text ?? "")
    .join("\n");
}

function imagesOfParts(parts: ContentPart[]): Array<{ path: string; mediaType: string }> {
  const out: Array<{ path: string; mediaType: string }> = [];
  for (const p of parts) {
    if (p.type === "image" && p.source) {
      out.push({ path: p.source.path, mediaType: p.source.media_type });
    }
  }
  return out;
}

function epochOf(timestamp: string | null | undefined): number | undefined {
  if (!timestamp) return undefined;
  const t = Date.parse(timestamp);
  return Number.isNaN(t) ? undefined : t;
}

function resolveToolCall(call: ToolCall): ResolvedToolCall {
  return { call, argsJson: safeStringify(call.arguments ?? {}) };
}

/** Conservative error detection: only flag when the result content opens
 *  with a clear error marker, or when ATIF carries an explicit
 *  `extra.is_error` flag. Greedy keyword matching on "error" anywhere
 *  pulled in legit content (linter rules, log lines, "Errors: 0" in
 *  passing test output). */
function detectError(result: ObservationResult, head: string): boolean {
  const explicit = result.extra?.is_error;
  if (typeof explicit === "boolean") return explicit;
  return /^(Traceback|Error[:\s]|Exception[:\s]|FAILED|ERROR[:\s])/m.test(head);
}

function resolveToolResult(
  result: ObservationResult,
  fallbackCallId: string,
  subagentMap: Map<string, ViewTrajectory>,
): ResolvedToolResult {
  const parts = toParts(result.content);
  const text = textOfParts(parts);
  const isError = detectError(result, text.slice(0, 200));

  let subagent: ViewTrajectory | undefined;
  for (const ref of result.subagent_trajectory_ref ?? []) {
    if (ref.trajectory_id) {
      const sub = subagentMap.get(ref.trajectory_id);
      if (sub) {
        subagent = sub;
        break;
      }
    }
  }

  return {
    result,
    toolCallId: result.source_call_id ?? fallbackCallId,
    text,
    imageParts: imagesOfParts(parts),
    isError,
    subagent,
  };
}

function buildViewStep(
  step: Step,
  index: number,
  subagentMap: Map<string, ViewTrajectory>,
): ViewStep {
  const toolCalls = (step.tool_calls ?? []).map(resolveToolCall);
  const callIds = toolCalls.map((c) => c.call.tool_call_id);
  const rawResults = step.observation?.results ?? [];
  const results = rawResults.map((r, i) => resolveToolResult(r, callIds[i] ?? "", subagentMap));

  const resultsByCallId = new Map<string, ResolvedToolResult>();
  const orphanResults: ResolvedToolResult[] = [];
  const callIdSet = new Set(callIds);
  for (const r of results) {
    if (r.toolCallId && callIdSet.has(r.toolCallId)) {
      resultsByCallId.set(r.toolCallId, r);
    } else {
      orphanResults.push(r);
    }
  }

  return {
    step,
    index,
    epochMs: epochOf(step.timestamp),
    parts: toParts(step.message),
    toolCalls,
    results,
    resultsByCallId,
    orphanResults,
  };
}

/** Adapt a top-level ATIF trajectory (or a sub-agent one) into a
 *  view-friendly sidecar without altering its structure. */
export function adaptTrajectory(traj: Trajectory): ViewTrajectory {
  // Sub-agents are first-class in ATIF v1.7 (`subagent_trajectories`).
  // Build an id index up front so each `subagent_trajectory_ref` on a
  // tool result can resolve to a fully-built ViewTrajectory.
  const subagentMap = new Map<string, ViewTrajectory>();
  for (const sub of traj.subagent_trajectories ?? []) {
    if (sub.trajectory_id) subagentMap.set(sub.trajectory_id, adaptTrajectory(sub));
  }

  const steps = traj.steps.map((s, i) => buildViewStep(s, i, subagentMap));

  const agentName = traj.agent.name;
  const model = traj.agent.model_name ?? undefined;

  return {
    trajectory: traj,
    agent: {
      name: agentName,
      version: traj.agent.version,
      model,
      displayName: model ? `${agentName} · ${model}` : agentName,
    },
    sessionId: traj.session_id ?? undefined,
    steps,
    totalCostUsd: traj.final_metrics?.total_cost_usd ?? undefined,
    totalSteps: traj.final_metrics?.total_steps ?? undefined,
  };
}

function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}
