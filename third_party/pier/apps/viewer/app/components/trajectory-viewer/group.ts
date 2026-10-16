//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * Split a flat list of `ViewStep`s into a leading system preamble
 * (collapsed banner above the conversation) and a list of turn rows
 * (each turn is exactly one ATIF step).
 *
 * Earlier versions split each ATIF step into two synthetic steps
 * (assistant + tool_result) and re-merged them here. Now that the view
 * sidecar carries `toolCalls` and `results` directly on the same
 * `ViewStep`, that round trip is gone — one ATIF step = one turn.
 */
import type { ViewStep } from "./types";

export interface GroupedTrajectory {
  /** Leading `source: system` steps, collapsed into a single banner
   *  above the conversation flow. */
  preamble: ViewStep[];
  /** One entry per non-preamble ATIF step (empty assistant steps
   *  filtered out). */
  turns: ViewStep[];
}

function isEmptyAssistant(step: ViewStep): boolean {
  if (step.step.source !== "agent") return false;
  const hasText = step.parts.some((p) => (p.text ?? "").trim().length > 0);
  const hasReasoning = !!step.step.reasoning_content?.trim();
  const hasCalls = step.toolCalls.length > 0;
  const hasResults = step.results.length > 0;
  return !hasText && !hasReasoning && !hasCalls && !hasResults;
}

export function groupSteps(steps: ViewStep[]): GroupedTrajectory {
  const preamble: ViewStep[] = [];
  const turns: ViewStep[] = [];

  let i = 0;
  while (i < steps.length && steps[i]!.step.source === "system") {
    preamble.push(steps[i]!);
    i += 1;
  }
  for (; i < steps.length; i += 1) {
    const s = steps[i]!;
    if (isEmptyAssistant(s)) continue;
    turns.push(s);
  }
  return { preamble, turns };
}
