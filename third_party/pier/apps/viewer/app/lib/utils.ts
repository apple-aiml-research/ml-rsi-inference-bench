//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Pick the metric entry to display for a metric dict. Prefers the "reward"
 * key when present; otherwise falls back to the first entry.
 */
export function primaryMetricEntry(
  metric: Record<string, number | string>,
): [string, number | string] {
  if ("reward" in metric) return ["reward", metric.reward];
  return Object.entries(metric)[0];
}

/**
 * Split a chart label of the form "model [effort]" into its base name and the
 * bracketed effort suffix, so the effort can be rendered de-emphasised.
 * Labels without an effort suffix return `effort: null`.
 */
export function splitEffortLabel(label: string): {
  base: string;
  effort: string | null;
} {
  const match = label.match(/^(.*) (\[[^\]]+\])$/);
  if (match) return { base: match[1], effort: match[2] };
  return { base: label, effort: null };
}
