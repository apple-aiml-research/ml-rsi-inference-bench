//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * Top-level trajectory viewer.
 *
 * Renders the body of one ATIF trajectory:
 *   - per-step duration bar (one segment per step, width ∝ step duration)
 *   - the steps themselves, grouped into turns and rendered with a left
 *     gutter for cadence / step number
 *
 * Pier's host route wraps this in `<Card><CardHeader>Trajectory</CardHeader>`
 * with the "N steps / $X total" subtitle, so we don't render any chrome
 * here at depth 0 — the viewer is pure content.
 *
 * Sub-agent trajectories spawned inside a tool result render inline by
 * recursing into this same component with `depth + 1` (which suppresses
 * the gutter and the duration bar).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "~/lib/utils";
import type { Trajectory as ApiTrajectory } from "~/lib/types";
import { adaptTrajectory } from "./adapter";
import { groupSteps, type GroupedTrajectory } from "./group";
import { StepRow } from "./StepRow";
import { TrajectoryImageProvider, type TrajectoryImageContext } from "./TrajectoryImage";
import type { ResolvedToolResult, ViewStep, ViewTrajectory } from "./types";

export interface TrajectoryViewerProps {
  /** Either a raw ATIF (pier API shape) or a pre-adapted view one. */
  trajectory: ApiTrajectory | ViewTrajectory;
  /** Trial coordinates used to fetch inline images. When omitted, image
   *  parts render a `[path]` placeholder. */
  imageContext?: TrajectoryImageContext;
  /** Embedded mode (rendered as a sub-agent) — drops the gutter and
   *  duration bar. Internal: leave as 0 at the call site. */
  depth?: number;
  className?: string;
}

function isView(t: TrajectoryViewerProps["trajectory"]): t is ViewTrajectory {
  return "trajectory" in t && "agent" in t && (t as ViewTrajectory).agent.displayName != null;
}

export function TrajectoryViewer({
  trajectory: input,
  imageContext,
  depth = 0,
  className,
}: TrajectoryViewerProps) {
  const trajectory = useMemo<ViewTrajectory>(
    () => (isView(input) ? input : adaptTrajectory(input)),
    [input],
  );

  const grouped = useMemo(() => groupSteps(trajectory.steps), [trajectory.steps]);

  const stepRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const [highlightedStepId, setHighlightedStepId] = useState<number | null>(null);
  const [flashStepId, setFlashStepId] = useState<number | null>(null);
  const flashTimeout = useRef<number | null>(null);

  const startMs = useMemo(() => firstEpoch(trajectory), [trajectory]);

  const scrollToStep = useCallback((stepIndex: number) => {
    const target = stepRefs.current.get(stepIndex);
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlashStepId(null);
    requestAnimationFrame(() => setFlashStepId(stepIndex));
    if (flashTimeout.current != null) window.clearTimeout(flashTimeout.current);
    flashTimeout.current = window.setTimeout(() => {
      setFlashStepId((id) => (id === stepIndex ? null : id));
      flashTimeout.current = null;
    }, 900);
  }, []);

  useEffect(
    () => () => {
      if (flashTimeout.current != null) window.clearTimeout(flashTimeout.current);
    },
    [],
  );

  const registerRef = useCallback((stepIndex: number, el: HTMLDivElement | null) => {
    if (el) stepRefs.current.set(stepIndex, el);
    else stepRefs.current.delete(stepIndex);
  }, []);

  const renderSubagent = useCallback(
    (sub: NonNullable<ResolvedToolResult["subagent"]>) => (
      // Sub-agents inherit the outer image context via the same provider —
      // we don't re-wrap, since they share the trial directory.
      <TrajectoryViewer trajectory={sub} depth={depth + 1} />
    ),
    [depth],
  );

  if (trajectory.steps.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
        No trajectory data
      </div>
    );
  }

  // Pre-flatten so each row knows its predecessor's epoch — for the
  // cadence delta in the gutter.
  const turns = grouped.turns;
  let prev: number | undefined;
  const prevEpochByTurn: (number | undefined)[] = turns.map((s) => {
    const v = prev;
    if (s.epochMs != null) prev = s.epochMs;
    return v;
  });

  const allSteps: ViewStep[] = [...grouped.preamble, ...turns];
  const allPrev: (number | undefined)[] = [
    ...grouped.preamble.map(() => undefined),
    ...prevEpochByTurn,
  ];

  const body = (
    // `min-w-0` prevents grid/flex children from forcing overflow when a
    // deeply nested code block has long lines.
    <div className={cn("flex min-w-0 flex-col", className)}>
      <FlashStyles />
      {depth === 0 && (
        <StepDurationBar
          grouped={grouped}
          onStepClick={scrollToStep}
          onStepHover={setHighlightedStepId}
          highlightedStepIndex={highlightedStepId}
        />
      )}

      <div className={cn("min-w-0", depth === 0 ? "pb-[20vh]" : "py-1")}>
        {allSteps.map((step, i) => (
          <StepRow
            key={`step-${step.index}`}
            step={step}
            depth={depth}
            prevEpochMs={allPrev[i]}
            startMs={startMs}
            highlightedStepIndex={highlightedStepId}
            flashStepIndex={flashStepId}
            onRegisterRef={registerRef}
            onHoverChange={setHighlightedStepId}
            renderSubagent={renderSubagent}
          />
        ))}
      </div>
    </div>
  );

  // Only the top-level viewer installs the image-context provider — sub-
  // agent recursions inherit it.
  if (depth === 0 && imageContext) {
    return <TrajectoryImageProvider value={imageContext}>{body}</TrajectoryImageProvider>;
  }
  return body;
}

// ---------- StepDurationBar ----------

interface StepDurationBarProps {
  grouped: GroupedTrajectory;
  onStepClick?: (stepIndex: number) => void;
  onStepHover?: (stepIndex: number | null) => void;
  highlightedStepIndex?: number | null;
}

interface StepDurationInfo {
  stepIndex: number;
  stepId: number;
  durationMs: number;
  elapsedMs: number;
}

/**
 * Per-step duration bar — pier's original timeline. One segment per
 * ATIF step. Width ∝ each step's wall-clock distance from the previous
 * step. Colors oscillate through 4 neutral shades in a 1-2-3-4-3-2
 * period-of-6 pattern so adjacent segments are visually distinguishable
 * without tying color to step semantics.
 *
 * Hover: tooltip with step #, duration, and elapsed-from-start.
 * Click: scroll the step into view.
 */
function StepDurationBar({
  grouped,
  onStepClick,
  onStepHover,
  highlightedStepIndex,
}: StepDurationBarProps) {
  // Flatten preamble + turns back into trajectory order.
  const steps = useMemo<ViewStep[]>(() => {
    const out: ViewStep[] = [...grouped.preamble, ...grouped.turns];
    out.sort((a, b) => a.index - b.index);
    return out;
  }, [grouped]);

  const stepDurations = useMemo<StepDurationInfo[]>(() => {
    if (steps.length === 0) return [];
    const startMs = steps[0]!.epochMs ?? 0;
    return steps.map((s, idx) => {
      const t = s.epochMs ?? 0;
      const prevT = idx > 0 ? (steps[idx - 1]!.epochMs ?? t) : t;
      return {
        stepIndex: s.index,
        stepId: s.step.step_id,
        durationMs: Math.max(0, t - prevT),
        elapsedMs: Math.max(0, t - startMs),
      };
    });
  }, [steps]);

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [hoverPos, setHoverPos] = useState<number>(0);

  if (stepDurations.length === 0) return null;

  const totalMs = stepDurations.reduce((acc, s) => acc + s.durationMs, 0);
  if (totalMs === 0) {
    return (
      <div className="mb-4">
        <div className="h-6 bg-muted" />
      </div>
    );
  }

  const widths = stepDurations.map((s) => (s.durationMs / totalMs) * 100);
  const cumulative: number[] = [];
  let acc = 0;
  for (const w of widths) {
    cumulative.push(acc);
    acc += w;
  }

  const hovered = hoveredIdx != null ? stepDurations[hoveredIdx] : null;

  // External highlight (e.g. a step row hovered in the body) — find the
  // closest step entry.
  let externalIdx: number | null = null;
  if (highlightedStepIndex != null && hoveredIdx == null) {
    let best = -1;
    for (let i = 0; i < stepDurations.length; i++) {
      if (stepDurations[i]!.stepIndex <= highlightedStepIndex) best = i;
      else break;
    }
    externalIdx = best >= 0 ? best : null;
  }
  const activeIdx = hoveredIdx ?? externalIdx;

  return (
    <div className="mb-4">
      <div className="relative">
        {hovered && (
          <div
            className="pointer-events-none absolute bottom-full z-10 mb-2 -translate-x-1/2"
            style={{ left: `${hoverPos}%` }}
          >
            <div className="whitespace-nowrap rounded-md border border-border bg-popover px-3 py-2 shadow-md">
              <div className="text-sm font-medium">Step #{hovered.stepId}</div>
              <div className="text-sm text-muted-foreground">
                Duration: {formatMs(hovered.durationMs)}
              </div>
              <div className="text-sm text-muted-foreground">
                Started at: {formatMs(hovered.elapsedMs)}
              </div>
            </div>
          </div>
        )}
        <div className="flex h-6 overflow-hidden">
          {stepDurations.map((s, idx) => {
            const w = widths[idx]!;
            if (w <= 0) return null;
            const isActive = activeIdx === idx;
            const someoneActive = activeIdx != null;
            const center = cumulative[idx]! + w / 2;
            return (
              <div
                key={`${s.stepIndex}-${idx}`}
                role="button"
                tabIndex={0}
                aria-label={`Step ${s.stepId}: ${formatMs(s.durationMs)}`}
                title={`#${s.stepId} · ${formatMs(s.durationMs)}`}
                className="cursor-pointer transition-opacity duration-150"
                style={{
                  width: `${w}%`,
                  backgroundColor: oscillatingColor(idx),
                  opacity: someoneActive && !isActive ? 0.3 : 1,
                }}
                onMouseEnter={() => {
                  setHoveredIdx(idx);
                  setHoverPos(center);
                  onStepHover?.(s.stepIndex);
                }}
                onMouseLeave={() => {
                  setHoveredIdx(null);
                  onStepHover?.(null);
                }}
                onClick={() => onStepClick?.(s.stepIndex)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onStepClick?.(s.stepIndex);
                  }
                }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ---------- helpers ----------

function firstEpoch(traj: ViewTrajectory): number | undefined {
  for (const s of traj.steps) {
    if (typeof s.epochMs === "number") return s.epochMs;
  }
  return undefined;
}

function oscillatingColor(index: number): string {
  // Pier's original "1-2-3-4-3-2" period of 6.
  const colors = [
    "var(--color-neutral-400)",
    "var(--color-neutral-500)",
    "var(--color-neutral-600)",
    "var(--color-neutral-700)",
  ];
  const position = index % 6;
  const colorIndex = position <= 3 ? position : 6 - position;
  return colors[colorIndex]!;
}

function formatMs(ms: number): string {
  if (ms <= 0) return "0ms";
  if (ms < 1000) return `${ms}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

/** Inject the row-flash keyframe once per page. Used by `StepRow` when a
 *  step gets scrolled into view from the duration bar. */
function FlashStyles() {
  useEffect(() => {
    if (typeof document === "undefined") return;
    if (document.getElementById("tv-flash-styles")) return;
    const el = document.createElement("style");
    el.id = "tv-flash-styles";
    el.textContent = `
@keyframes tv-flash {
  0%   { background-color: color-mix(in oklab, var(--primary) 22%, transparent); }
  60%  { background-color: color-mix(in oklab, var(--primary) 10%, transparent); }
  100% { background-color: transparent; }
}
.tv-flash { animation: tv-flash 850ms ease-out; }
`;
    document.head.appendChild(el);
  }, []);
  return null;
}
