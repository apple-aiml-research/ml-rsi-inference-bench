//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * One turn rendered as a row.
 *
 * Layout: a 184px right-aligned gutter on the left (step #, cadence delta,
 * T+ elapsed) and the message body on the right (markdown, reasoning,
 * inline tool calls). The gutter is click-toggleable; expanding adds a
 * column of additional metadata (took, time, tokens, cost, model, kind)
 * and reveals reasoning text in the body.
 *
 * The body is capped at ~896px (1080 grid - 184 gutter) for comfortable
 * reading and centered within whatever container we live in.
 */
import { useState } from "react";
import { Brain, FileText, MessageCircle } from "lucide-react";
import { cn } from "~/lib/utils";
import type { ContentPart } from "~/lib/types";
import { Markdown } from "./Markdown";
import { ToolCallView } from "./ToolCallView";
import { DisclosureChevron } from "./icons";
import { TrajectoryImage } from "./TrajectoryImage";
import type { ResolvedToolResult, ViewStep } from "./types";

interface StepRowProps {
  step: ViewStep;
  /** When > 0 we're inside a sub-agent block — drop the gutter affordances
   *  to keep nested viewers visually quiet. */
  depth: number;
  highlightedStepIndex?: number | null;
  flashStepIndex?: number | null;
  /** Previous step's epoch — used for the "+Δt" cadence attributed to
   *  *this* turn. */
  prevEpochMs?: number;
  /** Wall-clock ms of the run's first step (for `T+` elapsed). */
  startMs?: number;
  onRegisterRef?: (stepIndex: number, el: HTMLDivElement | null) => void;
  onHoverChange?: (stepIndex: number | null) => void;
  renderSubagent?: (subagent: NonNullable<ResolvedToolResult["subagent"]>) => React.ReactNode;
}

export function StepRow({
  step,
  depth,
  highlightedStepIndex,
  flashStepIndex,
  prevEpochMs,
  startMs,
  onRegisterRef,
  onHoverChange,
  renderSubagent,
}: StepRowProps) {
  const isHighlighted = highlightedStepIndex === step.index;
  const isFlash = flashStepIndex === step.index;
  const reasoning =
    step.step.source === "agent" ? (step.step.reasoning_content ?? undefined) : undefined;
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      ref={(el) => onRegisterRef?.(step.index, el)}
      id={`step-${step.index}`}
      onMouseEnter={() => onHoverChange?.(step.index)}
      onMouseLeave={() => onHoverChange?.(null)}
      className={cn(
        "tv-step group/step relative w-full transition-colors",
        isHighlighted && "bg-muted/40",
        isFlash && "tv-flash",
      )}
    >
      <div
        className="mx-auto grid max-w-[1080px]"
        style={{ gridTemplateColumns: depth === 0 ? "184px 1fr" : "0 1fr" }}
      >
        <Gutter
          step={step}
          visible={depth === 0}
          prevEpochMs={prevEpochMs}
          startMs={startMs}
          expanded={expanded}
          onToggle={() => setExpanded((e) => !e)}
          hasReasoning={!!reasoning?.trim()}
        />
        {/* `min-w-0` lets the body column shrink below its content's natural
         *  width — without it, a wide inline tool-call code chip would push
         *  the whole grid past the page width and force horizontal scroll. */}
        <div className="min-w-0">
          <Body
            step={step}
            expandedReasoning={expanded ? reasoning : undefined}
            renderSubagent={renderSubagent}
          />
        </div>
      </div>
    </div>
  );
}

function Gutter({
  step,
  visible,
  prevEpochMs,
  startMs,
  expanded,
  onToggle,
  hasReasoning,
}: {
  step: ViewStep;
  visible: boolean;
  prevEpochMs?: number;
  startMs?: number;
  expanded: boolean;
  onToggle: () => void;
  hasReasoning: boolean;
}) {
  if (!visible) return <div />;

  const { step: atif, epochMs } = step;
  const stepNum = atif.step_id;
  const delta = deltaFromEpochs(prevEpochMs, epochMs);
  const time = atif.timestamp ? formatClock(atif.timestamp) : null;
  const elapsed = elapsedLabel(epochMs, startMs);
  const elapsedShort = elapsedShortLabel(epochMs, startMs);
  const tokens = tokenLine(atif.metrics);
  const cost = atif.metrics?.cost_usd ?? null;
  const model = compactModel(atif.model_name);
  return (
    <div className="relative px-3 pt-2 text-[10.5px] font-mono leading-[1.4] select-none">
      <button
        type="button"
        onClick={onToggle}
        className="w-full cursor-pointer text-muted-foreground transition-colors hover:text-foreground"
        aria-expanded={expanded}
        aria-label={expanded ? `Collapse step ${stepNum}` : `Expand step ${stepNum}`}
      >
        {/* Three rigid columns, right-aligned, tabular digits — so #N,
         *  delta, and T+ stack into clean vertical columns no matter how
         *  the values widen across rows. The brain icon (when this turn
         *  carried reasoning) lives in a narrow 4th column so it never
         *  shifts the numeric cols when present/absent.
         *
         *  `items-center` aligns the icon to the same vertical line as
         *  the digit glyphs (baseline alignment leaves the icon sitting
         *  at the bottom of its inline-flex box, looking dropped). */}
        <div className="grid grid-cols-[2.25rem_2.75rem_3.5rem_0.75rem] items-center gap-x-1 whitespace-nowrap tabular-nums text-right">
          <span className="text-foreground/80">#{stepNum}</span>
          <span>{delta ?? ""}</span>
          <span>{elapsedShort ?? ""}</span>
          <span className="inline-flex items-center justify-end">
            {hasReasoning && (
              <Brain
                size={11}
                strokeWidth={1.75}
                className={cn(
                  "shrink-0 transition-opacity",
                  expanded ? "opacity-100 text-foreground" : "opacity-60",
                )}
              />
            )}
          </span>
        </div>

        {expanded && (
          <div className="mt-1.5 flex flex-col gap-0.5 text-right">
            {elapsed && <Pop k="T+" v={elapsed} />}
            {time && <Pop k="time" v={time} />}
            {tokens && <Pop k="tokens" v={tokens} />}
            {cost != null && <Pop k="cost" v={`$${cost.toFixed(4)}`} />}
            {atif.model_name && (
              <Pop k="model" v={model ?? atif.model_name} title={atif.model_name} />
            )}
            <Pop k="kind" v={kindLabel(step)} />
          </div>
        )}
      </button>
    </div>
  );
}

/** Trim a model name to a short tag for the gutter — full name shown on
 *  hover and in the expanded metadata. We keep the family + size hint and
 *  drop org prefixes / dated suffixes that just take up space. */
function compactModel(name: string | null | undefined): string | null {
  if (!name) return null;
  let s = name;
  // Strip a "vendor/" prefix.
  const slash = s.lastIndexOf("/");
  if (slash >= 0) s = s.slice(slash + 1);
  // Drop a trailing "-YYYYMMDD" or "-YYYY-MM-DD" date stamp.
  s = s.replace(/-\d{4}(?:-\d{2}-\d{2}|\d{4})$/, "");
  // Cap.
  if (s.length > 22) s = s.slice(0, 21) + "…";
  return s;
}

function Pop({ k, v, title }: { k: string; v: string; title?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 break-words" title={title}>
      <span className="shrink-0 text-[10px] uppercase tracking-wider text-muted-foreground">
        {k}
      </span>
      <span className="text-right text-foreground">{v}</span>
    </div>
  );
}

function Body({
  step,
  expandedReasoning,
  renderSubagent,
}: {
  step: ViewStep;
  expandedReasoning?: string;
  renderSubagent?: (subagent: NonNullable<ResolvedToolResult["subagent"]>) => React.ReactNode;
}) {
  if (step.step.source === "user") return <UserStep step={step} />;
  if (step.step.source === "system") return <SystemStep step={step} />;
  return (
    <AssistantStep
      step={step}
      expandedReasoning={expandedReasoning}
      renderSubagent={renderSubagent}
    />
  );
}

function ContentParts({ parts }: { parts: ContentPart[] }) {
  return (
    <>
      {parts.map((part, idx) => {
        if (part.type === "text" && part.text) {
          return (
            <div key={idx} className="text-sm leading-relaxed">
              <Markdown text={part.text} />
            </div>
          );
        }
        if (part.type === "image" && part.source) {
          return <TrajectoryImage key={idx} path={part.source.path} />;
        }
        return null;
      })}
    </>
  );
}

function UserStep({ step }: { step: ViewStep }) {
  const hasContent = step.parts.length > 0;
  // No auto-collapse: even very long task prompts are interesting context
  // when reviewing a trajectory, and the page-level scroll handles bulk.
  return (
    <div className="flex min-w-0 flex-col items-end px-4 py-2">
      <div className="mb-1 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        <MessageCircle size={10} strokeWidth={1.75} />
        <span>user</span>
      </div>
      <div className="w-full max-w-[860px] min-w-0 rounded-md border bg-muted/40 px-3.5 py-2.5">
        {hasContent ? (
          <ContentParts parts={step.parts} />
        ) : (
          <span className="text-sm italic text-muted-foreground">(empty message)</span>
        )}
      </div>
    </div>
  );
}

function SystemStep({ step }: { step: ViewStep }) {
  // Expanded by default — system prompts are useful context for skimming
  // a trajectory. The block is height-capped and internally scrolled so
  // it doesn't dominate the page.
  const [open, setOpen] = useState(true);
  const text = step.parts
    .filter((p) => p.type === "text")
    .map((p) => p.text ?? "")
    .join("\n");
  if (!text) return null;
  return (
    <div className="min-w-0 px-4 py-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex cursor-pointer items-center gap-1.5 font-mono text-[11px] text-muted-foreground transition-colors hover:text-foreground"
      >
        <DisclosureChevron open={open} />
        <FileText size={11} strokeWidth={1.75} />
        <span className="uppercase tracking-wider">System prompt</span>
        <span className="text-muted-foreground/70">· {text.length.toLocaleString()} chars</span>
      </button>
      {open && (
        <pre className="ml-3.5 mt-2 max-h-[360px] overflow-auto whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
          {text}
        </pre>
      )}
    </div>
  );
}

/**
 * Observation results without a matching tool call on this step. ATIF
 * v1.7 allows results with `source_call_id: null` (system-initiated
 * operations), so render them as their own block beneath the inline
 * tool calls so no data is hidden.
 */
function OrphanResultsBlock({ results }: { results: ResolvedToolResult[] }) {
  if (results.length === 0) return null;
  return (
    <div className="mt-2 min-w-0">
      <div className="mb-1 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>tool result{results.length === 1 ? "" : "s"}</span>
      </div>
      <div className="space-y-1.5">
        {results.map((r, idx) => (
          <div key={idx} className="overflow-hidden rounded-md border bg-card min-w-0">
            <div className="flex items-center gap-2 border-b bg-muted/40 px-3 py-1.5 font-mono text-[10.5px] uppercase tracking-wider text-muted-foreground">
              <span>Observations</span>
              {r.text && (
                <span className="text-muted-foreground/80">
                  · {r.text.split("\n").length.toLocaleString()} lines
                </span>
              )}
              {r.isError && <span className="text-destructive">· error</span>}
            </div>
            {r.text && (
              <div className="max-h-[360px] overflow-auto whitespace-pre-wrap break-words px-3 py-2 font-mono text-xs leading-relaxed">
                {r.text}
              </div>
            )}
            {r.imageParts.length > 0 && (
              <div className="space-y-2 px-3 pb-3 pt-1">
                {r.imageParts.map((img, i) => (
                  <TrajectoryImage key={i} path={img.path} />
                ))}
              </div>
            )}
            {!r.text && r.imageParts.length === 0 && (
              <div className="px-3 py-2 text-xs italic text-muted-foreground">(empty)</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function AssistantStep({
  step,
  expandedReasoning,
  renderSubagent,
}: {
  step: ViewStep;
  expandedReasoning?: string;
  renderSubagent?: (subagent: NonNullable<ResolvedToolResult["subagent"]>) => React.ReactNode;
}) {
  const { toolCalls, resultsByCallId, orphanResults } = step;

  return (
    <div className="min-w-0 px-4 py-1">
      {expandedReasoning && <ReasoningBlock text={expandedReasoning} />}
      {step.parts.length > 0 && (
        <div className="flex min-w-0 flex-col gap-1.5 py-1 text-sm leading-relaxed">
          <ContentParts parts={step.parts} />
        </div>
      )}
      {toolCalls.length > 0 && (
        <div className="mt-1 min-w-0">
          {toolCalls.map((tc, i) => (
            <ToolCallView
              key={`${step.index}-${tc.call.tool_call_id}-${i}`}
              toolCall={tc}
              result={resultsByCallId.get(tc.call.tool_call_id)}
              isFirst={i === 0}
              isLast={i === toolCalls.length - 1}
              renderSubagent={renderSubagent}
            />
          ))}
        </div>
      )}
      <OrphanResultsBlock results={orphanResults} />
    </div>
  );
}

function ReasoningBlock({ text }: { text: string }) {
  return (
    <div className="my-2 overflow-hidden rounded-md border bg-muted/40 text-sm">
      <div className="flex items-center gap-1.5 border-b bg-muted/40 px-3 py-1.5">
        <Brain size={12} strokeWidth={1.75} className="text-muted-foreground" />
        <span className="font-mono text-[10.5px] uppercase tracking-wider text-muted-foreground">
          Reasoning · {text.length.toLocaleString()} chars
        </span>
      </div>
      <div className="px-3 py-2 text-muted-foreground">
        <Markdown text={text} />
      </div>
    </div>
  );
}

// ---------- formatting helpers ----------

function formatClock(iso: string): string | null {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString().slice(11, 19);
}

function shortNum(n: number): string {
  if (n < 1000) return String(n);
  if (n < 100_000) return `${(n / 1000).toFixed(1)}k`;
  return `${Math.round(n / 1000)}k`;
}

function deltaFromEpochs(prev?: number, curr?: number): string | null {
  if (prev == null || curr == null) return null;
  return formatDeltaMs(Math.max(0, curr - prev));
}

function formatDeltaMs(ms: number): string {
  if (ms < 1000) return ms === 0 ? "+0s" : `+${ms}ms`;
  if (ms < 10_000) return `+${(ms / 1000).toFixed(1)}s`;
  if (ms < 60_000) return `+${Math.round(ms / 1000)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return s ? `+${m}m${s}s` : `+${m}m`;
}

function elapsedLabel(epochMs?: number, startMs?: number): string | null {
  if (startMs == null || epochMs == null) return null;
  const total = Math.max(0, epochMs - startMs);
  const s = Math.floor(total / 1000) % 60;
  const m = Math.floor(total / 60_000) % 60;
  const h = Math.floor(total / 3_600_000);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Short elapsed since trajectory start, formatted for the single-line
 *  gutter — e.g. "23s", "6m 43s", "1h 02m". Different shape from
 *  `elapsedLabel` (M:SS) so the gutter reads at a glance instead of
 *  looking like a clock time. */
function elapsedShortLabel(epochMs?: number, startMs?: number): string | null {
  if (startMs == null || epochMs == null) return null;
  const total = Math.max(0, epochMs - startMs);
  if (total < 1000) return "0s";
  if (total < 60_000) return `${Math.floor(total / 1000)}s`;
  const totalMin = Math.floor(total / 60_000);
  const s = Math.floor((total % 60_000) / 1000);
  if (totalMin < 60) return s ? `${totalMin}m ${s}s` : `${totalMin}m`;
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return `${h}h ${String(m).padStart(2, "0")}m`;
}

function tokenLine(metrics: ViewStep["step"]["metrics"]): string | null {
  if (!metrics) return null;
  const parts: string[] = [];
  if (metrics.prompt_tokens != null) parts.push(`${shortNum(metrics.prompt_tokens)}↑`);
  if (metrics.completion_tokens != null) parts.push(`${shortNum(metrics.completion_tokens)}↓`);
  return parts.length ? parts.join(" ") : null;
}

function kindLabel(step: ViewStep): string {
  const source = step.step.source;
  if (source === "user") return "user";
  if (source === "system") return "system";
  if (step.toolCalls.length > 0) return `tool turn (${step.toolCalls.length})`;
  return "assistant";
}
