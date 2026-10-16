//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * Single tool call. Collapsed by default — one line summarizing what the
 * model did ("Read app/foo.tsx", "Ran git status"). Expanded shows the
 * full input arguments and the tool result (text + image attachments).
 *
 * When part of a vertical thread (multiple calls per turn) the rows
 * connect with a continuous left rail; the icon punches a hole in the
 * rail via a CSS mask so it stays visually clean.
 */
import { useState } from "react";
import { cn } from "~/lib/utils";
import { CodeBlock } from "~/components/ui/code-block";
import { DisclosureChevron, getToolIcon } from "./icons";
import { getToolCategory, getToolLabel } from "./tools";
import { TrajectoryImage } from "./TrajectoryImage";
import type { ResolvedToolCall, ResolvedToolResult } from "./types";

export interface ToolCallViewProps {
  toolCall: ResolvedToolCall;
  result?: ResolvedToolResult;
  isFirst?: boolean;
  isLast?: boolean;
  defaultOpen?: boolean;
  /** Render a sub-agent trajectory inline. The host supplies the recursive
   *  TrajectoryViewer (we can't import it here without creating a cycle). */
  renderSubagent?: (subagent: NonNullable<ResolvedToolResult["subagent"]>) => React.ReactNode;
}

export function ToolCallView({
  toolCall,
  result,
  isFirst = true,
  isLast = true,
  defaultOpen = false,
  renderSubagent,
}: ToolCallViewProps) {
  const [open, setOpen] = useState(defaultOpen);
  const onlyRow = isFirst && isLast;
  const category = getToolCategory(toolCall.call.function_name);
  const Icon = getToolIcon(category);
  const label = getToolLabel(toolCall);
  const resultText = result?.text ?? "";
  // Suppress the err badge when this call delegates to a sub-agent — the
  // sub-agent's own trajectory is shown inline on expand, where any actual
  // failure is already obvious.
  const isError = (result?.isError ?? false) && !result?.subagent;
  const lineCount = resultText ? resultText.split("\n").length : 0;
  const subagent = result?.subagent;

  // Thread mask geometry — the icon sits at row top + 12px (py-px on a
  // leading-5 row, so 1 + 10 = 11). Cut a 14px window around it so the
  // connector doesn't draw through the icon.
  const TOP_CUT = 5;
  const BOTTOM_CUT = 19;
  let maskImage: string | undefined;
  if (!onlyRow) {
    if (isFirst && !isLast) {
      maskImage = `linear-gradient(to bottom, transparent 0 ${BOTTOM_CUT}px, #000 ${BOTTOM_CUT}px 100%)`;
    } else if (isLast && !isFirst) {
      maskImage = `linear-gradient(to bottom, #000 0 ${TOP_CUT}px, transparent ${TOP_CUT}px 100%)`;
    } else {
      maskImage = `linear-gradient(to bottom, #000 0 ${TOP_CUT}px, transparent ${TOP_CUT}px ${BOTTOM_CUT}px, #000 ${BOTTOM_CUT}px 100%)`;
    }
  }

  return (
    // `min-w-0` is essential — without it, an oversized inline `code` token
    // in `label.code` (e.g. a 200-char shell command) would grow the
    // surrounding grid track and force horizontal scroll on the whole page,
    // even when the tool call is collapsed. With min-w-0 the truncate
    // class on the label can actually clip.
    <div className="relative min-w-0 pl-6 pr-1 py-[2px]">
      {!onlyRow && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute left-[10.5px] top-0 bottom-0 w-px bg-muted-foreground/35"
          style={{ maskImage, WebkitMaskImage: maskImage }}
        />
      )}

      {/* The whole row (including the icon overhang) is the click target,
       *  with a subtle hover wash so a lone tool call still reads as
       *  clickable affordance — without it a single-call row felt like an
       *  isolated label floating in the body column. */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={cn(
          "group flex w-full min-w-0 cursor-pointer items-center gap-2 rounded-sm py-px text-left text-sm leading-5 transition-colors",
          "hover:bg-muted/50",
          open ? "text-foreground" : "text-muted-foreground hover:text-foreground",
        )}
      >
        <span
          aria-hidden="true"
          className="relative z-10 -ml-5 flex shrink-0 items-center justify-center w-[14px] h-[14px] text-muted-foreground"
        >
          <Icon size={12} strokeWidth={1.75} />
        </span>
        <span className="min-w-0 flex-1 truncate">
          <span>{label.prefix}</span>
          {label.code && (
            <>
              {" "}
              <code className="rounded-sm bg-muted px-[3px] py-px font-mono text-[0.86em] text-foreground">
                {label.code}
              </code>
            </>
          )}
          {isError && (
            <>
              {" "}
              <span className="font-mono text-[10px] uppercase tracking-wider text-destructive">
                err
              </span>
            </>
          )}
          {subagent && (
            <>
              {" "}
              <span className="font-mono text-[10px] uppercase tracking-wider text-primary/70">
                sub-agent
              </span>
            </>
          )}
        </span>
        {/* Chevron is always present at low opacity so a single-call row
         *  shows the disclosure affordance even before hover. */}
        <span
          className={cn(
            "inline-flex h-3 w-3 shrink-0 items-center justify-center text-muted-foreground transition-opacity",
            open ? "opacity-70" : "opacity-30 group-hover:opacity-70",
          )}
          aria-hidden="true"
        >
          <DisclosureChevron open={open} />
        </span>
      </button>

      {open && (
        <div className="mt-1.5 mb-1 ml-[2px] min-w-0 space-y-1.5">
          <DetailBlock label={`Input · ${toolCall.call.function_name}`}>
            <CodeBlock code={toolCall.argsJson} lang="json" wrap />
          </DetailBlock>
          {result && (
            <DetailBlock
              label={
                <>
                  <span>Observations</span>
                  {lineCount > 0 && (
                    <span className="text-muted-foreground/80">
                      · {lineCount.toLocaleString()} {lineCount === 1 ? "line" : "lines"}
                    </span>
                  )}
                  {isError && <span className="text-destructive">· error</span>}
                </>
              }
            >
              {resultText && (
                <CodeBlock
                  code={resultText}
                  lang={category === "execute" ? "console" : "text"}
                  wrap
                />
              )}
              {result.imageParts.length > 0 && (
                <div className="space-y-2 px-3 pb-3 pt-1">
                  {result.imageParts.map((img, idx) => (
                    <TrajectoryImage key={idx} path={img.path} />
                  ))}
                </div>
              )}
              {!resultText && result.imageParts.length === 0 && (
                <div className="px-3 py-2 text-xs italic text-muted-foreground">(empty)</div>
              )}
            </DetailBlock>
          )}
          {subagent && renderSubagent && (
            <DetailBlock label={`Sub-agent · ${subagent.agent.displayName}`} cap={false}>
              <div className="border-t">{renderSubagent(subagent)}</div>
            </DetailBlock>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Default-capped scrollable region. Long observations or inputs (file
 * dumps, big JSON, multi-page test output) get an internal vertical
 * scroll instead of pushing every later turn off-screen. Click "expand"
 * to remove the cap when you actually want to read it all in flow.
 *
 * `cap={false}` opts out of the height cap — used for sub-agent
 * trajectories where the inner viewer is itself the layout.
 */
function DetailBlock({
  label,
  children,
  cap = true,
}: {
  label: React.ReactNode;
  children: React.ReactNode;
  cap?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const capped = cap && !expanded;
  return (
    <div className="overflow-hidden rounded-md border bg-card min-w-0">
      <div className="flex items-center gap-2 border-b bg-muted/40 px-3 py-1.5 font-mono text-[10.5px] uppercase tracking-wider text-muted-foreground">
        <span className="flex min-w-0 flex-1 items-center gap-2">{label}</span>
        {cap && (
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="shrink-0 cursor-pointer normal-case tracking-normal text-muted-foreground transition-colors hover:text-foreground"
          >
            {expanded ? "collapse" : "expand"}
          </button>
        )}
      </div>
      {/* Strip the inner CodeBlock's border + shadow — they would
       *  otherwise add 2px to the rendered height and trigger a spurious
       *  1px scroll inside the cap even when content fits. The DetailBlock
       *  itself already provides the visual containment. */}
      <div
        className={cn(
          "min-w-0 [&_figure]:!border-0 [&_figure]:!shadow-none",
          capped && "max-h-[360px] overflow-auto",
        )}
      >
        {children}
      </div>
    </div>
  );
}
