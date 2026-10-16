//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import * as React from "react";
import { ChevronDownIcon, InfoIcon } from "lucide-react";

import { Checkbox } from "~/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/lib/utils";

// ---------------------------------------------------------------------------
// Demo state — shared across variants so changes feel consistent.
// ---------------------------------------------------------------------------

export interface DemoState {
  metric: string;
  scale: string;
  trend: string;
  bins: string;
  line: string;
  band: boolean;
  raw: boolean;
  smooth: number;
}

export const DEFAULT_DEMO_STATE: DemoState = {
  metric: "agent_steps",
  scale: "log",
  trend: "binned",
  bins: "auto",
  line: "linear",
  band: true,
  raw: false,
  smooth: 50,
};

const METRIC_OPTIONS = [
  { value: "duration", label: "Wall clock time" },
  { value: "output_tokens", label: "Output tokens" },
  { value: "agent_steps", label: "Agent steps" },
  { value: "cost", label: "Cost" },
];

const SCALE_OPTIONS = [
  { value: "log", label: "Log scale" },
  { value: "linear", label: "Linear" },
];

const TREND_OPTIONS = [
  { value: "binned", label: "Binned" },
  { value: "local", label: "Local avg" },
  { value: "direct", label: "Direct" },
];

const BIN_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "8", label: "8" },
  { value: "12", label: "12" },
  { value: "16", label: "16" },
];

const LINE_OPTIONS = [
  { value: "linear", label: "Linear" },
  { value: "monotone", label: "Monotone" },
  { value: "spline", label: "Spline" },
];

export const SHORT_DESCRIPTION = "Performance vs task scale.";
export const LONG_DESCRIPTION =
  "Performance vs task scale. Each point is one (agent + model) measured on one dataset; the trend line is a binned average. Switch the X metric to see how reward changes with compute, tokens, or wall-clock time.";

// ---------------------------------------------------------------------------
// Existing pattern (control — for comparison)
// ---------------------------------------------------------------------------

export function VariantCurrent({
  description,
  state,
  setState,
}: {
  description: string;
  state: DemoState;
  setState: React.Dispatch<React.SetStateAction<DemoState>>;
}) {
  return (
    <div className="border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 text-xs text-muted-foreground">
        <span className="max-w-3xl">{description}</span>
        <div className="flex flex-wrap items-center gap-3">
          <InlinePair label="X metric">
            <BarelessSelect
              value={state.metric}
              onValueChange={(v) => setState((s) => ({ ...s, metric: v }))}
              options={METRIC_OPTIONS}
            />
          </InlinePair>
          <InlinePair label="X">
            <BarelessSelect
              value={state.scale}
              onValueChange={(v) => setState((s) => ({ ...s, scale: v }))}
              options={SCALE_OPTIONS}
            />
          </InlinePair>
          <InlinePair label="Trend">
            <BarelessSelect
              value={state.trend}
              onValueChange={(v) => setState((s) => ({ ...s, trend: v }))}
              options={TREND_OPTIONS}
            />
          </InlinePair>
          <InlinePair label="Bins">
            <BarelessSelect
              value={state.bins}
              onValueChange={(v) => setState((s) => ({ ...s, bins: v }))}
              options={BIN_OPTIONS}
            />
          </InlinePair>
          <InlinePair label="Line">
            <BarelessSelect
              value={state.line}
              onValueChange={(v) => setState((s) => ({ ...s, line: v }))}
              options={LINE_OPTIONS}
            />
          </InlinePair>
          <label className="flex items-center gap-2">
            <span>Smooth</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={state.smooth}
              onChange={(e) =>
                setState((s) => ({ ...s, smooth: Number(e.target.value) }))
              }
              className="w-24 accent-foreground"
            />
            <span className="w-8 text-right tabular-nums text-foreground">
              {state.smooth}%
            </span>
          </label>
          <label className="flex items-center gap-2">
            <Checkbox
              checked={state.band}
              onCheckedChange={(c) =>
                setState((s) => ({ ...s, band: c === true }))
              }
            />
            <span>Band</span>
          </label>
          <label className="flex items-center gap-2">
            <Checkbox
              checked={state.raw}
              onCheckedChange={(c) =>
                setState((s) => ({ ...s, raw: c === true }))
              }
            />
            <span>Raw points</span>
          </label>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Variant A — "Filter-bar style"
// Matches the Combobox filter row above: each control sits in its own
// card-styled cell with a left divider; description is on its own row so it
// can never push controls out of place.
// ---------------------------------------------------------------------------

export function VariantA({
  description,
  state,
  setState,
}: {
  description: string;
  state: DemoState;
  setState: React.Dispatch<React.SetStateAction<DemoState>>;
}) {
  return (
    <div className="border bg-card">
      <div className="border-b px-4 py-2.5 text-xs text-muted-foreground">
        {description}
      </div>
      <div className="flex flex-wrap">
        <ToolbarCell label="X metric">
          <CellSelect
            value={state.metric}
            onValueChange={(v) => setState((s) => ({ ...s, metric: v }))}
            options={METRIC_OPTIONS}
          />
        </ToolbarCell>
        <ToolbarCell label="X scale">
          <CellSelect
            value={state.scale}
            onValueChange={(v) => setState((s) => ({ ...s, scale: v }))}
            options={SCALE_OPTIONS}
          />
        </ToolbarCell>
        <ToolbarCell label="Trend">
          <CellSelect
            value={state.trend}
            onValueChange={(v) => setState((s) => ({ ...s, trend: v }))}
            options={TREND_OPTIONS}
          />
        </ToolbarCell>
        <ToolbarCell label="Bins">
          <CellSelect
            value={state.bins}
            onValueChange={(v) => setState((s) => ({ ...s, bins: v }))}
            options={BIN_OPTIONS}
          />
        </ToolbarCell>
        <ToolbarCell label="Line">
          <CellSelect
            value={state.line}
            onValueChange={(v) => setState((s) => ({ ...s, line: v }))}
            options={LINE_OPTIONS}
          />
        </ToolbarCell>
        <ToolbarCell label="Smooth">
          <div className="flex h-12 items-center gap-2 px-3">
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={state.smooth}
              onChange={(e) =>
                setState((s) => ({ ...s, smooth: Number(e.target.value) }))
              }
              className="w-20 accent-foreground"
            />
            <span className="w-8 text-right tabular-nums text-xs text-foreground">
              {state.smooth}%
            </span>
          </div>
        </ToolbarCell>
        <ToolbarToggleCell
          label="Band"
          checked={state.band}
          onCheckedChange={(c) => setState((s) => ({ ...s, band: c }))}
        />
        <ToolbarToggleCell
          label="Raw points"
          checked={state.raw}
          onCheckedChange={(c) => setState((s) => ({ ...s, raw: c }))}
        />
      </div>
    </div>
  );
}

function ToolbarCell({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-w-0 flex-col border-l border-border first:border-l-0">
      <span className="px-3 pt-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      {children}
    </div>
  );
}

function ToolbarToggleCell({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex h-[58px] cursor-pointer items-center gap-2 border-l border-border px-3 text-xs hover:bg-accent first:border-l-0">
      <Checkbox
        checked={checked}
        onCheckedChange={(c) => onCheckedChange(c === true)}
      />
      <span>{label}</span>
    </label>
  );
}

function CellSelect({
  value,
  onValueChange,
  options,
}: {
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        size="sm"
        className="h-9 border-0 bg-transparent px-3 text-xs text-foreground shadow-none hover:bg-accent focus-visible:ring-0 [&_svg]:opacity-60"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

// ---------------------------------------------------------------------------
// Variant B — "Bordered toolbar group"
// Description and toolbar are siblings; description is single-line truncated
// with a tooltip for the rest; the toolbar is a self-contained pill with
// vertical dividers between segments.
// ---------------------------------------------------------------------------

export function VariantB({
  description,
  state,
  setState,
}: {
  description: string;
  state: DemoState;
  setState: React.Dispatch<React.SetStateAction<DemoState>>;
}) {
  return (
    <div className="border bg-card">
      <div className="flex items-center gap-4 px-4 py-3">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
              {description}
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom" align="start" className="max-w-md">
            <p className="text-xs leading-relaxed">{description}</p>
          </TooltipContent>
        </Tooltip>
        <div className="flex shrink-0 items-stretch divide-x divide-border overflow-hidden rounded-md border border-border bg-background">
          <PillSelect
            label="X"
            value={state.metric}
            onValueChange={(v) => setState((s) => ({ ...s, metric: v }))}
            options={METRIC_OPTIONS}
          />
          <PillSelect
            label="Scale"
            value={state.scale}
            onValueChange={(v) => setState((s) => ({ ...s, scale: v }))}
            options={SCALE_OPTIONS}
          />
          <PillSelect
            label="Trend"
            value={state.trend}
            onValueChange={(v) => setState((s) => ({ ...s, trend: v }))}
            options={TREND_OPTIONS}
          />
          <PillSelect
            label="Bins"
            value={state.bins}
            onValueChange={(v) => setState((s) => ({ ...s, bins: v }))}
            options={BIN_OPTIONS}
          />
          <PillSelect
            label="Line"
            value={state.line}
            onValueChange={(v) => setState((s) => ({ ...s, line: v }))}
            options={LINE_OPTIONS}
          />
          <div className="flex items-center gap-2 px-3 text-xs text-muted-foreground">
            <span className="text-[10px] font-medium uppercase tracking-wider">
              Smooth
            </span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={state.smooth}
              onChange={(e) =>
                setState((s) => ({ ...s, smooth: Number(e.target.value) }))
              }
              className="w-20 accent-foreground"
            />
            <span className="w-7 text-right tabular-nums text-foreground">
              {state.smooth}%
            </span>
          </div>
          <PillToggle
            label="Band"
            checked={state.band}
            onCheckedChange={(c) => setState((s) => ({ ...s, band: c }))}
          />
          <PillToggle
            label="Raw"
            checked={state.raw}
            onCheckedChange={(c) => setState((s) => ({ ...s, raw: c }))}
          />
        </div>
      </div>
    </div>
  );
}

function PillSelect({
  label,
  value,
  onValueChange,
  options,
}: {
  label: string;
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        size="sm"
        className="h-9 rounded-none border-0 bg-transparent px-3 text-xs shadow-none hover:bg-accent focus-visible:ring-0 [&_svg]:opacity-60"
      >
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function PillToggle({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (c: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 px-3 text-xs hover:bg-accent">
      <Checkbox
        checked={checked}
        onCheckedChange={(c) => onCheckedChange(c === true)}
      />
      <span>{label}</span>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Variant C — "Description on top, chip-style controls"
// Two-row layout: description with info icon on top (clamped to one line,
// expandable with hover/click via tooltip); chip-shaped controls below with
// subtle borders. The chip baseline is the same regardless of description.
// ---------------------------------------------------------------------------

export function VariantC({
  description,
  state,
  setState,
}: {
  description: string;
  state: DemoState;
  setState: React.Dispatch<React.SetStateAction<DemoState>>;
}) {
  return (
    <div className="border bg-card">
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
        <ChipSelect
          label="X metric"
          value={state.metric}
          onValueChange={(v) => setState((s) => ({ ...s, metric: v }))}
          options={METRIC_OPTIONS}
        />
        <ChipSelect
          label="Scale"
          value={state.scale}
          onValueChange={(v) => setState((s) => ({ ...s, scale: v }))}
          options={SCALE_OPTIONS}
        />
        <ChipSelect
          label="Trend"
          value={state.trend}
          onValueChange={(v) => setState((s) => ({ ...s, trend: v }))}
          options={TREND_OPTIONS}
        />
        <ChipSelect
          label="Bins"
          value={state.bins}
          onValueChange={(v) => setState((s) => ({ ...s, bins: v }))}
          options={BIN_OPTIONS}
        />
        <ChipSelect
          label="Line"
          value={state.line}
          onValueChange={(v) => setState((s) => ({ ...s, line: v }))}
          options={LINE_OPTIONS}
        />
        <div className="inline-flex h-8 items-center gap-2 rounded-md border border-border bg-background px-3 text-xs">
          <span className="text-muted-foreground">Smooth</span>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={state.smooth}
            onChange={(e) =>
              setState((s) => ({ ...s, smooth: Number(e.target.value) }))
            }
            className="w-20 accent-foreground"
          />
          <span className="w-8 text-right tabular-nums">{state.smooth}%</span>
        </div>
        <ChipToggle
          label="Band"
          checked={state.band}
          onCheckedChange={(c) => setState((s) => ({ ...s, band: c }))}
        />
        <ChipToggle
          label="Raw points"
          checked={state.raw}
          onCheckedChange={(c) => setState((s) => ({ ...s, raw: c }))}
        />
      </div>
      <div className="flex items-start gap-2 px-4 py-2.5">
        <InfoIcon className="size-3.5 shrink-0 translate-y-0.5 text-muted-foreground" />
        <span className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">
          {description}
        </span>
      </div>
    </div>
  );
}

function ChipSelect({
  label,
  value,
  onValueChange,
  options,
}: {
  label: string;
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        size="sm"
        className="h-8 rounded-md border-border bg-background text-xs hover:bg-accent"
      >
        <span className="text-muted-foreground">{label}</span>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function ChipToggle({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (c: boolean) => void;
}) {
  return (
    <label className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-md border border-border bg-background px-3 text-xs hover:bg-accent">
      <Checkbox
        checked={checked}
        onCheckedChange={(c) => onCheckedChange(c === true)}
      />
      <span>{label}</span>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Variant D — "Two-column grid: description left, controls right"
// CSS grid with fixed description column so layout never reflows the
// controls under the description. Controls grouped in a single bordered
// container with internal dividers.
// ---------------------------------------------------------------------------

export function VariantD({
  description,
  state,
  setState,
}: {
  description: string;
  state: DemoState;
  setState: React.Dispatch<React.SetStateAction<DemoState>>;
}) {
  return (
    <div className="border bg-card">
      <div className="grid gap-3 px-4 py-3 lg:grid-cols-[minmax(0,320px)_1fr] lg:items-center">
        <span className="text-xs leading-relaxed text-muted-foreground">
          {description}
        </span>
        <div className="flex flex-wrap items-stretch justify-end gap-px overflow-hidden rounded-md border border-border bg-border">
          <GridSelect
            label="X metric"
            value={state.metric}
            onValueChange={(v) => setState((s) => ({ ...s, metric: v }))}
            options={METRIC_OPTIONS}
          />
          <GridSelect
            label="Scale"
            value={state.scale}
            onValueChange={(v) => setState((s) => ({ ...s, scale: v }))}
            options={SCALE_OPTIONS}
          />
          <GridSelect
            label="Trend"
            value={state.trend}
            onValueChange={(v) => setState((s) => ({ ...s, trend: v }))}
            options={TREND_OPTIONS}
          />
          <GridSelect
            label="Bins"
            value={state.bins}
            onValueChange={(v) => setState((s) => ({ ...s, bins: v }))}
            options={BIN_OPTIONS}
          />
          <GridSelect
            label="Line"
            value={state.line}
            onValueChange={(v) => setState((s) => ({ ...s, line: v }))}
            options={LINE_OPTIONS}
          />
          <div className="flex items-center gap-2 bg-background px-3 py-1.5 text-xs">
            <span className="text-muted-foreground">Smooth</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={state.smooth}
              onChange={(e) =>
                setState((s) => ({ ...s, smooth: Number(e.target.value) }))
              }
              className="w-20 accent-foreground"
            />
            <span className="w-8 text-right tabular-nums">{state.smooth}%</span>
          </div>
          <GridToggle
            label="Band"
            checked={state.band}
            onCheckedChange={(c) => setState((s) => ({ ...s, band: c }))}
          />
          <GridToggle
            label="Raw points"
            checked={state.raw}
            onCheckedChange={(c) => setState((s) => ({ ...s, raw: c }))}
          />
        </div>
      </div>
    </div>
  );
}

function GridSelect({
  label,
  value,
  onValueChange,
  options,
}: {
  label: string;
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        size="sm"
        className="h-auto rounded-none border-0 bg-background px-3 py-1.5 text-xs shadow-none hover:bg-accent focus-visible:ring-0 [&_svg]:opacity-60"
      >
        <span className="text-muted-foreground">{label}</span>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function GridToggle({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (c: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 bg-background px-3 py-1.5 text-xs hover:bg-accent">
      <Checkbox
        checked={checked}
        onCheckedChange={(c) => onCheckedChange(c === true)}
      />
      <span>{label}</span>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Shared "current pattern" helpers (kept here so the demo file is
// self-contained; not exported beyond VariantCurrent).
// ---------------------------------------------------------------------------

function InlinePair({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      <span>{label}</span>
      {children}
    </div>
  );
}

function BarelessSelect({
  value,
  onValueChange,
  options,
}: {
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        size="sm"
        className="h-7 border-0 bg-transparent px-2 text-xs text-foreground shadow-none hover:bg-accent focus-visible:ring-0"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

// A tiny visual placeholder for the chart body so each variant has a
// realistic body height under it.
export function MockChartBody({ height = 160 }: { height?: number }) {
  return (
    <div
      className="flex items-center justify-center border-t bg-muted/30 text-xs text-muted-foreground"
      style={{ height }}
    >
      <span>(chart body)</span>
    </div>
  );
}

// A small wrapper to render the combobox filter row above the chart so we
// can judge visual coherence with the surrounding UI.
export function MockComboboxFilterRow() {
  return (
    <div className="grid grid-cols-6 border bg-card">
      {[
        "Filter tasks…",
        "All agents",
        "All providers",
        "All models",
        "All datasets",
        "All tasks",
      ].map((label, i) => (
        <div
          key={label}
          className={cn(
            "flex h-12 items-center justify-between px-4 text-sm text-muted-foreground",
            i > 0 && "border-l"
          )}
        >
          <span>{label}</span>
          <ChevronDownIcon className="size-4 opacity-50" />
        </div>
      ))}
    </div>
  );
}

// Allow consumers to wrap a variant in a narrower container for testing
// reflow behavior.
export function WidthClamp({
  width,
  children,
}: {
  width: number | "full";
  children: React.ReactNode;
}) {
  const style = width === "full" ? undefined : { maxWidth: width };
  return (
    <div className="mx-auto" style={style}>
      {children}
    </div>
  );
}
