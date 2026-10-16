//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { useEffect, useMemo, useRef, useState } from "react";
import { BarChart3 } from "lucide-react";

import {
  ChartToolbar,
  ChartToolbarSelect,
  ChartToolbarToggle,
} from "~/components/ui/chart-toolbar";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty";
import { IndeterminateBar } from "~/components/ui/indeterminate-bar";
import type { CritiqueItemSummary } from "~/lib/types";
import { cn } from "~/lib/utils";

export type CritiqueDistributionXMode = "count" | "percent";
export type CritiqueDistributionYMode = "count" | "percent";
export type CritiqueDistributionTrialsFilter =
  | "all"
  | "non_errored"
  | "errored"
  | "successful";
export type CritiqueDistributionLayout = "overlay" | "grid";
export type CritiqueDistributionStyle = "bars" | "step";

const SUCCESSFUL_REWARD_THRESHOLD = 1;

// Categorical palette tuned for overlapping bars on light + dark backgrounds.
const DATASET_HUES = [220, 35, 145, 325, 270, 175, 95, 5, 245, 60];

function datasetColor(index: number, alpha = 1): string {
  const hue = DATASET_HUES[index % DATASET_HUES.length];
  if (alpha >= 1) return `oklch(0.65 0.17 ${hue})`;
  return `oklch(0.65 0.17 ${hue} / ${alpha})`;
}

function datasetEdgeColor(index: number): string {
  const hue = DATASET_HUES[index % DATASET_HUES.length];
  return `oklch(0.46 0.18 ${hue})`;
}

interface TaskStats {
  source: string;
  taskKey: string;
  taskName: string;
  nTotal: number;
  nBad: number;
  nGood: number;
  nUnrated: number;
  pctBad: number;
}

interface DatasetSeries {
  source: string;
  label: string;
  colorIndex: number;
  tasks: TaskStats[];
  nTasksWithData: number;
  totalTrials: number;
  totalBad: number;
  meanBadCount: number;
  meanBadPct: number;
  medianBadCount: number;
  medianBadPct: number;
}

interface Bin {
  label: string;
  shortLabel: string;
  lo: number;
  hi: number;
  isLast: boolean;
}

interface SeriesBin {
  bin: Bin;
  binIndex: number;
  series: DatasetSeries;
  count: number;
  percent: number;
  tasks: TaskStats[];
}

function passesTrialFilter(
  item: CritiqueItemSummary,
  filter: CritiqueDistributionTrialsFilter
): boolean {
  if (filter === "non_errored") return !item.source_error_type;
  if (filter === "errored") return !!item.source_error_type;
  if (filter === "successful") {
    if (item.source_error_type) return false;
    if (item.source_reward === null) return false;
    return item.source_reward >= SUCCESSFUL_REWARD_THRESHOLD;
  }
  return true;
}

function computeSeries(
  items: CritiqueItemSummary[],
  filter: CritiqueDistributionTrialsFilter
): DatasetSeries[] {
  type Acc = Map<string, TaskStats>;
  const byDataset = new Map<string, { source: string; tasks: Acc }>();
  for (const item of items) {
    if (!passesTrialFilter(item, filter)) continue;
    const source = item.source ?? "(no dataset)";
    const taskLabel = item.task_name ?? "(unknown task)";
    const taskKey = `${source}::${taskLabel}`;
    const bucket =
      byDataset.get(source) ?? { source, tasks: new Map<string, TaskStats>() };
    const existing =
      bucket.tasks.get(taskKey) ?? {
        source,
        taskKey,
        taskName: taskLabel,
        nTotal: 0,
        nBad: 0,
        nGood: 0,
        nUnrated: 0,
        pctBad: 0,
      };
    existing.nTotal += 1;
    if (item.rating === "bad") existing.nBad += 1;
    else if (item.rating === "good") existing.nGood += 1;
    else existing.nUnrated += 1;
    bucket.tasks.set(taskKey, existing);
    byDataset.set(source, bucket);
  }

  const sources = [...byDataset.keys()].sort((a, b) => a.localeCompare(b));
  return sources.map((source, colorIndex) => {
    const bucket = byDataset.get(source)!;
    const tasks = [...bucket.tasks.values()];
    for (const task of tasks) {
      task.pctBad = task.nTotal > 0 ? task.nBad / task.nTotal : 0;
    }
    tasks.sort((a, b) => a.taskName.localeCompare(b.taskName));
    const sortedBadCount = [...tasks.map((t) => t.nBad)].sort((a, b) => a - b);
    const sortedBadPct = [...tasks.map((t) => t.pctBad)].sort((a, b) => a - b);
    const median = (arr: number[]) => {
      if (arr.length === 0) return 0;
      const m = Math.floor(arr.length / 2);
      return arr.length % 2 === 0 ? (arr[m - 1] + arr[m]) / 2 : arr[m];
    };
    const totalTrials = tasks.reduce((sum, t) => sum + t.nTotal, 0);
    const totalBad = tasks.reduce((sum, t) => sum + t.nBad, 0);
    return {
      source,
      label: source,
      colorIndex,
      tasks,
      nTasksWithData: tasks.length,
      totalTrials,
      totalBad,
      meanBadCount:
        tasks.length > 0 ? totalBad / tasks.length : 0,
      meanBadPct:
        tasks.length > 0
          ? tasks.reduce((sum, t) => sum + t.pctBad, 0) / tasks.length
          : 0,
      medianBadCount: median(sortedBadCount),
      medianBadPct: median(sortedBadPct),
    };
  });
}

function buildIntegerBins(maxValue: number): Bin[] {
  const bins: Bin[] = [];
  const upper = Math.max(0, Math.floor(maxValue));
  for (let i = 0; i <= upper; i += 1) {
    bins.push({
      label: i.toString(),
      shortLabel: i.toString(),
      lo: i - 0.5,
      hi: i + 0.5,
      isLast: i === upper,
    });
  }
  if (bins.length === 0) {
    bins.push({ label: "0", shortLabel: "0", lo: -0.5, hi: 0.5, isLast: true });
  }
  return bins;
}

function buildPercentBins(count: number): Bin[] {
  const bins: Bin[] = [];
  const step = 1 / count;
  for (let i = 0; i < count; i += 1) {
    const lo = i * step;
    const hi = (i + 1) * step;
    const formatPct = (v: number) => `${Math.round(v * 100)}%`;
    bins.push({
      label: `${formatPct(lo)}–${formatPct(hi)}`,
      shortLabel: formatPct(lo),
      lo,
      hi,
      isLast: i === count - 1,
    });
  }
  return bins;
}

function assignTaskToBin(value: number, bins: Bin[]): number {
  for (let i = 0; i < bins.length; i += 1) {
    const bin = bins[i];
    if (value >= bin.lo && (value < bin.hi || (bin.isLast && value <= bin.hi))) {
      return i;
    }
  }
  return -1;
}

function buildSeriesBins(
  series: DatasetSeries[],
  bins: Bin[],
  xMode: CritiqueDistributionXMode
): Map<string, SeriesBin[]> {
  const result = new Map<string, SeriesBin[]>();
  for (const dataset of series) {
    const datasetBins: SeriesBin[] = bins.map((bin, binIndex) => ({
      bin,
      binIndex,
      series: dataset,
      count: 0,
      percent: 0,
      tasks: [],
    }));
    for (const task of dataset.tasks) {
      const value = xMode === "count" ? task.nBad : task.pctBad;
      const idx = assignTaskToBin(value, bins);
      if (idx < 0) continue;
      datasetBins[idx].count += 1;
      datasetBins[idx].tasks.push(task);
    }
    const total = dataset.tasks.length;
    if (total > 0) {
      for (const sb of datasetBins) {
        sb.percent = sb.count / total;
      }
    }
    result.set(dataset.source, datasetBins);
  }
  return result;
}

function maxBadCount(series: DatasetSeries[]): number {
  let m = 0;
  for (const dataset of series) {
    for (const task of dataset.tasks) {
      if (task.nBad > m) m = task.nBad;
    }
  }
  return m;
}

function formatPercent(value: number): string {
  if (value === 0) return "0%";
  if (value < 0.01) return "<1%";
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function formatTrialFilterShort(filter: CritiqueDistributionTrialsFilter): string {
  if (filter === "non_errored") return "non-errored";
  if (filter === "errored") return "errored";
  if (filter === "successful") return "successful";
  return "all";
}

const TRIALS_FILTER_LABELS: Record<CritiqueDistributionTrialsFilter, string> = {
  all: "All trials",
  non_errored: "Exclude errored",
  errored: "Only errored",
  successful: "Only successful",
};

const X_MODE_LABELS: Record<CritiqueDistributionXMode, string> = {
  count: "# bad ratings",
  percent: "% bad ratings",
};

const Y_MODE_LABELS: Record<CritiqueDistributionYMode, string> = {
  count: "# tasks",
  percent: "% of tasks",
};

// ---------------------------------------------------------------------------
// Chart geometry
// ---------------------------------------------------------------------------

interface ChartDims {
  width: number;
  height: number;
  marginTop: number;
  marginRight: number;
  marginBottom: number;
  marginLeft: number;
  plotW: number;
  plotH: number;
}

function chartDims(width: number, height: number, compact: boolean): ChartDims {
  const marginTop = compact ? 22 : 32;
  const marginRight = compact ? 18 : 32;
  const marginBottom = compact ? 44 : 56;
  const marginLeft = compact ? 44 : 56;
  return {
    width,
    height,
    marginTop,
    marginRight,
    marginBottom,
    marginLeft,
    plotW: width - marginLeft - marginRight,
    plotH: height - marginTop - marginBottom,
  };
}

function yAxisTicks(maxValue: number, yMode: CritiqueDistributionYMode): number[] {
  if (maxValue <= 0) return [0];
  if (yMode === "percent") {
    const candidates = [0.1, 0.2, 0.25, 0.5, 1];
    for (const step of candidates) {
      const n = Math.ceil(maxValue / step);
      if (n <= 6) {
        return Array.from({ length: n + 1 }, (_, i) => i * step);
      }
    }
    return [0, 0.25, 0.5, 0.75, 1];
  }
  const niceStep = (() => {
    const target = maxValue / 5;
    const pow = 10 ** Math.floor(Math.log10(target));
    const candidates = [1, 2, 2.5, 5, 10].map((m) => m * pow);
    for (const c of candidates) {
      if (c >= target) return c;
    }
    return candidates[candidates.length - 1];
  })();
  const ticks: number[] = [];
  for (let v = 0; v <= maxValue + niceStep * 1e-6; v += niceStep) {
    ticks.push(v);
  }
  if (ticks[ticks.length - 1] < maxValue) {
    ticks.push(ticks[ticks.length - 1] + niceStep);
  }
  return ticks;
}

function formatYTickValue(value: number, yMode: CritiqueDistributionYMode): string {
  if (yMode === "percent") {
    return `${Math.round(value * 100)}%`;
  }
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return value.toFixed(value < 1 ? 1 : 0);
}

function formatXTickValue(
  bin: Bin,
  xMode: CritiqueDistributionXMode,
  compact: boolean
): string {
  if (xMode === "count") return bin.shortLabel;
  return compact ? `${Math.round(bin.lo * 100)}` : bin.shortLabel;
}

// ---------------------------------------------------------------------------
// HistogramChart: renders a single set of axes with N overlaid series.
// ---------------------------------------------------------------------------

interface HoveredBin {
  binIndex: number;
  seriesSource: string;
  cx: number;
  cy: number;
  count: number;
  percent: number;
  bin: Bin;
  series: DatasetSeries;
  tasks: TaskStats[];
}

interface HistogramChartProps {
  series: DatasetSeries[];
  bins: Bin[];
  binData: Map<string, SeriesBin[]>;
  xMode: CritiqueDistributionXMode;
  yMode: CritiqueDistributionYMode;
  style: CritiqueDistributionStyle;
  layout: CritiqueDistributionLayout;
  width: number;
  height: number;
  hiddenSources: ReadonlySet<string>;
  hoveredSource: string | null;
  setHoveredSource: (source: string | null) => void;
  showMeans: boolean;
  /** When set, this chart only draws this one series (grid mode). */
  isolateSource?: string;
}

function HistogramChart({
  series,
  bins,
  binData,
  xMode,
  yMode,
  style,
  layout,
  width,
  height,
  hiddenSources,
  hoveredSource,
  setHoveredSource,
  showMeans,
  isolateSource,
}: HistogramChartProps) {
  const compact = isolateSource != null;
  const dims = chartDims(width, height, compact);
  const [hovered, setHoveredState] = useState<HoveredBin | null>(null);
  const hideTimerRef = useRef<number | null>(null);

  const cancelHide = () => {
    if (hideTimerRef.current !== null) {
      window.clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
  };
  const setHovered = (next: HoveredBin | null) => {
    cancelHide();
    setHoveredState(next);
  };
  const scheduleHide = () => {
    cancelHide();
    hideTimerRef.current = window.setTimeout(() => {
      setHoveredState(null);
      hideTimerRef.current = null;
    }, 180);
  };
  useEffect(() => () => cancelHide(), []);

  const visibleSeries = useMemo(
    () =>
      series.filter(
        (s) =>
          (!isolateSource || s.source === isolateSource) &&
          !hiddenSources.has(s.source)
      ),
    [series, hiddenSources, isolateSource]
  );

  const maxY = useMemo(() => {
    let m = 0;
    for (const s of visibleSeries) {
      const seriesBins = binData.get(s.source) ?? [];
      for (const sb of seriesBins) {
        const value = yMode === "count" ? sb.count : sb.percent;
        if (value > m) m = value;
      }
    }
    if (yMode === "percent") return Math.min(1, Math.max(m, 0.1));
    return Math.max(m, 1);
  }, [visibleSeries, binData, yMode]);

  const yTicks = yAxisTicks(maxY, yMode);
  const yDomainMax = yTicks[yTicks.length - 1] || maxY;

  const binWidth = dims.plotW / bins.length;
  const xForBin = (binIndex: number) =>
    dims.marginLeft + binIndex * binWidth;
  const yForValue = (value: number) =>
    dims.marginTop + (1 - value / yDomainMax) * dims.plotH;

  // For "bars" style we draw overlapping rectangles with transparency.
  // In overlay mode with multiple visible series we slightly inset each
  // series so smaller bars peek out from larger ones.
  const insetAmount = layout === "overlay" && visibleSeries.length > 1 ? 2 : 0;

  // Decide tick spacing: skip labels when there's too many bins to draw.
  const labelEvery = (() => {
    const maxLabels = compact ? 8 : 12;
    if (bins.length <= maxLabels) return 1;
    return Math.ceil(bins.length / maxLabels);
  })();

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${dims.width} ${dims.height}`}
        width="100%"
        height={dims.height}
        preserveAspectRatio="xMidYMid meet"
        style={{ display: "block" }}
        role="img"
        aria-label={`Histogram of ${X_MODE_LABELS[xMode]}`}
        onMouseLeave={scheduleHide}
      >
        {/* Plot frame */}
        <rect
          x={dims.marginLeft}
          y={dims.marginTop}
          width={dims.plotW}
          height={dims.plotH}
          fill="transparent"
          stroke="var(--border)"
          strokeWidth={1}
        />

        {/* Y gridlines + labels */}
        {yTicks.map((tick) => {
          const y = yForValue(tick);
          return (
            <g key={`y-${tick}`}>
              <line
                x1={dims.marginLeft}
                x2={dims.marginLeft + dims.plotW}
                y1={y}
                y2={y}
                stroke="var(--border)"
                strokeDasharray="2 4"
                opacity={0.4}
              />
              <text
                x={dims.marginLeft - 8}
                y={y + 4}
                textAnchor="end"
                fontSize={compact ? 10 : 11}
                className="fill-muted-foreground tabular-nums"
              >
                {formatYTickValue(tick, yMode)}
              </text>
            </g>
          );
        })}

        {/* X axis labels */}
        {bins.map((bin, i) => {
          if (i % labelEvery !== 0 && i !== bins.length - 1) return null;
          const x = xForBin(i) + binWidth / 2;
          return (
            <text
              key={`x-${i}`}
              x={x}
              y={dims.marginTop + dims.plotH + (compact ? 16 : 20)}
              textAnchor="middle"
              fontSize={compact ? 10 : 11}
              className="fill-muted-foreground tabular-nums"
            >
              {formatXTickValue(bin, xMode, compact)}
            </text>
          );
        })}
        {xMode === "percent" && (
          <text
            x={dims.marginLeft + dims.plotW}
            y={dims.marginTop + dims.plotH + (compact ? 16 : 20)}
            textAnchor="end"
            fontSize={compact ? 10 : 11}
            className="fill-muted-foreground tabular-nums"
          >
            100%
          </text>
        )}

        {/* Bars / step lines */}
        {[...visibleSeries]
          .sort((a, b) => {
            // Hovered series last so it draws on top.
            if (hoveredSource === a.source) return 1;
            if (hoveredSource === b.source) return -1;
            // Smaller bars drawn after bigger so they peek through.
            const aMax = Math.max(
              ...(binData.get(a.source) ?? []).map((sb) =>
                yMode === "count" ? sb.count : sb.percent
              ),
              0
            );
            const bMax = Math.max(
              ...(binData.get(b.source) ?? []).map((sb) =>
                yMode === "count" ? sb.count : sb.percent
              ),
              0
            );
            return bMax - aMax;
          })
          .map((s) => {
            const seriesBins = binData.get(s.source) ?? [];
            const isHovered = hoveredSource === s.source;
            const isDim = hoveredSource !== null && !isHovered;
            const fillAlpha =
              layout === "overlay" && visibleSeries.length > 1
                ? isHovered
                  ? 0.85
                  : 0.55
                : 0.85;
            const fill = datasetColor(s.colorIndex, fillAlpha);
            const stroke = datasetEdgeColor(s.colorIndex);

            if (style === "step") {
              const strokeSegs: string[] = [];
              const fillSegs: string[] = [];
              for (let i = 0; i < seriesBins.length; i += 1) {
                const sb = seriesBins[i];
                const v = yMode === "count" ? sb.count : sb.percent;
                const x1 = xForBin(i);
                const x2 = xForBin(i) + binWidth;
                const y = yForValue(v);
                if (i === 0) {
                  strokeSegs.push(`M ${x1} ${y}`);
                  fillSegs.push(`M ${x1} ${yForValue(0)}`);
                  fillSegs.push(`L ${x1} ${y}`);
                } else {
                  strokeSegs.push(`L ${x1} ${y}`);
                  fillSegs.push(`L ${x1} ${y}`);
                }
                strokeSegs.push(`L ${x2} ${y}`);
                fillSegs.push(`L ${x2} ${y}`);
              }
              const lastX = xForBin(seriesBins.length - 1) + binWidth;
              fillSegs.push(`L ${lastX} ${yForValue(0)} Z`);
              return (
                <g
                  key={`series-${s.source}`}
                  style={{ opacity: isDim ? 0.18 : 1 }}
                  onMouseEnter={() => setHoveredSource(s.source)}
                >
                  <path
                    d={fillSegs.join(" ")}
                    fill={fill}
                    fillOpacity={layout === "overlay" ? 0.18 : 0.28}
                    stroke="none"
                  />
                  <path
                    d={strokeSegs.join(" ")}
                    fill="none"
                    stroke={stroke}
                    strokeWidth={isHovered ? 2.4 : 1.8}
                    strokeLinejoin="miter"
                    strokeLinecap="square"
                  />
                  {/* Invisible hit areas for hover tooltip in step mode. */}
                  {seriesBins.map((sb, i) => {
                    const v = yMode === "count" ? sb.count : sb.percent;
                    const x = xForBin(i);
                    const y = yForValue(v);
                    return (
                      <rect
                        key={`step-hit-${s.source}-${i}`}
                        x={x}
                        y={dims.marginTop}
                        width={binWidth}
                        height={dims.plotH}
                        fill="transparent"
                        pointerEvents="all"
                        onMouseEnter={() => {
                          setHoveredSource(s.source);
                          setHovered({
                            binIndex: i,
                            seriesSource: s.source,
                            cx: xForBin(i) + binWidth / 2,
                            cy: y,
                            count: sb.count,
                            percent: sb.percent,
                            bin: sb.bin,
                            series: s,
                            tasks: sb.tasks,
                          });
                        }}
                        onMouseLeave={scheduleHide}
                      />
                    );
                  })}
                </g>
              );
            }

            return (
              <g
                key={`series-${s.source}`}
                style={{ opacity: isDim ? 0.22 : 1 }}
                onMouseEnter={() => setHoveredSource(s.source)}
              >
                {seriesBins.map((sb, i) => {
                  const v = yMode === "count" ? sb.count : sb.percent;
                  if (v <= 0) return null;
                  const x = xForBin(i) + insetAmount;
                  const y = yForValue(v);
                  const w = Math.max(binWidth - insetAmount * 2, 1);
                  const h = dims.marginTop + dims.plotH - y;
                  const isHoveredBin =
                    hovered?.binIndex === i &&
                    hovered.seriesSource === s.source;
                  return (
                    <g key={`bar-${s.source}-${i}`}>
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={isHoveredBin ? 1.5 : 0.8}
                        strokeOpacity={isHoveredBin ? 1 : 0.7}
                        shapeRendering={
                          binWidth < 8 ? "crispEdges" : "geometricPrecision"
                        }
                      />
                      {/* Bigger invisible hover target. */}
                      <rect
                        x={xForBin(i)}
                        y={dims.marginTop}
                        width={binWidth}
                        height={dims.plotH}
                        fill="transparent"
                        pointerEvents="all"
                        onMouseEnter={() => {
                          setHoveredSource(s.source);
                          setHovered({
                            binIndex: i,
                            seriesSource: s.source,
                            cx: xForBin(i) + binWidth / 2,
                            cy: y,
                            count: sb.count,
                            percent: sb.percent,
                            bin: sb.bin,
                            series: s,
                            tasks: sb.tasks,
                          });
                        }}
                        onMouseLeave={scheduleHide}
                      />
                    </g>
                  );
                })}
              </g>
            );
          })}

        {/* Mean reference lines */}
        {showMeans &&
          visibleSeries.map((s) => {
            const meanValue =
              xMode === "count" ? s.meanBadCount : s.meanBadPct;
            // Find which bin index the mean falls into to map x position.
            const xPos = (() => {
              if (xMode === "count") {
                // Integer bins centered on integers.
                const offset = meanValue - Math.floor(meanValue);
                return (
                  xForBin(Math.floor(meanValue)) +
                  binWidth * 0.5 +
                  offset * binWidth
                );
              }
              // Percent bins.
              const total = bins[bins.length - 1].hi - bins[0].lo;
              const fraction = (meanValue - bins[0].lo) / total;
              return dims.marginLeft + fraction * dims.plotW;
            })();
            const isHovered = hoveredSource === s.source;
            const isDim = hoveredSource !== null && !isHovered;
            return (
              <g
                key={`mean-${s.source}`}
                style={{ opacity: isDim ? 0.18 : isHovered ? 1 : 0.7 }}
                pointerEvents="none"
              >
                <line
                  x1={xPos}
                  y1={dims.marginTop}
                  x2={xPos}
                  y2={dims.marginTop + dims.plotH}
                  stroke={datasetEdgeColor(s.colorIndex)}
                  strokeWidth={isHovered ? 2 : 1.2}
                  strokeDasharray="4 3"
                />
                <text
                  x={xPos}
                  y={dims.marginTop - 4}
                  textAnchor="middle"
                  fontSize={compact ? 9 : 10}
                  fill={datasetEdgeColor(s.colorIndex)}
                  className="font-mono tabular-nums"
                  style={{
                    paintOrder: "stroke",
                    stroke: "var(--background)",
                    strokeWidth: 3,
                    strokeLinejoin: "round",
                  }}
                >
                  μ{xMode === "percent"
                    ? formatPercent(meanValue)
                    : meanValue.toFixed(1)}
                </text>
              </g>
            );
          })}

        {/* Hovered marker */}
        {hovered && (
          <line
            x1={hovered.cx}
            y1={dims.marginTop}
            x2={hovered.cx}
            y2={dims.marginTop + dims.plotH}
            stroke="var(--foreground)"
            strokeDasharray="2 3"
            opacity={0.3}
            pointerEvents="none"
          />
        )}

        {/* Axis labels */}
        <text
          x={dims.marginLeft + dims.plotW / 2}
          y={dims.height - (compact ? 6 : 8)}
          textAnchor="middle"
          fontSize={compact ? 10 : 12}
          className="fill-foreground"
        >
          {X_MODE_LABELS[xMode]}
        </text>
        <text
          x={compact ? 12 : 16}
          y={dims.marginTop + dims.plotH / 2}
          textAnchor="middle"
          fontSize={compact ? 10 : 12}
          transform={`rotate(-90 ${compact ? 12 : 16} ${dims.marginTop + dims.plotH / 2})`}
          className="fill-foreground"
        >
          {Y_MODE_LABELS[yMode]}
        </text>

        {/* In grid mode the series title sits above the chart. */}
        {isolateSource && visibleSeries.length > 0 && (
          <text
            x={dims.marginLeft}
            y={dims.marginTop - 8}
            fontSize={11}
            className="fill-foreground"
          >
            <tspan
              fontFamily="var(--font-mono, ui-monospace)"
              fontSize={11}
              style={{ fill: datasetEdgeColor(visibleSeries[0].colorIndex) }}
            >
              ■{" "}
            </tspan>
            <tspan className="fill-foreground">{visibleSeries[0].label}</tspan>
          </text>
        )}
      </svg>

      {hovered && (
        <Tooltip
          dims={dims}
          hovered={hovered}
          yMode={yMode}
          xMode={xMode}
          onMouseEnter={cancelHide}
          onMouseLeave={scheduleHide}
        />
      )}
    </div>
  );
}

function Tooltip({
  dims,
  hovered,
  yMode,
  xMode,
  onMouseEnter,
  onMouseLeave,
}: {
  dims: ChartDims;
  hovered: HoveredBin;
  yMode: CritiqueDistributionYMode;
  xMode: CritiqueDistributionXMode;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}) {
  // Anchor on either side of the bar so it doesn't fall off the chart.
  const placeRight = hovered.cx < dims.marginLeft + dims.plotW / 2;
  const left = `${(hovered.cx / dims.width) * 100}%`;
  const top = `${(hovered.cy / dims.height) * 100}%`;
  const transform = placeRight
    ? "translate(12px, -8px)"
    : "translate(calc(-100% - 12px), -8px)";
  const previewTasks = hovered.tasks.slice(0, 4);
  const moreTasks = hovered.tasks.length - previewTasks.length;

  return (
    <div
      className="absolute z-10 min-w-[16rem] rounded-md border bg-popover text-popover-foreground shadow-md px-3 py-2 text-xs select-text cursor-default"
      style={{ left, top, transform }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block size-2.5 rounded-sm"
            style={{ background: datasetColor(hovered.series.colorIndex) }}
          />
          <span className="font-medium">{hovered.series.label}</span>
        </span>
        <span className="text-muted-foreground tabular-nums">
          {xMode === "count" ? `${hovered.bin.label} bad` : hovered.bin.label}
        </span>
      </div>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
        <span className="text-muted-foreground">Tasks</span>
        <span className="font-mono tabular-nums">
          {hovered.count}
          <span className="text-muted-foreground">
            {" "}
            ({formatPercent(hovered.percent)} of {hovered.series.tasks.length})
          </span>
        </span>
        <span className="text-muted-foreground">Y value</span>
        <span className="font-mono tabular-nums">
          {yMode === "count"
            ? hovered.count.toString()
            : formatPercent(hovered.percent)}
        </span>
      </div>
      {previewTasks.length > 0 && (
        <div className="mt-2 border-t pt-1.5">
          <div className="text-muted-foreground mb-1">Sample tasks</div>
          <div className="space-y-0.5">
            {previewTasks.map((task) => (
              <div
                key={task.taskKey}
                className="flex justify-between gap-2 font-mono text-[10.5px]"
              >
                <span className="truncate">{task.taskName}</span>
                <span className="text-muted-foreground tabular-nums shrink-0">
                  {task.nBad}/{task.nTotal}
                </span>
              </div>
            ))}
            {moreTasks > 0 && (
              <div className="text-muted-foreground text-[10.5px]">
                +{moreTasks} more
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Top-level CritiqueDistributionChart component.
// ---------------------------------------------------------------------------

export interface CritiqueDistributionChartProps {
  items: CritiqueItemSummary[] | undefined;
  isLoading?: boolean;
  isFetching?: boolean;

  xMode: CritiqueDistributionXMode;
  setXMode: (mode: CritiqueDistributionXMode) => void;
  yMode: CritiqueDistributionYMode;
  setYMode: (mode: CritiqueDistributionYMode) => void;
  trialsFilter: CritiqueDistributionTrialsFilter;
  setTrialsFilter: (filter: CritiqueDistributionTrialsFilter) => void;
  layout: CritiqueDistributionLayout;
  setLayout: (layout: CritiqueDistributionLayout) => void;
  style: CritiqueDistributionStyle;
  setStyle: (style: CritiqueDistributionStyle) => void;
  binCountMode: string;
  setBinCountMode: (mode: string) => void;
  showMeans: boolean;
  setShowMeans: (show: boolean) => void;
}

export function CritiqueDistributionChart({
  items,
  isLoading,
  isFetching,
  xMode,
  setXMode,
  yMode,
  setYMode,
  trialsFilter,
  setTrialsFilter,
  layout,
  setLayout,
  style,
  setStyle,
  binCountMode,
  setBinCountMode,
  showMeans,
  setShowMeans,
}: CritiqueDistributionChartProps) {
  const series = useMemo(
    () => (items ? computeSeries(items, trialsFilter) : []),
    [items, trialsFilter]
  );

  const [hiddenSources, setHiddenSources] = useState<Set<string>>(
    () => new Set()
  );
  const [hoveredSource, setHoveredSource] = useState<string | null>(null);

  const bins = useMemo(() => {
    if (xMode === "count") {
      return buildIntegerBins(Math.max(maxBadCount(series), 1));
    }
    const requested =
      binCountMode === "auto" ? 10 : Math.max(2, parseInt(binCountMode, 10) || 10);
    return buildPercentBins(requested);
  }, [xMode, binCountMode, series]);

  const binData = useMemo(
    () => buildSeriesBins(series, bins, xMode),
    [series, bins, xMode]
  );

  const totalTasks = series.reduce((sum, s) => sum + s.tasks.length, 0);

  const toggleSource = (source: string) => {
    setHiddenSources((prev) => {
      const next = new Set(prev);
      if (next.has(source)) next.delete(source);
      else next.add(source);
      return next;
    });
  };

  const showAll = () => setHiddenSources(new Set());
  const isolateLegendSource = (source: string) => {
    setHiddenSources(() => {
      const next = new Set(series.map((s) => s.source));
      next.delete(source);
      return next;
    });
  };

  const visibleSeries = series.filter((s) => !hiddenSources.has(s.source));

  if (isLoading) {
    return (
      <div className="relative min-h-80 border bg-card">
        <IndeterminateBar className="-top-px" />
      </div>
    );
  }

  if (!items || items.length === 0 || series.length === 0) {
    return (
      <div className="relative border bg-card">
        {isFetching && <IndeterminateBar className="-top-px" />}
        <DistributionToolbar
          xMode={xMode}
          setXMode={setXMode}
          yMode={yMode}
          setYMode={setYMode}
          trialsFilter={trialsFilter}
          setTrialsFilter={setTrialsFilter}
          layout={layout}
          setLayout={setLayout}
          style={style}
          setStyle={setStyle}
          binCountMode={binCountMode}
          setBinCountMode={setBinCountMode}
          showMeans={showMeans}
          setShowMeans={setShowMeans}
          datasetCount={0}
          totalTasks={0}
        />
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <BarChart3 />
            </EmptyMedia>
            <EmptyTitle>No distribution data</EmptyTitle>
            <EmptyDescription>
              No critique items match the current trial filter. Try widening
              the filter to include more source trials.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    );
  }

  return (
    <div className="relative border bg-card">
      {isFetching && <IndeterminateBar className="-top-px" />}
      <DistributionToolbar
        xMode={xMode}
        setXMode={setXMode}
        yMode={yMode}
        setYMode={setYMode}
        trialsFilter={trialsFilter}
        setTrialsFilter={setTrialsFilter}
        layout={layout}
        setLayout={setLayout}
        style={style}
        setStyle={setStyle}
        binCountMode={binCountMode}
        setBinCountMode={setBinCountMode}
        showMeans={showMeans}
        setShowMeans={setShowMeans}
        datasetCount={series.length}
        totalTasks={totalTasks}
      />

      <div className="relative px-4 py-4">
        {layout === "overlay" ? (
          <HistogramChart
            series={series}
            bins={bins}
            binData={binData}
            xMode={xMode}
            yMode={yMode}
            style={style}
            layout="overlay"
            width={920}
            height={460}
            hiddenSources={hiddenSources}
            hoveredSource={hoveredSource}
            setHoveredSource={setHoveredSource}
            showMeans={showMeans}
          />
        ) : (
          <DistributionGrid
            series={series}
            bins={bins}
            binData={binData}
            xMode={xMode}
            yMode={yMode}
            style={style}
            hiddenSources={hiddenSources}
            hoveredSource={hoveredSource}
            setHoveredSource={setHoveredSource}
            showMeans={showMeans}
          />
        )}
      </div>

      <DistributionLegend
        series={series}
        hiddenSources={hiddenSources}
        hoveredSource={hoveredSource}
        onToggle={toggleSource}
        onHover={setHoveredSource}
        onIsolate={isolateLegendSource}
        onShowAll={showAll}
        visibleCount={visibleSeries.length}
      />
    </div>
  );
}

function DistributionToolbar({
  xMode,
  setXMode,
  yMode,
  setYMode,
  trialsFilter,
  setTrialsFilter,
  layout,
  setLayout,
  style,
  setStyle,
  binCountMode,
  setBinCountMode,
  showMeans,
  setShowMeans,
  datasetCount,
  totalTasks,
}: {
  xMode: CritiqueDistributionXMode;
  setXMode: (mode: CritiqueDistributionXMode) => void;
  yMode: CritiqueDistributionYMode;
  setYMode: (mode: CritiqueDistributionYMode) => void;
  trialsFilter: CritiqueDistributionTrialsFilter;
  setTrialsFilter: (filter: CritiqueDistributionTrialsFilter) => void;
  layout: CritiqueDistributionLayout;
  setLayout: (layout: CritiqueDistributionLayout) => void;
  style: CritiqueDistributionStyle;
  setStyle: (style: CritiqueDistributionStyle) => void;
  binCountMode: string;
  setBinCountMode: (mode: string) => void;
  showMeans: boolean;
  setShowMeans: (show: boolean) => void;
  datasetCount: number;
  totalTasks: number;
}) {
  return (
    <ChartToolbar
      description={
        <>
          Distribution of <span className="text-foreground">{X_MODE_LABELS[xMode]}</span>{" "}
          per task across{" "}
          <span className="text-foreground">{datasetCount}</span> dataset
          {datasetCount === 1 ? "" : "s"} (
          <span className="text-foreground">{totalTasks}</span> task
          {totalTasks === 1 ? "" : "s"} total). Each task aggregates its{" "}
          {formatTrialFilterShort(trialsFilter)} source trials; the Y axis shows{" "}
          <span className="text-foreground">{Y_MODE_LABELS[yMode]}</span> within
          each dataset.
        </>
      }
    >
      <ChartToolbarSelect
        label="X"
        value={xMode}
        onValueChange={(v) => setXMode(v as CritiqueDistributionXMode)}
        options={[
          { value: "count", label: "# bad" },
          { value: "percent", label: "% bad" },
        ]}
      />
      <ChartToolbarSelect
        label="Y"
        value={yMode}
        onValueChange={(v) => setYMode(v as CritiqueDistributionYMode)}
        options={[
          { value: "count", label: "# tasks" },
          { value: "percent", label: "% tasks" },
        ]}
      />
      <ChartToolbarSelect
        label="Trials"
        value={trialsFilter}
        onValueChange={(v) =>
          setTrialsFilter(v as CritiqueDistributionTrialsFilter)
        }
        options={[
          { value: "all", label: TRIALS_FILTER_LABELS.all },
          { value: "non_errored", label: TRIALS_FILTER_LABELS.non_errored },
          { value: "errored", label: TRIALS_FILTER_LABELS.errored },
          { value: "successful", label: TRIALS_FILTER_LABELS.successful },
        ]}
      />
      <ChartToolbarSelect
        label="Layout"
        value={layout}
        onValueChange={(v) => setLayout(v as CritiqueDistributionLayout)}
        options={[
          { value: "overlay", label: "Overlay" },
          { value: "grid", label: "Grid" },
        ]}
      />
      <ChartToolbarSelect
        label="Style"
        value={style}
        onValueChange={(v) => setStyle(v as CritiqueDistributionStyle)}
        options={[
          { value: "bars", label: "Bars" },
          { value: "step", label: "Step" },
        ]}
      />
      {xMode === "percent" && (
        <ChartToolbarSelect
          label="Bins"
          value={binCountMode}
          onValueChange={setBinCountMode}
          options={[
            { value: "auto", label: "Auto (10)" },
            { value: "5", label: "5" },
            { value: "10", label: "10" },
            { value: "20", label: "20" },
            { value: "25", label: "25" },
          ]}
        />
      )}
      <ChartToolbarToggle
        label="Mean"
        checked={showMeans}
        onCheckedChange={setShowMeans}
      />
    </ChartToolbar>
  );
}

function DistributionGrid({
  series,
  bins,
  binData,
  xMode,
  yMode,
  style,
  hiddenSources,
  hoveredSource,
  setHoveredSource,
  showMeans,
}: {
  series: DatasetSeries[];
  bins: Bin[];
  binData: Map<string, SeriesBin[]>;
  xMode: CritiqueDistributionXMode;
  yMode: CritiqueDistributionYMode;
  style: CritiqueDistributionStyle;
  hiddenSources: ReadonlySet<string>;
  hoveredSource: string | null;
  setHoveredSource: (source: string | null) => void;
  showMeans: boolean;
}) {
  const visibleSeries = series.filter((s) => !hiddenSources.has(s.source));
  const cols = visibleSeries.length >= 4 ? 3 : visibleSeries.length >= 2 ? 2 : 1;
  return (
    <div
      className={cn(
        "grid gap-3",
        cols === 1 && "grid-cols-1",
        cols === 2 && "grid-cols-1 md:grid-cols-2",
        cols === 3 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
      )}
    >
      {visibleSeries.map((s) => (
        <div
          key={s.source}
          className="border rounded-md bg-background"
          onMouseEnter={() => setHoveredSource(s.source)}
          onMouseLeave={() => setHoveredSource(null)}
        >
          <HistogramChart
            series={series}
            bins={bins}
            binData={binData}
            xMode={xMode}
            yMode={yMode}
            style={style}
            layout="grid"
            width={420}
            height={260}
            hiddenSources={hiddenSources}
            hoveredSource={hoveredSource}
            setHoveredSource={setHoveredSource}
            showMeans={showMeans}
            isolateSource={s.source}
          />
          <div className="border-t px-3 py-2 text-[11px] text-muted-foreground flex items-center justify-between gap-3">
            <span className="font-mono">
              {s.tasks.length} task{s.tasks.length === 1 ? "" : "s"}
            </span>
            <span className="font-mono tabular-nums">
              {s.totalBad}/{s.totalTrials} bad
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function DistributionLegend({
  series,
  hiddenSources,
  hoveredSource,
  onToggle,
  onHover,
  onIsolate,
  onShowAll,
  visibleCount,
}: {
  series: DatasetSeries[];
  hiddenSources: ReadonlySet<string>;
  hoveredSource: string | null;
  onToggle: (source: string) => void;
  onHover: (source: string | null) => void;
  onIsolate: (source: string) => void;
  onShowAll: () => void;
  visibleCount: number;
}) {
  if (series.length === 0) return null;

  return (
    <div className="border-t">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 text-xs">
        <span className="text-muted-foreground">Datasets</span>
        {series.map((s) => {
          const isHidden = hiddenSources.has(s.source);
          const isHoveredOther =
            hoveredSource !== null && hoveredSource !== s.source;
          const color = datasetColor(s.colorIndex);
          return (
            <button
              key={s.source}
              type="button"
              onClick={() => onToggle(s.source)}
              onDoubleClick={(e) => {
                e.preventDefault();
                onIsolate(s.source);
              }}
              onMouseEnter={() => onHover(s.source)}
              onMouseLeave={() => onHover(null)}
              className={cn(
                "inline-flex items-center gap-2 rounded-md border px-2 py-1 transition-all",
                isHidden
                  ? "border-dashed border-border text-muted-foreground opacity-50"
                  : "border-border hover:bg-accent",
                isHoveredOther && "opacity-40"
              )}
              title={`Click to ${isHidden ? "show" : "hide"} • Double-click to isolate`}
            >
              <span
                className="inline-block size-2.5 rounded-sm"
                style={{ background: color }}
              />
              <span className="font-mono">{s.label}</span>
              <span className="text-muted-foreground tabular-nums">
                {s.tasks.length} task{s.tasks.length === 1 ? "" : "s"}
              </span>
              <span className="text-muted-foreground tabular-nums">
                · μ {s.meanBadCount.toFixed(1)} bad ({formatPercent(s.meanBadPct)})
              </span>
            </button>
          );
        })}
        {hiddenSources.size > 0 && (
          <button
            type="button"
            onClick={onShowAll}
            className="text-foreground underline-offset-2 hover:underline"
          >
            Show all ({visibleCount}/{series.length})
          </button>
        )}
      </div>
    </div>
  );
}
