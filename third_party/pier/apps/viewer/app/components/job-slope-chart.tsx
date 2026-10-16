//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";

import {
  ChartToolbar,
  ChartToolbarAction,
  ChartToolbarSelect,
  ChartToolbarSlider,
} from "~/components/ui/chart-toolbar";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty";
import { IndeterminateBar } from "~/components/ui/indeterminate-bar";
import {
  bareModelName,
  FAMILY_CONFIG,
  FAMILY_ORDER,
  familyColor,
  getFamily,
  sortByFamilyRank,
} from "~/lib/model-family";
import type { JobHeatmapCell, JobHeatmapData } from "~/lib/types";
import { cn, splitEffortLabel } from "~/lib/utils";

interface ModelSeries {
  rowKey: string;
  /** Friendly label, e.g. "gpt-5.5". */
  label: string;
  /** Full provider/model string for the tooltip. */
  fullLabel: string;
  sourceCount: number;
  family: string;
  rankIndex: number;
  rankCount: number;
  values: ({
    columnKey: string;
    columnLabel: string;
    rawValue: number;
    value: number;
    n: number;
  } | null)[];
}

export type JobSlopeChartConnectionMode = "config" | "model";
export type JobSlopeChartScoreMode = "raw" | "normalized";

function colorForSeries(series: ModelSeries): string {
  return familyColor(series.family, series.rankIndex, series.rankCount);
}

// ---------------------------------------------------------------------------
// Series construction
// ---------------------------------------------------------------------------

interface BuiltChart {
  series: ModelSeries[];
  /** Columns reordered (easiest → hardest by overall avg). */
  columns: { key: string; label: string }[];
  /** Min/max display value across all visible cells. */
  minValue: number;
  maxValue: number;
  /** Min/max raw reward across all visible cells. */
  rawMinValue: number;
  rawMaxValue: number;
}

function buildChart(
  data: JobHeatmapData,
  columnOrderOverride: string[] | null,
  connectionMode: JobSlopeChartConnectionMode,
  scoreMode: JobSlopeChartScoreMode,
  normalizationAmount: number
): BuiltChart | null {
  if (data.rows.length === 0 || data.columns.length === 0) return null;

  // Pull numeric reward + completed counts for every (row, col).
  const cellLookup = (
    rowKey: string,
    colKey: string
  ): JobHeatmapCell | undefined => data.cells[rowKey]?.[colKey];

  // Order columns by overall average reward, descending. This produces a
  // natural "easier → harder" left-to-right reading order.
  const columnAverages = data.columns.map((col) => {
    const values: number[] = [];
    for (const row of data.rows) {
      const cell = cellLookup(row.key, col.key);
      if (cell && cell.avg_reward !== null && cell.n_completed > 0) {
        values.push(cell.avg_reward);
      }
    }
    const avg =
      values.length === 0
        ? 0
        : values.reduce((a, b) => a + b, 0) / values.length;
    return { col, avg };
  });
  columnAverages.sort((a, b) => b.avg - a.avg);
  let orderedCols = columnAverages.map(({ col }) => col);

  // Apply user reordering, if any. Unknown override keys are dropped;
  // newly-seen columns (not in the override) are appended in default order.
  if (columnOrderOverride && columnOrderOverride.length > 0) {
    const byKey = new Map(orderedCols.map((c) => [c.key, c]));
    const ordered: typeof orderedCols = [];
    const seen = new Set<string>();
    for (const key of columnOrderOverride) {
      const col = byKey.get(key);
      if (col && !seen.has(key)) {
        ordered.push(col);
        seen.add(key);
      }
    }
    for (const col of orderedCols) {
      if (!seen.has(col.key)) ordered.push(col);
    }
    orderedCols = ordered;
  }

  type CellAccumulator = {
    columnKey: string;
    columnLabel: string;
    total: number;
    n: number;
  };
  type SeriesAccumulator = {
    rowKey: string;
    label: string;
    fullLabel: string;
    sourceRows: Set<string>;
    family: string;
    valuesByColumn: Map<string, CellAccumulator>;
  };
  const seriesByKey = new Map<string, SeriesAccumulator>();

  const modelKeyForRow = (modelName: string | null): string =>
    bareModelName(modelName ?? "(unknown)");

  const seriesKeyForRow = (row: (typeof data.rows)[number]): string => {
    const modelKey = modelKeyForRow(row.model_name);
    // Effort is part of the series identity so different effort levels of the
    // same model stay as separate lines instead of being averaged together.
    const effortKey = row.reasoning_effort ?? "";
    if (connectionMode === "model") {
      return `model::${modelKey}::${effortKey}`;
    }
    return `config::${row.agent_name ?? ""}::${modelKey}::${effortKey}`;
  };

  const seriesLabelForRow = (row: (typeof data.rows)[number]): string => {
    const modelLabel = row.reasoning_effort
      ? `${modelKeyForRow(row.model_name)} [${row.reasoning_effort}]`
      : modelKeyForRow(row.model_name);
    if (connectionMode === "model") return modelLabel;
    return (
      [row.agent_name, modelLabel].filter(Boolean).join(" / ") || "(unknown)"
    );
  };

  // Build raw series. Multiple job rows can intentionally collapse into one
  // visual series; when they do, averages are weighted by completed trials.
  for (const row of data.rows) {
    const seriesKey = seriesKeyForRow(row);
    const family = getFamily(row.model_provider, row.model_name);
    const existing =
      seriesByKey.get(seriesKey) ??
      {
        rowKey: seriesKey,
        label: seriesLabelForRow(row),
        fullLabel: seriesLabelForRow(row),
        sourceRows: new Set<string>(),
        family,
        valuesByColumn: new Map<string, CellAccumulator>(),
      };
    existing.sourceRows.add(row.key);
    if (row.label && !existing.fullLabel.includes(row.label)) {
      existing.fullLabel =
        existing.fullLabel === existing.label
          ? row.label
          : `${existing.fullLabel}, ${row.label}`;
    }

    for (const col of orderedCols) {
      const cell = cellLookup(row.key, col.key);
      if (!cell || cell.avg_reward === null || cell.n_completed <= 0) {
        continue;
      }
      const value = existing.valuesByColumn.get(col.key) ?? {
        columnKey: col.key,
        columnLabel: col.label,
        total: 0,
        n: 0,
      };
      value.total += cell.avg_reward * cell.n_completed;
      value.n += cell.n_completed;
      existing.valuesByColumn.set(col.key, value);
    }

    if (existing.valuesByColumn.size > 0) {
      seriesByKey.set(seriesKey, existing);
    }
  }

  const rawSeries: Omit<ModelSeries, "rankIndex" | "rankCount">[] = [];
  for (const acc of seriesByKey.values()) {
    const values: ModelSeries["values"] = orderedCols.map((col) => {
      const cell = acc.valuesByColumn.get(col.key);
      if (!cell || cell.n <= 0) return null;
      const rawValue = cell.total / cell.n;
      return {
        columnKey: cell.columnKey,
        columnLabel: cell.columnLabel,
        rawValue,
        value: rawValue,
        n: cell.n,
      };
    });
    if (values.every((v) => v === null)) continue;
    rawSeries.push({
      rowKey: acc.rowKey,
      label: acc.label,
      fullLabel: acc.fullLabel,
      sourceCount: acc.sourceRows.size,
      family: acc.family,
      values,
    });
  }

  if (rawSeries.length === 0) return null;

  const effectiveNormalization =
    scoreMode === "normalized" ? normalizationAmount : 0;
  if (effectiveNormalization > 0) {
    let globalMin = Number.POSITIVE_INFINITY;
    let globalMax = Number.NEGATIVE_INFINITY;
    for (const series of rawSeries) {
      for (const point of series.values) {
        if (!point) continue;
        if (point.rawValue < globalMin) globalMin = point.rawValue;
        if (point.rawValue > globalMax) globalMax = point.rawValue;
      }
    }
    const globalRange = Math.max(globalMax - globalMin, 1e-9);
    orderedCols.forEach((col, colIndex) => {
      const values = rawSeries
        .map((series) => series.values[colIndex]?.rawValue)
        .filter((value): value is number => value !== undefined);
      if (values.length === 0) return;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min;
      rawSeries.forEach((series) => {
        const point = series.values[colIndex];
        if (!point) return;
        const normalized =
          range > 1e-9
            ? globalMin + ((point.rawValue - min) / range) * globalRange
            : globalMin + globalRange / 2;
        point.value =
          point.rawValue +
          (normalized - point.rawValue) * effectiveNormalization;
      });
    });
  }

  // Group by family and assign within-family rank.
  const byFamily = new Map<string, typeof rawSeries>();
  for (const s of rawSeries) {
    const list = byFamily.get(s.family) ?? [];
    list.push(s);
    byFamily.set(s.family, list);
  }

  const series: ModelSeries[] = [];
  for (const [family, list] of byFamily.entries()) {
    const sorted = sortByFamilyRank(list, family, (s) => s.label);
    sorted.forEach((s, i) => {
      series.push({
        ...s,
        rankIndex: i,
        rankCount: sorted.length,
      });
    });
  }

  // Compute min/max across visible values.
  let minValue = Number.POSITIVE_INFINITY;
  let maxValue = Number.NEGATIVE_INFINITY;
  let rawMinValue = Number.POSITIVE_INFINITY;
  let rawMaxValue = Number.NEGATIVE_INFINITY;
  for (const s of series) {
    for (const v of s.values) {
      if (!v) continue;
      if (v.value < minValue) minValue = v.value;
      if (v.value > maxValue) maxValue = v.value;
      if (v.rawValue < rawMinValue) rawMinValue = v.rawValue;
      if (v.rawValue > rawMaxValue) rawMaxValue = v.rawValue;
    }
  }
  if (!Number.isFinite(minValue)) minValue = 0;
  if (!Number.isFinite(maxValue)) maxValue = 1;
  if (!Number.isFinite(rawMinValue)) rawMinValue = 0;
  if (!Number.isFinite(rawMaxValue)) rawMaxValue = 1;

  return {
    series,
    columns: orderedCols.map((c) => ({ key: c.key, label: c.label })),
    minValue,
    maxValue,
    rawMinValue,
    rawMaxValue,
  };
}

// ---------------------------------------------------------------------------
// Label collision avoidance
// ---------------------------------------------------------------------------

interface LabelEntry {
  /** True y position (where the dot is). */
  anchorY: number;
  /** Adjusted y position for the label, after collision resolution. */
  labelY: number;
  series: ModelSeries;
  value: number;
  rawValue: number;
}

/**
 * Place labels so consecutive ones are at least `minGap` apart while
 * minimising deviation from each label's anchor y. This is a balanced /
 * centred relaxation: when several labels would collide, they're stacked
 * symmetrically around the *mean* of their anchor positions so the cluster
 * doesn't cascade in one direction. Mutates `entries.labelY` in place.
 */
function relaxLabels(
  entries: LabelEntry[],
  minGap: number,
  yMin: number,
  yMax: number
) {
  if (entries.length === 0) return;

  // Sort by anchor y (top → bottom). The cluster algorithm assumes this
  // order is the final visual order, which is consistent with "labels
  // appear in the same order as their data points".
  entries.sort((a, b) => a.anchorY - b.anchorY);

  // Each entry starts as its own cluster; merge greedily while overlaps
  // would exist if cluster centres were placed at the cluster's mean
  // anchor.
  type Cluster = { startIdx: number; size: number; sumAnchor: number };
  const clusters: Cluster[] = entries.map((e, i) => ({
    startIdx: i,
    size: 1,
    sumAnchor: e.anchorY,
  }));

  const mergeOnce = (): boolean => {
    let merged = false;
    for (let i = clusters.length - 1; i > 0; i -= 1) {
      const a = clusters[i - 1];
      const b = clusters[i];
      const aMean = a.sumAnchor / a.size;
      const bMean = b.sumAnchor / b.size;
      const aLast = aMean + ((a.size - 1) * minGap) / 2;
      const bFirst = bMean - ((b.size - 1) * minGap) / 2;
      if (aLast + minGap > bFirst + 1e-6) {
        a.size += b.size;
        a.sumAnchor += b.sumAnchor;
        clusters.splice(i, 1);
        merged = true;
      }
    }
    return merged;
  };

  // Iterate to fixed point. Edge clamping (next loop) can introduce new
  // overlaps, so we re-merge after clamping until stable.
  for (let iter = 0; iter < 32; iter += 1) {
    while (mergeOnce()) {}
    let clampChanged = false;
    for (const c of clusters) {
      const half = ((c.size - 1) * minGap) / 2;
      const center = c.sumAnchor / c.size;
      const minCenter = yMin + half;
      const maxCenter = yMax - half;
      let target = center;
      if (target < minCenter) target = minCenter;
      if (target > maxCenter) target = maxCenter;
      const newSum = target * c.size;
      if (Math.abs(newSum - c.sumAnchor) > 1e-6) {
        c.sumAnchor = newSum;
        clampChanged = true;
      }
    }
    if (!clampChanged) break;
  }

  // Assign labelY for every entry based on its cluster.
  for (const c of clusters) {
    const half = ((c.size - 1) * minGap) / 2;
    const center = c.sumAnchor / c.size;
    const top = center - half;
    for (let k = 0; k < c.size; k += 1) {
      entries[c.startIdx + k].labelY = top + k * minGap;
    }
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const HEIGHT = 1080;
const WIDTH = 1880;
const MARGIN_TOP = 64;
const MARGIN_BOTTOM = 80;
const MARGIN_LEFT = 620;
const MARGIN_RIGHT = 560;
const PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
const PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;
const LABEL_GAP = 14;
const BAR_W = 50;
const BAR_H = 2;
const LINE_W = 1.5;
const LINE_OPACITY = 0.55;

function formatPercentScore(value: number): string {
  return `${(value * 100).toFixed(1)}`;
}

function formatRawScore(value: number, usePercentScale: boolean): string {
  if (usePercentScale) return formatPercentScore(value);
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function buildTicks(min: number, max: number, usePercentScale: boolean): number[] {
  if (usePercentScale) {
    return [0, 0.25, 0.5, 0.75, 1].filter(
      (tick) => tick >= min - 1e-9 && tick <= max + 1e-9
    );
  }
  const range = Math.max(max - min, 1e-6);
  return Array.from({ length: 5 }, (_, i) => min + (range * i) / 4);
}

interface HoveredPoint {
  rowKey: string;
  fullLabel: string;
  sourceCount: number;
  family: string;
  columnLabel: string;
  rawValue: number;
  value: number;
  n: number;
  /** Pixel coords inside the SVG viewBox. */
  cx: number;
  cy: number;
}

export interface JobSlopeChartProps {
  data: JobHeatmapData | undefined;
  isLoading?: boolean;
  isFetching?: boolean;
  defaultConnectionMode?: JobSlopeChartConnectionMode;
  defaultScoreMode?: JobSlopeChartScoreMode;
}

export function JobSlopeChart({
  data,
  isLoading,
  isFetching,
  defaultConnectionMode = "config",
  defaultScoreMode = "raw",
}: JobSlopeChartProps) {
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<HoveredPoint | null>(null);
  const [columnOrder, setColumnOrder] = useState<string[] | null>(null);
  const [connectionMode, setConnectionMode] =
    useState<JobSlopeChartConnectionMode>(defaultConnectionMode);
  const [scoreMode, setScoreMode] =
    useState<JobSlopeChartScoreMode>(defaultScoreMode);
  const [normalizationAmount, setNormalizationAmount] = useState(1);

  const chart = useMemo(
    () =>
      data
        ? buildChart(
            data,
            columnOrder,
            connectionMode,
            scoreMode,
            normalizationAmount
          )
        : null,
    [data, columnOrder, connectionMode, scoreMode, normalizationAmount]
  );

  const swapColumns = (i: number, j: number) => {
    if (!chart) return;
    if (i < 0 || j < 0 || i >= chart.columns.length || j >= chart.columns.length) {
      return;
    }
    const next = chart.columns.map((c) => c.key);
    [next[i], next[j]] = [next[j], next[i]];
    setColumnOrder(next);
  };

  const resetColumnOrder = () => setColumnOrder(null);

  if (isLoading || (!chart && isFetching)) {
    return (
      <div className="border bg-card relative min-h-80">
        {(isLoading || isFetching) && <IndeterminateBar className="-top-px" />}
      </div>
    );
  }

  if (!chart) {
    return (
      <div className="border bg-card relative">
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <Search />
            </EmptyMedia>
            <EmptyTitle>No data</EmptyTitle>
            <EmptyDescription>
              No completed, non-errored trials match the current filters.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    );
  }

  const { series, columns, minValue, maxValue, rawMinValue, rawMaxValue } = chart;

  const usePercentScale = rawMinValue >= 0 && rawMaxValue <= 1;
  const yMin = usePercentScale ? 0 : Math.floor(minValue * 10) / 10;
  const yMax = usePercentScale ? 1 : Math.ceil(maxValue * 10) / 10;
  const yRange = Math.max(yMax - yMin, 1e-6);

  const xForCol = (i: number): number => {
    if (columns.length === 1) return MARGIN_LEFT + PLOT_W / 2;
    return MARGIN_LEFT + (i / (columns.length - 1)) * PLOT_W;
  };

  const yForValue = (v: number): number => {
    const t = (v - yMin) / yRange;
    return MARGIN_TOP + (1 - t) * PLOT_H;
  };

  // Build label entries for each axis. Every column gets a column of
  // labels (anchored to that column's value), with collision avoidance
  // applied independently per column. Side labels (leftmost / rightmost)
  // include the model name; interior labels show the score number only
  // so the chart stays legible.
  type AxisLabel = LabelEntry & { colIndex: number };

  const labelsPerCol: AxisLabel[][] = columns.map(() => []);
  for (const s of series) {
    s.values.forEach((v, i) => {
      if (!v) return;
      labelsPerCol[i].push({
        anchorY: yForValue(v.value),
        labelY: yForValue(v.value),
        series: s,
        value: v.value,
        rawValue: v.rawValue,
        colIndex: i,
      });
    });
  }

  // Anti-collision for the side columns uses a larger gap (model name fits
  // on each row); interior columns just need to keep score numbers from
  // colliding so the gap can be tighter.
  labelsPerCol.forEach((labels, i) => {
    const isSide = i === 0 || i === columns.length - 1;
    relaxLabels(
      labels,
      isSide ? LABEL_GAP : LABEL_GAP - 4,
      MARGIN_TOP - 6,
      MARGIN_TOP + PLOT_H + 6
    );
  });

  const leftLabels = labelsPerCol[0] ?? [];
  const rightLabels = labelsPerCol[columns.length - 1] ?? [];
  const interiorLabels: AxisLabel[] = [];
  for (let i = 1; i < columns.length - 1; i += 1) {
    interiorLabels.push(...labelsPerCol[i]);
  }

  const yTicks = buildTicks(yMin, yMax, usePercentScale);
  const connectionLabel =
    connectionMode === "model" ? "same model" : "same agent + model";
  const scaleLabel =
    scoreMode === "normalized"
      ? `${Math.round(normalizationAmount * 100)}% normalized per dataset`
      : "raw reward";

  return (
    <div className="border bg-card relative">
      {isFetching && <IndeterminateBar className="-top-px" />}
      <ChartToolbar
        description={
          <>
            Avg reward per dataset, connected by {connectionLabel}. Showing{" "}
            {scaleLabel}; completed runs only. Hue = model family.
          </>
        }
      >
        <ChartToolbarSelect
          label="Connect"
          value={connectionMode}
          onValueChange={(value) =>
            setConnectionMode(value as JobSlopeChartConnectionMode)
          }
          options={[
            { value: "model", label: "Same model" },
            { value: "config", label: "Same agent + model" },
          ]}
        />
        <ChartToolbarSelect
          label="Scale"
          value={scoreMode}
          onValueChange={(value) =>
            setScoreMode(value as JobSlopeChartScoreMode)
          }
          options={[
            { value: "raw", label: "Raw reward" },
            { value: "normalized", label: "Normalize per dataset" },
          ]}
        />
        {scoreMode === "normalized" && (
          <ChartToolbarSlider
            label="Amount"
            ariaLabel="Normalization amount"
            value={Math.round(normalizationAmount * 100)}
            onValueChange={(value) => setNormalizationAmount(value / 100)}
          />
        )}
        {columnOrder !== null && (
          <ChartToolbarAction onClick={resetColumnOrder}>
            Reset order
          </ChartToolbarAction>
        )}
      </ChartToolbar>
      <div className="overflow-x-auto">
        <div
          className="relative mx-auto"
          style={{ width: WIDTH }}
        >
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            width={WIDTH}
            style={{ display: "block" }}
            role="img"
            aria-label="Cross-benchmark slope chart"
          >
          {/* Vertical axes for each benchmark */}
          {columns.map((col, i) => {
            const x = xForCol(i);
            return (
              <g key={col.key}>
                <line
                  x1={x}
                  x2={x}
                  y1={MARGIN_TOP}
                  y2={MARGIN_TOP + PLOT_H}
                  stroke="var(--border)"
                  strokeWidth={1}
                />
              </g>
            );
          })}

          {/* Y-axis tick labels (mid-axis on the leftmost column) */}
          {yTicks.map((t) => {
            const y = yForValue(t);
            return (
              <g key={t}>
                <line
                  x1={MARGIN_LEFT - 6}
                  x2={MARGIN_LEFT}
                  y1={y}
                  y2={y}
                  stroke="var(--border)"
                  strokeWidth={1}
                />
                <text
                  x={MARGIN_LEFT - 10}
                  y={y + 3}
                  textAnchor="end"
                  fontSize={10}
                  className="fill-muted-foreground"
                >
                  {usePercentScale
                    ? (t * 100).toFixed(0)
                    : formatRawScore(t, false)}
                </text>
                <line
                  x1={MARGIN_LEFT}
                  x2={MARGIN_LEFT + PLOT_W}
                  y1={y}
                  y2={y}
                  stroke="var(--border)"
                  strokeWidth={1}
                  strokeDasharray="2 4"
                  opacity={0.35}
                />
              </g>
            );
          })}

          {/* Lines (drawn first so dots/labels sit on top). Hovered series
              renders last so it always wins z-order. */}
          {[...series]
            .sort((a, b) => {
              if (hoveredKey === a.rowKey) return 1;
              if (hoveredKey === b.rowKey) return -1;
              return 0;
            })
            .map((s) => {
              const color = colorForSeries(s);
              const isDimmed = hoveredKey !== null && hoveredKey !== s.rowKey;
              const isHovered = hoveredKey === s.rowKey;
              // Build segments between adjacent non-null values. Lines
              // attach to the inside edges of the bars so the bars look
              // like nodes connected by edges, not pierced through.
              const segments: { x1: number; y1: number; x2: number; y2: number }[] = [];
              let prev: { x: number; y: number } | null = null;
              s.values.forEach((v, i) => {
                if (!v) {
                  prev = null;
                  return;
                }
                const x = xForCol(i);
                const y = yForValue(v.value);
                if (prev) {
                  segments.push({
                    x1: prev.x + BAR_W / 2,
                    y1: prev.y,
                    x2: x - BAR_W / 2,
                    y2: y,
                  });
                }
                prev = { x, y };
              });
              return (
                <g
                  key={`line-${s.rowKey}`}
                  style={{ opacity: isDimmed ? 0.18 : 1 }}
                >
                  {segments.map((seg, j) => (
                    <line
                      key={j}
                      x1={seg.x1}
                      y1={seg.y1}
                      x2={seg.x2}
                      y2={seg.y2}
                      stroke={color}
                      strokeWidth={isHovered ? 2 : LINE_W}
                      strokeOpacity={isHovered ? 1 : LINE_OPACITY}
                      strokeLinecap="butt"
                    />
                  ))}
                </g>
              );
            })}

          {/* Dots + per-axis score badges */}
          {series.map((s) => {
            const color = colorForSeries(s);
            const isDimmed = hoveredKey !== null && hoveredKey !== s.rowKey;
            const isHovered = hoveredKey === s.rowKey;
            return (
              <g
                key={`dots-${s.rowKey}`}
                style={{ opacity: isDimmed ? 0.18 : 1 }}
              >
                {s.values.map((v, i) => {
                  if (!v) return null;
                  const x = xForCol(i);
                  const y = yForValue(v.value);
                  return (
                    <g
                      key={`${s.rowKey}-${i}`}
                      onMouseEnter={() => {
                        setHoveredKey(s.rowKey);
                        setHoveredPoint({
                          rowKey: s.rowKey,
                          fullLabel: s.fullLabel,
                          sourceCount: s.sourceCount,
                          family: s.family,
                          columnLabel: v.columnLabel,
                          rawValue: v.rawValue,
                          value: v.value,
                          n: v.n,
                          cx: x,
                          cy: y,
                        });
                      }}
                      onMouseLeave={() => {
                        setHoveredKey(null);
                        setHoveredPoint(null);
                      }}
                      style={{ cursor: "pointer" }}
                    >
                      <rect
                        x={x - BAR_W / 2 - 4}
                        y={y - BAR_H / 2 - 6}
                        width={BAR_W + 8}
                        height={BAR_H + 12}
                        fill="transparent"
                        pointerEvents="all"
                      />
                      <rect
                        x={x - BAR_W / 2}
                        y={y - BAR_H / 2}
                        width={BAR_W}
                        height={BAR_H}
                        fill={color}
                        shapeRendering="crispEdges"
                      />
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* Interior labels: small score-only badges to the right of the
              bar, with collision avoidance per axis. */}
          {interiorLabels.map((entry) => {
            const color = colorForSeries(entry.series);
            const barRight = xForCol(entry.colIndex) + BAR_W / 2;
            const x = barRight + 6;
            const isDimmed =
              hoveredKey !== null && hoveredKey !== entry.series.rowKey;
            const isHovered = hoveredKey === entry.series.rowKey;
            return (
              <g
                key={`lbl-int-${entry.series.rowKey}-${entry.colIndex}`}
                style={{ opacity: isDimmed ? 0.18 : 1 }}
              >
                {Math.abs(entry.labelY - entry.anchorY) > 1.5 && (
                  <line
                    x1={barRight + 2}
                    y1={entry.anchorY}
                    x2={x}
                    y2={entry.labelY}
                    stroke={color}
                    strokeWidth={1}
                    opacity={0.4}
                  />
                )}
                <text
                  x={x}
                  y={entry.labelY + 4}
                  textAnchor="start"
                  fontSize={14}
                  fontWeight={isHovered ? 600 : 400}
                  fontFamily="var(--font-mono, ui-monospace)"
                  fill={color}
                  style={{
                    paintOrder: "stroke",
                    stroke: "var(--background)",
                    strokeWidth: 3,
                    strokeLinejoin: "round",
                  }}
                >
                  {formatRawScore(entry.rawValue, usePercentScale)}
                </text>
              </g>
            );
          })}

          {/* Left labels: model name + score, anchored to first known value. */}
          {leftLabels.map((entry) => {
            const color = colorForSeries(entry.series);
            const barLeft = xForCol(entry.colIndex) - BAR_W / 2;
            const x = barLeft - 8;
            const isDimmed =
              hoveredKey !== null && hoveredKey !== entry.series.rowKey;
            const isHovered = hoveredKey === entry.series.rowKey;
            return (
              <g
                key={`lbl-l-${entry.series.rowKey}`}
                style={{ opacity: isDimmed ? 0.18 : 1, cursor: "pointer" }}
                onMouseEnter={() => setHoveredKey(entry.series.rowKey)}
                onMouseLeave={() => setHoveredKey(null)}
              >
                {/* Leader line if label was nudged off-axis */}
                {Math.abs(entry.labelY - entry.anchorY) > 1.5 && (
                  <line
                    x1={x}
                    y1={entry.labelY}
                    x2={barLeft - 2}
                    y2={entry.anchorY}
                    stroke={color}
                    strokeWidth={1}
                    opacity={0.5}
                  />
                )}
                <text
                  x={x}
                  y={entry.labelY + 4}
                  textAnchor="end"
                  fontSize={14}
                  fontWeight={isHovered ? 600 : 400}
                  fill={color}
                >
                  {(() => {
                    const { base, effort } = splitEffortLabel(entry.series.label);
                    return (
                      <>
                        <tspan>{base}</tspan>
                        {effort && (
                          <tspan dx={4} opacity={0.55}>
                            {effort}
                          </tspan>
                        )}
                      </>
                    );
                  })()}
                  <tspan
                    dx={6}
                    fontFamily="var(--font-mono, ui-monospace)"
                    fontWeight={isHovered ? 700 : 500}
                  >
                    {formatRawScore(entry.rawValue, usePercentScale)}
                  </tspan>
                </text>
              </g>
            );
          })}

          {/* Right labels: model name + score, anchored to last known value. */}
          {rightLabels.map((entry) => {
            const color = colorForSeries(entry.series);
            const barRight = xForCol(entry.colIndex) + BAR_W / 2;
            const x = barRight + 8;
            const isDimmed =
              hoveredKey !== null && hoveredKey !== entry.series.rowKey;
            const isHovered = hoveredKey === entry.series.rowKey;
            return (
              <g
                key={`lbl-r-${entry.series.rowKey}`}
                style={{ opacity: isDimmed ? 0.18 : 1, cursor: "pointer" }}
                onMouseEnter={() => setHoveredKey(entry.series.rowKey)}
                onMouseLeave={() => setHoveredKey(null)}
              >
                {Math.abs(entry.labelY - entry.anchorY) > 1.5 && (
                  <line
                    x1={x}
                    y1={entry.labelY}
                    x2={barRight + 2}
                    y2={entry.anchorY}
                    stroke={color}
                    strokeWidth={1}
                    opacity={0.5}
                  />
                )}
                <text
                  x={x}
                  y={entry.labelY + 4}
                  textAnchor="start"
                  fontSize={14}
                  fontWeight={isHovered ? 600 : 400}
                  fill={color}
                >
                  <tspan
                    fontFamily="var(--font-mono, ui-monospace)"
                    fontWeight={isHovered ? 700 : 500}
                  >
                    {formatRawScore(entry.rawValue, usePercentScale)}
                  </tspan>
                  {(() => {
                    const { base, effort } = splitEffortLabel(entry.series.label);
                    return (
                      <>
                        <tspan dx={6}>{base}</tspan>
                        {effort && (
                          <tspan dx={4} opacity={0.55}>
                            {effort}
                          </tspan>
                        )}
                      </>
                    );
                  })()}
                </text>
              </g>
            );
          })}
          </svg>
          {/* Dataset column headers (HTML overlay) with reorder buttons. */}
          {columns.map((col, i) => (
            <div
              key={`hdr-${col.key}`}
              className="absolute flex items-center gap-1"
              style={{
                left: `${(xForCol(i) / WIDTH) * 100}%`,
                top: `${((MARGIN_TOP + PLOT_H + 14) / HEIGHT) * 100}%`,
                transform: "translateX(-50%)",
              }}
            >
              <button
                type="button"
                onClick={() => swapColumns(i, i - 1)}
                disabled={i === 0}
                className="text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:hover:text-muted-foreground transition-colors px-1 -mr-0.5"
                aria-label={`Move ${col.label} left`}
                title={`Move ${col.label} left`}
              >
                <ChevronLeft className="size-3" />
              </button>
              <span className="text-[13px] font-semibold tracking-tight whitespace-nowrap">
                {col.label}
              </span>
              <button
                type="button"
                onClick={() => swapColumns(i, i + 1)}
                disabled={i === columns.length - 1}
                className="text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:hover:text-muted-foreground transition-colors px-1 -ml-0.5"
                aria-label={`Move ${col.label} right`}
                title={`Move ${col.label} right`}
              >
                <ChevronRight className="size-3" />
              </button>
            </div>
          ))}
          {hoveredPoint && (
            <div
              className="pointer-events-none absolute z-10 rounded-md border bg-popover text-popover-foreground shadow-md px-3 py-2 text-xs"
              style={{
                // Position relative to viewBox by converting to percentage.
                left: `${(hoveredPoint.cx / WIDTH) * 100}%`,
                top: `${(hoveredPoint.cy / HEIGHT) * 100}%`,
                transform: "translate(12px, -50%)",
                minWidth: 220,
              }}
            >
            <div className="font-medium mb-1.5">{hoveredPoint.fullLabel}</div>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
              <span className="text-muted-foreground">Dataset</span>
              <span>{hoveredPoint.columnLabel}</span>
              <span className="text-muted-foreground">Raw reward</span>
              <span className="font-mono">
                {hoveredPoint.rawValue.toFixed(4)}
              </span>
              <span className="text-muted-foreground">Trials</span>
              <span className="font-mono">{hoveredPoint.n}</span>
              {hoveredPoint.sourceCount > 1 && (
                <>
                  <span className="text-muted-foreground">Grouped rows</span>
                  <span className="font-mono">{hoveredPoint.sourceCount}</span>
                </>
              )}
              <span className="text-muted-foreground">Family</span>
              <span>
                {FAMILY_CONFIG[hoveredPoint.family]?.label ?? hoveredPoint.family}
              </span>
            </div>
            </div>
          )}
        </div>
      </div>
      <Legend series={series} hoveredKey={hoveredKey} setHovered={setHoveredKey} />
    </div>
  );
}

function Legend({
  series,
  hoveredKey,
  setHovered,
}: {
  series: ModelSeries[];
  hoveredKey: string | null;
  setHovered: (k: string | null) => void;
}) {
  // Group by family, then sort within family by rank for a tidy legend.
  const grouped = useMemo(() => {
    const map = new Map<string, ModelSeries[]>();
    for (const s of series) {
      const list = map.get(s.family) ?? [];
      list.push(s);
      map.set(s.family, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => a.rankIndex - b.rankIndex);
    }
    return FAMILY_ORDER
      .filter((f) => map.has(f))
      .map((f) => ({ family: f, members: map.get(f)! }));
  }, [series]);

  return (
    <div className="flex flex-wrap gap-x-6 gap-y-2 px-4 py-3 border-t text-xs">
      {grouped.map(({ family, members }) => (
        <div key={family} className="flex items-center gap-2">
          <span className="text-muted-foreground">
            {FAMILY_CONFIG[family]?.label ?? family}
          </span>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            {members.map((s) => {
              const color = colorForSeries(s);
              const isDimmed = hoveredKey !== null && hoveredKey !== s.rowKey;
              return (
                <button
                  key={s.rowKey}
                  type="button"
                  className={cn(
                    "inline-flex items-center gap-1.5 transition-opacity",
                    isDimmed && "opacity-30"
                  )}
                  onMouseEnter={() => setHovered(s.rowKey)}
                  onMouseLeave={() => setHovered(null)}
                  title={s.fullLabel}
                >
                  <span
                    className="inline-block h-[3px] w-4 rounded-full"
                    style={{ background: color }}
                  />
                  <span className="font-mono">
                    {(() => {
                      const { base, effort } = splitEffortLabel(s.label);
                      return (
                        <>
                          {base}
                          {effort && (
                            <span className="opacity-[0.55]"> {effort}</span>
                          )}
                        </>
                      );
                    })()}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
