//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { useNavigate } from "react-router";

import {
  ChartToolbar,
  ChartToolbarSelect,
  ChartToolbarSlider,
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
import {
  bareModelName,
  FAMILY_CONFIG,
  FAMILY_ORDER,
  familyColor,
  getFamily,
  sortByFamilyRank,
} from "~/lib/model-family";
import type { JobHeatmapCell, JobHeatmapData } from "~/lib/types";
import { cn } from "~/lib/utils";

type ScaleMetricKey =
  | "avg_duration_ms"
  | "avg_output_tokens"
  | "avg_peak_context_tokens"
  | "avg_agent_steps"
  | "avg_cost_usd"
  | "avg_input_tokens"
  | "avg_cached_input_tokens";

type XScaleMode = "linear" | "log";
type TrendMethod = "local" | "binned" | "direct";
type InterpolationMode = "linear" | "monotone" | "spline";

interface ScaleMetric {
  key: ScaleMetricKey;
  label: string;
  axisLabel: string;
  prefersLog: boolean;
}

const SCALE_METRICS: ScaleMetric[] = [
  {
    key: "avg_duration_ms",
    label: "Wall clock time",
    axisLabel: "Avg task wall clock",
    prefersLog: true,
  },
  {
    key: "avg_output_tokens",
    label: "Output tokens",
    axisLabel: "Avg task output tokens",
    prefersLog: true,
  },
  {
    key: "avg_peak_context_tokens",
    label: "Peak trajectory tokens",
    axisLabel: "Avg task peak trajectory tokens",
    prefersLog: true,
  },
  {
    key: "avg_agent_steps",
    label: "Agent steps",
    axisLabel: "Avg task agent steps",
    prefersLog: false,
  },
  {
    key: "avg_cost_usd",
    label: "Cost",
    axisLabel: "Avg task cost",
    prefersLog: true,
  },
  {
    key: "avg_input_tokens",
    label: "Input tokens",
    axisLabel: "Avg task uncached input tokens",
    prefersLog: true,
  },
  {
    key: "avg_cached_input_tokens",
    label: "Cached input tokens",
    axisLabel: "Avg task cached input tokens",
    prefersLog: true,
  },
];

interface TaskScale {
  columnKey: string;
  columnLabel: string;
  x: number;
  rewardMin: number;
  rewardMax: number;
}

interface ScalePoint {
  key: string;
  columnKey: string;
  taskLabel: string;
  taskUrl: string | null;
  x: number;
  y: number;
  rawY: number;
  taskNormalizedY: number;
  n: number;
  configScale: number | null;
}

interface ScaleSeries {
  rowKey: string;
  label: string;
  fullLabel: string;
  family: string;
  rankIndex: number;
  rankCount: number;
  sourceCount: number;
  points: ScalePoint[];
}

interface BuiltScalingChart {
  series: ScaleSeries[];
  taskScales: TaskScale[];
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  rawMin: number;
  rawMax: number;
}

interface TrendSource {
  x: number;
  scaledX: number;
  y: number;
  n: number;
}

type HoveredPoint =
  | {
      kind: "point";
      series: ScaleSeries;
      point: ScalePoint;
      cx: number;
      cy: number;
    }
  | {
      kind: "trend";
      series: ScaleSeries;
      trend: TrendPoint;
      cx: number;
      cy: number;
      sourceCount: number;
    };

interface TrendPoint {
  x: number;
  scaledX: number;
  y: number;
  lo: number;
  hi: number;
  n: number;
}

function metricValue(
  cell: JobHeatmapCell,
  metricKey: ScaleMetricKey
): number | null {
  switch (metricKey) {
    case "avg_duration_ms":
      return cell.avg_duration_ms;
    case "avg_output_tokens":
      return cell.avg_output_tokens;
    case "avg_peak_context_tokens":
      return cell.avg_peak_context_tokens;
    case "avg_agent_steps":
      return cell.avg_agent_steps;
    case "avg_cost_usd":
      return cell.avg_cost_usd;
    case "avg_input_tokens":
      return cell.avg_input_tokens;
    case "avg_cached_input_tokens":
      return cell.avg_cached_input_tokens;
  }
}

function getCellTaskUrl(jobName: string, cell: JobHeatmapCell): string | null {
  const params = cell.route_params;
  if (!params) return null;
  const targetJobName = params.job_name ?? jobName;
  const source = params.source || "_";
  const agent = params.agent_name || "_";
  const modelProvider = params.model_provider || "_";
  const modelName = params.model_name || "_";
  return `/jobs/${encodeURIComponent(targetJobName)}/tasks/${encodeURIComponent(source)}/${encodeURIComponent(agent)}/${encodeURIComponent(modelProvider)}/${encodeURIComponent(modelName)}/${encodeURIComponent(params.task_name)}`;
}

function buildScalingChart(
  jobName: string,
  data: JobHeatmapData,
  metricKey: ScaleMetricKey,
  normalizationAmount: number
): BuiltScalingChart | null {
  if (data.rows.length === 0 || data.columns.length === 0) return null;

  const cellLookup = (
    rowKey: string,
    colKey: string
  ): JobHeatmapCell | undefined => data.cells[rowKey]?.[colKey];

  const taskScales: TaskScale[] = [];
  for (const col of data.columns) {
    let xTotal = 0;
    let xWeight = 0;
    let rewardMin = Number.POSITIVE_INFINITY;
    let rewardMax = Number.NEGATIVE_INFINITY;

    for (const row of data.rows) {
      const cell = cellLookup(row.key, col.key);
      if (!cell || cell.n_completed <= 0) continue;

      const x = metricValue(cell, metricKey);
      if (x !== null && Number.isFinite(x)) {
        const weight = Math.max(cell.n_completed, 1);
        xTotal += x * weight;
        xWeight += weight;
      }

      if (cell.avg_reward !== null && Number.isFinite(cell.avg_reward)) {
        rewardMin = Math.min(rewardMin, cell.avg_reward);
        rewardMax = Math.max(rewardMax, cell.avg_reward);
      }
    }

    if (
      xWeight > 0 &&
      Number.isFinite(rewardMin) &&
      Number.isFinite(rewardMax)
    ) {
      taskScales.push({
        columnKey: col.key,
        columnLabel: col.label,
        x: xTotal / xWeight,
        rewardMin,
        rewardMax,
      });
    }
  }

  if (taskScales.length === 0) return null;

  taskScales.sort((a, b) => a.x - b.x || a.columnLabel.localeCompare(b.columnLabel));
  const taskByKey = new Map(taskScales.map((task) => [task.columnKey, task]));

  type SeriesAcc = {
    rowKey: string;
    label: string;
    fullLabel: string;
    sourceRows: Set<string>;
    family: string;
    points: ScalePoint[];
  };
  const seriesByKey = new Map<string, SeriesAcc>();

  for (const row of data.rows) {
    const model = bareModelName(row.model_name ?? "(unknown)");
    const agent = row.agent_name ?? "(unknown)";
    const seriesKey = `${agent}::${model}`;
    const family = getFamily(row.model_provider, row.model_name);
    const existing =
      seriesByKey.get(seriesKey) ??
      {
        rowKey: seriesKey,
        label: model,
        fullLabel: `${agent} / ${model}`,
        sourceRows: new Set<string>(),
        family,
        points: [],
      };
    existing.sourceRows.add(row.key);
    if (row.label && !existing.fullLabel.includes(row.label)) {
      existing.fullLabel =
        existing.fullLabel === `${agent} / ${model}`
          ? row.label
          : `${existing.fullLabel}, ${row.label}`;
    }

    for (const task of taskScales) {
      const cell = cellLookup(row.key, task.columnKey);
      if (
        !cell ||
        cell.n_completed <= 0 ||
        cell.avg_reward === null ||
        !Number.isFinite(cell.avg_reward)
      ) {
        continue;
      }
      const range = task.rewardMax - task.rewardMin;
      const taskNormalizedY =
        range > 1e-9 ? (cell.avg_reward - task.rewardMin) / range : 0.5;
      const y =
        cell.avg_reward +
        (taskNormalizedY - cell.avg_reward) * normalizationAmount;
      existing.points.push({
        key: `${seriesKey}::${task.columnKey}`,
        columnKey: task.columnKey,
        taskLabel: task.columnLabel,
        taskUrl: getCellTaskUrl(jobName, cell),
        x: task.x,
        y,
        rawY: cell.avg_reward,
        taskNormalizedY,
        n: cell.n_completed,
        configScale: metricValue(cell, metricKey),
      });
    }

    if (existing.points.length > 0) {
      seriesByKey.set(seriesKey, existing);
    }
  }

  const rawSeries = [...seriesByKey.values()]
    .map((series) => ({
      ...series,
      points: series.points.sort(
        (a, b) => a.x - b.x || a.taskLabel.localeCompare(b.taskLabel)
      ),
    }))
    .filter((series) => series.points.length > 0);
  if (rawSeries.length === 0) return null;

  const byFamily = new Map<string, typeof rawSeries>();
  for (const series of rawSeries) {
    const list = byFamily.get(series.family) ?? [];
    list.push(series);
    byFamily.set(series.family, list);
  }

  const series: ScaleSeries[] = [];
  for (const [family, list] of byFamily.entries()) {
    const sorted = sortByFamilyRank(list, family, (item) => item.label);
    sorted.forEach((item, i) => {
      series.push({
        rowKey: item.rowKey,
        label: item.label,
        fullLabel: item.fullLabel,
        family: item.family,
        rankIndex: i,
        rankCount: sorted.length,
        sourceCount: item.sourceRows.size,
        points: item.points,
      });
    });
  }

  let xMin = Number.POSITIVE_INFINITY;
  let xMax = Number.NEGATIVE_INFINITY;
  let yMin = Number.POSITIVE_INFINITY;
  let yMax = Number.NEGATIVE_INFINITY;
  let rawMin = Number.POSITIVE_INFINITY;
  let rawMax = Number.NEGATIVE_INFINITY;
  for (const task of taskScales) {
    xMin = Math.min(xMin, task.x);
    xMax = Math.max(xMax, task.x);
  }
  for (const item of series) {
    for (const point of item.points) {
      yMin = Math.min(yMin, point.y);
      yMax = Math.max(yMax, point.y);
      rawMin = Math.min(rawMin, point.rawY);
      rawMax = Math.max(rawMax, point.rawY);
    }
  }

  return {
    series,
    taskScales,
    xMin,
    xMax,
    yMin: Number.isFinite(yMin) ? yMin : 0,
    yMax: Number.isFinite(yMax) ? yMax : 1,
    rawMin: Number.isFinite(rawMin) ? rawMin : 0,
    rawMax: Number.isFinite(rawMax) ? rawMax : 1,
  };
}

const WIDTH = 1180;
const HEIGHT = 680;
const MARGIN_TOP = 46;
const MARGIN_RIGHT = 260;
const MARGIN_BOTTOM = 76;
const MARGIN_LEFT = 72;
const PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
const PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

function formatCompact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${Math.round(value / 1_000)}k`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function formatDurationMs(ms: number): string {
  const seconds = ms / 1000;
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;
}

function formatMetricValue(value: number, metricKey: ScaleMetricKey): string {
  if (metricKey === "avg_duration_ms") return formatDurationMs(value);
  if (metricKey === "avg_cost_usd") return `$${formatCompact(value)}`;
  if (metricKey === "avg_agent_steps") return formatCompact(value);
  return formatCompact(value);
}

function formatScore(value: number, percentScale: boolean): string {
  if (percentScale) return `${(value * 100).toFixed(0)}`;
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function linearTicks(min: number, max: number, count = 6): number[] {
  if (Math.abs(max - min) < 1e-9) return [min];
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

function logTicks(min: number, max: number): number[] {
  const lo = Math.log10(Math.max(min, Number.MIN_VALUE));
  const hi = Math.log10(Math.max(max, Number.MIN_VALUE));
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const ticks: number[] = [];
  for (let p = Math.floor(lo); p <= Math.ceil(hi); p += 1) {
    for (const m of [1, 2, 5]) {
      const tick = m * 10 ** p;
      if (tick >= min * 0.999 && tick <= max * 1.001) ticks.push(tick);
    }
  }
  if (ticks.length > 8) {
    return linearTicks(lo, hi, 6).map((v) => 10 ** v);
  }
  return ticks.length > 0 ? ticks : [min, max];
}

function padDomain(min: number, max: number): [number, number] {
  if (Math.abs(max - min) < 1e-9) {
    const pad = Math.max(Math.abs(max) * 0.1, 1);
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function aggregateTrendSources(
  points: ScalePoint[],
  scaleX: (x: number) => number
): TrendSource[] {
  const byX = new Map<number, { xTotal: number; yTotal: number; n: number }>();
  for (const point of points) {
    const scaledX = scaleX(point.x);
    if (!Number.isFinite(scaledX)) continue;
    const key = Number(scaledX.toPrecision(12));
    const existing = byX.get(key);
    const weight = Math.max(point.n, 1);
    if (existing) {
      existing.xTotal += point.x * weight;
      existing.yTotal += point.y * weight;
      existing.n += weight;
    } else {
      byX.set(key, {
        xTotal: point.x * weight,
        yTotal: point.y * weight,
        n: weight,
      });
    }
  }
  return [...byX.entries()]
    .map(([scaledX, entry]) => ({
      x: entry.xTotal / Math.max(entry.n, 1),
      scaledX,
      y: entry.yTotal / Math.max(entry.n, 1),
      n: entry.n,
    }))
    .sort((a, b) => a.scaledX - b.scaledX);
}

function supportStats(
  sources: TrendSource[],
  weights: number[],
  meanY: number
): { lo: number; hi: number; n: number } {
  let weightTotal = 0;
  let varianceTotal = 0;
  for (let i = 0; i < sources.length; i += 1) {
    const weight = weights[i];
    if (weight <= 0) continue;
    weightTotal += weight;
    varianceTotal += weight * (sources[i].y - meanY) ** 2;
  }
  if (weightTotal <= 0) return { lo: meanY, hi: meanY, n: 0 };
  const variance = varianceTotal / weightTotal;
  const stderr = Math.sqrt(variance / Math.max(weightTotal, 1));
  const spread = 1.96 * stderr;
  return {
    lo: meanY - spread,
    hi: meanY + spread,
    n: weightTotal,
  };
}

function weightedTrendPoint(
  sources: TrendSource[],
  weights: number[]
): TrendPoint | null {
  let weightTotal = 0;
  let xTotal = 0;
  let scaledXTotal = 0;
  let yTotal = 0;
  for (let i = 0; i < sources.length; i += 1) {
    const weight = weights[i];
    if (weight <= 0) continue;
    weightTotal += weight;
    xTotal += sources[i].x * weight;
    scaledXTotal += sources[i].scaledX * weight;
    yTotal += sources[i].y * weight;
  }
  if (weightTotal <= 0) return null;
  const y = yTotal / weightTotal;
  const stats = supportStats(sources, weights, y);
  return {
    x: xTotal / weightTotal,
    scaledX: scaledXTotal / weightTotal,
    y,
    lo: stats.lo,
    hi: stats.hi,
    n: stats.n,
  };
}

function autoBinCount(pointCount: number): number {
  if (pointCount <= 0) return 0;
  return Math.min(24, Math.max(6, Math.round(Math.sqrt(pointCount) * 2)));
}

function parseBinCount(value: string, pointCount: number): number {
  if (value === "auto") return autoBinCount(pointCount);
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return autoBinCount(pointCount);
  return Math.max(2, parsed);
}

function buildDirectTrend(
  points: ScalePoint[],
  scaleX: (x: number) => number
): TrendPoint[] {
  return aggregateTrendSources(points, scaleX).map((source) => ({
    x: source.x,
    scaledX: source.scaledX,
    y: source.y,
    lo: source.y,
    hi: source.y,
    n: source.n,
  }));
}

function buildBinnedTrend(
  points: ScalePoint[],
  scaleX: (x: number) => number,
  binCount: number
): TrendPoint[] {
  const sources = aggregateTrendSources(points, scaleX);
  if (sources.length === 0) return [];
  if (sources.length <= binCount) {
    return sources.map((source) => ({
      x: source.x,
      scaledX: source.scaledX,
      y: source.y,
      lo: source.y,
      hi: source.y,
      n: source.n,
    }));
  }

  const bins: TrendPoint[] = [];
  for (let i = 0; i < binCount; i += 1) {
    const start = Math.floor((i * sources.length) / binCount);
    const end = Math.floor(((i + 1) * sources.length) / binCount);
    const slice = sources.slice(start, Math.max(end, start + 1));
    const point = weightedTrendPoint(
      slice,
      slice.map((source) => Math.max(source.n, 1))
    );
    if (point) bins.push(point);
  }
  return bins;
}

function buildLocalTrend(
  points: ScalePoint[],
  amount: number,
  scaleX: (x: number) => number,
  binCount: number
): TrendPoint[] {
  const sources = aggregateTrendSources(points, scaleX);
  if (amount <= 0 || sources.length < 3) {
    return sources.map((source) => ({
      x: source.x,
      scaledX: source.scaledX,
      y: source.y,
      lo: source.y,
      hi: source.y,
      n: source.n,
    }));
  }

  const scaledMin = sources[0].scaledX;
  const scaledMax = sources[sources.length - 1].scaledX;
  const span = Math.max(scaledMax - scaledMin, 1e-9);
  const bandwidth = Math.max(span * (0.04 + amount * 0.24), 1e-9);
  const centerCount = Math.min(Math.max(binCount, 3), sources.length);
  const centers = buildBinnedTrend(points, scaleX, centerCount);

  return centers
    .map((center) => {
      const weights = sources.map((source) => {
        const dx = (source.scaledX - center.scaledX) / bandwidth;
        return Math.exp(-0.5 * dx * dx) * Math.max(source.n, 1);
      });
      return weightedTrendPoint(sources, weights);
    })
    .filter((point): point is TrendPoint => point !== null)
    .sort((a, b) => a.scaledX - b.scaledX);
}

function buildTrend(
  points: ScalePoint[],
  method: TrendMethod,
  amount: number,
  scaleX: (x: number) => number,
  binCount: number
): TrendPoint[] {
  if (method === "direct") return buildDirectTrend(points, scaleX);
  if (method === "binned") return buildBinnedTrend(points, scaleX, binCount);
  return buildLocalTrend(points, amount, scaleX, binCount);
}

function linePath(
  points: { x: number; y: number }[],
  interpolation: InterpolationMode
): string {
  if (points.length === 0) return "";
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
  if (interpolation === "linear" || points.length === 2) {
    return points
      .map((point, i) => `${i === 0 ? "M" : "L"} ${point.x} ${point.y}`)
      .join(" ");
  }
  if (interpolation === "monotone") {
    const n = points.length;
    const slopes = Array.from({ length: n - 1 }, (_, i) => {
      const dx = points[i + 1].x - points[i].x;
      return Math.abs(dx) < 1e-9 ? 0 : (points[i + 1].y - points[i].y) / dx;
    });
    const tangents = Array.from({ length: n }, (_, i) => {
      if (i === 0) return slopes[0];
      if (i === n - 1) return slopes[n - 2];
      if (slopes[i - 1] * slopes[i] <= 0) return 0;
      return (slopes[i - 1] + slopes[i]) / 2;
    });
    const path = [`M ${points[0].x} ${points[0].y}`];
    for (let i = 0; i < n - 1; i += 1) {
      const p1 = points[i];
      const p2 = points[i + 1];
      const dx = p2.x - p1.x;
      path.push(
        `C ${p1.x + dx / 3} ${p1.y + (tangents[i] * dx) / 3}, ${p2.x - dx / 3} ${p2.y - (tangents[i + 1] * dx) / 3}, ${p2.x} ${p2.y}`
      );
    }
    return path.join(" ");
  }

  const path = [`M ${points[0].x} ${points[0].y}`];
  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    path.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`);
  }
  return path.join(" ");
}

function bandPath(points: { x: number; lo: number; hi: number }[]): string {
  if (points.length < 2) return "";
  const upper = points.map((point, i) =>
    `${i === 0 ? "M" : "L"} ${point.x} ${point.hi}`
  );
  const lower = [...points]
    .reverse()
    .map((point) => `L ${point.x} ${point.lo}`);
  return `${upper.join(" ")} ${lower.join(" ")} Z`;
}

function nearestTrendPoint(
  trend: TrendPoint[],
  clientX: number,
  svg: SVGSVGElement,
  xForValue: (value: number) => number,
  yForValue: (value: number) => number
): { trend: TrendPoint; cx: number; cy: number } | null {
  if (trend.length === 0) return null;
  const rect = svg.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * WIDTH;
  let nearest = trend[0];
  let nearestDistance = Math.abs(xForValue(nearest.x) - x);
  for (const point of trend.slice(1)) {
    const distance = Math.abs(xForValue(point.x) - x);
    if (distance < nearestDistance) {
      nearest = point;
      nearestDistance = distance;
    }
  }
  return {
    trend: nearest,
    cx: xForValue(nearest.x),
    cy: yForValue(nearest.y),
  };
}

export interface JobScalingChartProps {
  jobName: string;
  data: JobHeatmapData | undefined;
  isLoading?: boolean;
  isFetching?: boolean;
}

export function JobScalingChart({
  jobName,
  data,
  isLoading,
  isFetching,
}: JobScalingChartProps) {
  const navigate = useNavigate();
  const [metricKey, setMetricKey] =
    useState<ScaleMetricKey>("avg_agent_steps");
  const [xScaleMode, setXScaleMode] = useState<XScaleMode>("log");
  const [trendMethod, setTrendMethod] = useState<TrendMethod>("local");
  const [interpolationMode, setInterpolationMode] =
    useState<InterpolationMode>("linear");
  const [binCountMode, setBinCountMode] = useState("auto");
  const [normalizationAmount, setNormalizationAmount] = useState(0.5);
  const [smoothingAmount, setSmoothingAmount] = useState(0.25);
  const [showRawPoints, setShowRawPoints] = useState(true);
  const [showSupportBand, setShowSupportBand] = useState(true);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<HoveredPoint | null>(null);

  const metric = SCALE_METRICS.find((item) => item.key === metricKey)!;
  const chart = useMemo(
    () =>
      data ? buildScalingChart(jobName, data, metricKey, normalizationAmount) : null,
    [data, jobName, metricKey, normalizationAmount]
  );

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
            <EmptyTitle>No scale data</EmptyTitle>
            <EmptyDescription>
              No completed trials have both reward and the selected scale
              metric. Try another metric or widen the active filters.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    );
  }

  const positiveXMin = Math.min(
    ...chart.taskScales.map((task) => task.x).filter((x) => x > 0)
  );
  const canUseLog = Number.isFinite(positiveXMin) && chart.xMax > 0;
  const effectiveScale: XScaleMode =
    xScaleMode === "log" && canUseLog ? "log" : "linear";
  const [linearXLo, linearXHi] = padDomain(chart.xMin, chart.xMax);
  const xLo =
    effectiveScale === "log"
      ? Math.max(positiveXMin, Number.MIN_VALUE)
      : linearXLo;
  const xHi = effectiveScale === "log" ? chart.xMax : linearXHi;
  const xDomainLo =
    effectiveScale === "log" ? Math.log10(xLo) : xLo;
  const xDomainHi =
    effectiveScale === "log" ? Math.log10(Math.max(xHi, xLo * 1.001)) : xHi;
  const xRange = Math.max(xDomainHi - xDomainLo, 1e-9);

  const usePercentScale =
    chart.rawMin >= 0 &&
    chart.rawMax <= 1 &&
    chart.yMin >= 0 &&
    chart.yMax <= 1;
  const yLo = usePercentScale ? 0 : padDomain(chart.yMin, chart.yMax)[0];
  const yHi = usePercentScale ? 1 : padDomain(chart.yMin, chart.yMax)[1];
  const yRange = Math.max(yHi - yLo, 1e-9);

  const xForValue = (value: number) => {
    const scaled =
      effectiveScale === "log"
        ? Math.log10(Math.max(value, xLo))
        : value;
    return MARGIN_LEFT + ((scaled - xDomainLo) / xRange) * PLOT_W;
  };
  const scaleXValue = (value: number) =>
    effectiveScale === "log"
      ? value > 0
        ? Math.log10(value)
        : Number.NaN
      : value;
  const yForValue = (value: number) =>
    MARGIN_TOP + (1 - (value - yLo) / yRange) * PLOT_H;

  const xTicks =
    effectiveScale === "log"
      ? logTicks(xLo, xHi)
      : linearTicks(xLo, xHi, 6);
  const yTicks = usePercentScale
    ? [0, 0.25, 0.5, 0.75, 1]
    : linearTicks(yLo, yHi, 6);

  const seriesByFamily = (() => {
    const map = new Map<string, ScaleSeries[]>();
    for (const series of chart.series) {
      const list = map.get(series.family) ?? [];
      list.push(series);
      map.set(series.family, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => a.rankIndex - b.rankIndex);
    }
    return FAMILY_ORDER.filter((family) => map.has(family)).map((family) => ({
      family,
      members: map.get(family)!,
    }));
  })();

  return (
    <div className="border bg-card relative">
      {isFetching && <IndeterminateBar className="-top-px" />}
      <ChartToolbar description="Performance vs task scale.">
        <ChartToolbarSelect
          label="X metric"
          value={metricKey}
          onValueChange={(value) => {
            const next = value as ScaleMetricKey;
            setMetricKey(next);
            const nextMetric = SCALE_METRICS.find((item) => item.key === next);
            if (nextMetric?.prefersLog) setXScaleMode("log");
          }}
          options={SCALE_METRICS.map((item) => ({
            value: item.key,
            label: item.label,
          }))}
        />
        <ChartToolbarSelect
          label="X"
          value={xScaleMode}
          onValueChange={(value) => setXScaleMode(value as XScaleMode)}
          options={[
            { value: "log", label: "Log scale" },
            { value: "linear", label: "Linear" },
          ]}
        />
        <ChartToolbarSelect
          label="Trend"
          value={trendMethod}
          onValueChange={(value) => setTrendMethod(value as TrendMethod)}
          options={[
            { value: "binned", label: "Binned" },
            { value: "local", label: "Local avg" },
            { value: "direct", label: "Direct" },
          ]}
        />
        {trendMethod !== "direct" && (
          <ChartToolbarSelect
            label={trendMethod === "binned" ? "Bins" : "Samples"}
            value={binCountMode}
            onValueChange={setBinCountMode}
            options={[
              { value: "auto", label: "Auto" },
              { value: "8", label: "8" },
              { value: "12", label: "12" },
              { value: "16", label: "16" },
              { value: "24", label: "24" },
            ]}
          />
        )}
        <ChartToolbarSelect
          label="Line"
          value={interpolationMode}
          onValueChange={(value) =>
            setInterpolationMode(value as InterpolationMode)
          }
          options={[
            { value: "linear", label: "Linear" },
            { value: "monotone", label: "Monotone" },
            { value: "spline", label: "Spline" },
          ]}
        />
        <ChartToolbarSlider
          label="Normalize"
          ariaLabel="Task normalization amount"
          value={Math.round(normalizationAmount * 100)}
          onValueChange={(value) => setNormalizationAmount(value / 100)}
        />
        {trendMethod === "local" && (
          <ChartToolbarSlider
            label="Smooth"
            ariaLabel="Smoothing amount"
            value={Math.round(smoothingAmount * 100)}
            onValueChange={(value) => setSmoothingAmount(value / 100)}
          />
        )}
        {trendMethod !== "direct" && (
          <ChartToolbarToggle
            label="Band"
            checked={showSupportBand}
            onCheckedChange={setShowSupportBand}
          />
        )}
        <ChartToolbarToggle
          label="Raw points"
          checked={showRawPoints}
          onCheckedChange={setShowRawPoints}
        />
      </ChartToolbar>

      <div className="overflow-x-auto">
        <div className="relative mx-auto" style={{ width: WIDTH }}>
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            width={WIDTH}
            style={{ display: "block" }}
            role="img"
            aria-label="Model reward by task scale"
          >
            <rect
              x={MARGIN_LEFT}
              y={MARGIN_TOP}
              width={PLOT_W}
              height={PLOT_H}
              fill="transparent"
              stroke="var(--border)"
              strokeWidth={1}
            />
            {xTicks.map((tick) => {
              const x = xForValue(tick);
              return (
                <g key={`x-${tick}`}>
                  <line
                    x1={x}
                    x2={x}
                    y1={MARGIN_TOP}
                    y2={MARGIN_TOP + PLOT_H}
                    stroke="var(--border)"
                    strokeDasharray="2 4"
                    opacity={0.35}
                  />
                  <text
                    x={x}
                    y={MARGIN_TOP + PLOT_H + 20}
                    textAnchor="middle"
                    fontSize={11}
                    className="fill-muted-foreground"
                  >
                    {formatMetricValue(tick, metricKey)}
                  </text>
                </g>
              );
            })}
            {yTicks.map((tick) => {
              const y = yForValue(tick);
              return (
                <g key={`y-${tick}`}>
                  <line
                    x1={MARGIN_LEFT}
                    x2={MARGIN_LEFT + PLOT_W}
                    y1={y}
                    y2={y}
                    stroke="var(--border)"
                    strokeDasharray="2 4"
                    opacity={0.35}
                  />
                  <text
                    x={MARGIN_LEFT - 9}
                    y={y + 4}
                    textAnchor="end"
                    fontSize={11}
                    className="fill-muted-foreground"
                  >
                    {formatScore(tick, usePercentScale)}
                  </text>
                </g>
              );
            })}

            <text
              x={MARGIN_LEFT + PLOT_W / 2}
              y={HEIGHT - 24}
              textAnchor="middle"
              fontSize={12}
              className="fill-foreground"
            >
              {metric.axisLabel}
              {effectiveScale === "log" ? " (log)" : ""}
            </text>
            <text
              x={20}
              y={MARGIN_TOP + PLOT_H / 2}
              textAnchor="middle"
              fontSize={12}
              transform={`rotate(-90 20 ${MARGIN_TOP + PLOT_H / 2})`}
              className="fill-foreground"
            >
              Reward{normalizationAmount > 0 ? " with task normalization" : ""}
              {usePercentScale ? " (%)" : ""}
            </text>

            {[...chart.series]
              .sort((a, b) => {
                if (hoveredKey === a.rowKey) return 1;
                if (hoveredKey === b.rowKey) return -1;
                return 0;
              })
              .map((series) => {
                const color = familyColor(
                  series.family,
                  series.rankIndex,
                  series.rankCount
                );
                const isHovered = hoveredKey === series.rowKey;
                const isDimmed = hoveredKey !== null && !isHovered;
                const sourceCount = aggregateTrendSources(
                  series.points,
                  scaleXValue
                ).length;
                const trend = buildTrend(
                  series.points,
                  trendMethod,
                  smoothingAmount,
                  scaleXValue,
                  parseBinCount(binCountMode, sourceCount)
                ).filter((point) => effectiveScale !== "log" || point.x > 0);
                const linePoints = trend.map((point) => ({
                  x: xForValue(point.x),
                  y: yForValue(point.y),
                }));
                const path = linePath(linePoints, interpolationMode);
                const band = bandPath(
                  trend.map((point) => ({
                    x: xForValue(point.x),
                    hi: yForValue(Math.min(Math.max(point.hi, yLo), yHi)),
                    lo: yForValue(Math.min(Math.max(point.lo, yLo), yHi)),
                  }))
                );
                return (
                  <g
                    key={`line-${series.rowKey}`}
                    style={{ opacity: isDimmed ? 0.16 : 1 }}
                  >
                    {showSupportBand && trendMethod !== "direct" && band && (
                      <path
                        d={band}
                        fill={color}
                        opacity={isHovered ? 0.16 : 0.07}
                        stroke="none"
                      />
                    )}
                    <path
                      d={path}
                      fill="none"
                      stroke={color}
                      strokeWidth={isHovered ? 2.5 : 1.6}
                      strokeOpacity={isHovered ? 1 : 0.68}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    {path && (
                      <path
                        d={path}
                        fill="none"
                        stroke="transparent"
                        strokeWidth={16}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        style={{ cursor: "pointer" }}
                        onMouseEnter={() => setHoveredKey(series.rowKey)}
                        onMouseMove={(event) => {
                          const svg = event.currentTarget.ownerSVGElement;
                          if (!svg) return;
                          const nearest = nearestTrendPoint(
                            trend,
                            event.clientX,
                            svg,
                            xForValue,
                            yForValue
                          );
                          if (!nearest) return;
                          setHoveredKey(series.rowKey);
                          setHoveredPoint({
                            kind: "trend",
                            series,
                            trend: nearest.trend,
                            cx: nearest.cx,
                            cy: nearest.cy,
                            sourceCount,
                          });
                        }}
                        onMouseLeave={() => {
                          setHoveredKey(null);
                          setHoveredPoint(null);
                        }}
                      />
                    )}
                  </g>
                );
              })}

            {chart.series.map((series) => {
              const color = familyColor(
                series.family,
                series.rankIndex,
                series.rankCount
              );
              const isDimmed = hoveredKey !== null && hoveredKey !== series.rowKey;
              return (
                <g
                  key={`points-${series.rowKey}`}
                  style={{ opacity: isDimmed ? 0.18 : 1 }}
                >
                  {series.points
                    .filter((point) => effectiveScale !== "log" || point.x > 0)
                    .map((point) => {
                      const cx = xForValue(point.x);
                      const cy = yForValue(point.y);
                      const isHovered = hoveredKey === series.rowKey;
                      return (
                        <g
                          key={point.key}
                          role={point.taskUrl ? "link" : undefined}
                          tabIndex={point.taskUrl ? 0 : undefined}
                          aria-label={
                            point.taskUrl ? `Open task ${point.taskLabel}` : undefined
                          }
                          onMouseEnter={() => {
                            setHoveredKey(series.rowKey);
                            setHoveredPoint({
                              kind: "point",
                              series,
                              point,
                              cx,
                              cy,
                            });
                          }}
                          onMouseLeave={() => {
                            setHoveredKey(null);
                            setHoveredPoint(null);
                          }}
                          onClick={() => {
                            if (point.taskUrl) navigate(point.taskUrl);
                          }}
                          onKeyDown={(event) => {
                            if (
                              point.taskUrl &&
                              (event.key === "Enter" || event.key === " ")
                            ) {
                              event.preventDefault();
                              navigate(point.taskUrl);
                            }
                          }}
                          style={{ cursor: point.taskUrl ? "pointer" : "default" }}
                        >
                          <circle
                            cx={cx}
                            cy={cy}
                            r={9}
                            fill="transparent"
                            pointerEvents="all"
                          />
                          {showRawPoints && (
                            <circle
                              cx={cx}
                              cy={cy}
                              r={isHovered ? 4.2 : 3.1}
                              fill={color}
                              stroke="var(--background)"
                              strokeWidth={1.2}
                            />
                          )}
                        </g>
                      );
                    })}
                </g>
              );
            })}
          </svg>

          {hoveredPoint && (
            <div
              className="pointer-events-none absolute z-10 rounded-md border bg-popover text-popover-foreground shadow-md px-3 py-2 text-xs"
              style={{
                left: `${(hoveredPoint.cx / WIDTH) * 100}%`,
                top: `${(hoveredPoint.cy / HEIGHT) * 100}%`,
                transform: "translate(14px, -50%)",
                minWidth: 250,
              }}
            >
              <div className="font-medium mb-1.5">
                {hoveredPoint.series.fullLabel}
              </div>
              {hoveredPoint.kind === "point" ? (
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
                  <span className="text-muted-foreground">Task</span>
                  <span className="truncate">{hoveredPoint.point.taskLabel}</span>
                  <span className="text-muted-foreground">Task avg</span>
                  <span className="font-mono">
                    {formatMetricValue(hoveredPoint.point.x, metricKey)}
                  </span>
                  <span className="text-muted-foreground">Config avg</span>
                  <span className="font-mono">
                    {hoveredPoint.point.configScale === null
                      ? "-"
                      : formatMetricValue(
                          hoveredPoint.point.configScale,
                          metricKey
                        )}
                  </span>
                  <span className="text-muted-foreground">Reward</span>
                  <span className="font-mono">
                    {formatScore(hoveredPoint.point.rawY, usePercentScale)}
                    <span className="text-muted-foreground">
                      {" "}
                      (n = {hoveredPoint.point.n})
                    </span>
                  </span>
                  <span className="text-muted-foreground">Task norm</span>
                  <span className="font-mono">
                    {(hoveredPoint.point.taskNormalizedY * 100).toFixed(0)}
                  </span>
                  <span className="text-muted-foreground">Family</span>
                  <span>
                    {FAMILY_CONFIG[hoveredPoint.series.family]?.label ??
                      hoveredPoint.series.family}
                  </span>
                </div>
              ) : (
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
                  <span className="text-muted-foreground">Trend</span>
                  <span>
                    {trendMethod === "local"
                      ? "Local avg"
                      : trendMethod === "binned"
                        ? "Binned avg"
                        : "Direct"}
                  </span>
                  <span className="text-muted-foreground">Task scale</span>
                  <span className="font-mono">
                    {formatMetricValue(hoveredPoint.trend.x, metricKey)}
                  </span>
                  <span className="text-muted-foreground">Reward</span>
                  <span className="font-mono">
                    {formatScore(hoveredPoint.trend.y, usePercentScale)}
                  </span>
                  {trendMethod !== "direct" && (
                    <>
                      <span className="text-muted-foreground">Band</span>
                      <span className="font-mono">
                        {formatScore(hoveredPoint.trend.lo, usePercentScale)}-
                        {formatScore(hoveredPoint.trend.hi, usePercentScale)}
                      </span>
                    </>
                  )}
                  <span className="text-muted-foreground">Support</span>
                  <span className="font-mono">
                    n = {Math.round(hoveredPoint.trend.n)}
                    <span className="text-muted-foreground">
                      {" "}
                      / {hoveredPoint.sourceCount} x positions
                    </span>
                  </span>
                  <span className="text-muted-foreground">Family</span>
                  <span>
                    {FAMILY_CONFIG[hoveredPoint.series.family]?.label ??
                      hoveredPoint.series.family}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="border-t">
        <div className="flex flex-wrap gap-x-6 gap-y-2 px-4 py-3 text-xs">
          {seriesByFamily.map(({ family, members }) => (
            <div key={family} className="flex items-center gap-2">
              <span className="text-muted-foreground">
                {FAMILY_CONFIG[family]?.label ?? family}
              </span>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                {members.map((member) => {
                  const color = familyColor(
                    member.family,
                    member.rankIndex,
                    member.rankCount
                  );
                  const isDimmed =
                    hoveredKey !== null && hoveredKey !== member.rowKey;
                  return (
                    <button
                      key={member.rowKey}
                      type="button"
                      className={cn(
                        "inline-flex items-center gap-1.5 transition-opacity",
                        isDimmed && "opacity-30"
                      )}
                      onMouseEnter={() => setHoveredKey(member.rowKey)}
                      onMouseLeave={() => setHoveredKey(null)}
                      title={member.fullLabel}
                    >
                      <span
                        className="inline-block h-0.5 w-4"
                        style={{ background: color }}
                      />
                      <span className="font-mono">{member.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
