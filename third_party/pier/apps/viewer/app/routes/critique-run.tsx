//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { ColumnDef, SortingState, VisibilityState } from "@tanstack/react-table";
import { FileText, Search, X } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { parseAsArrayOf, parseAsInteger, parseAsString, useQueryState } from "nuqs";
import { Link, useNavigate, useParams } from "react-router";

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { CodeBlock } from "~/components/ui/code-block";
import { DataTable, SortableHeader } from "~/components/ui/data-table";
import { Combobox, type ComboboxOption } from "~/components/ui/combobox";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty";
import { LoadingDots } from "~/components/ui/loading-dots";
import { Input } from "~/components/ui/input";
import { Kbd } from "~/components/ui/kbd";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "~/components/ui/pagination";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "~/components/ui/hover-card";
import {
  ChartToolbar,
  ChartToolbarAction,
  ChartToolbarSelect,
} from "~/components/ui/chart-toolbar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { ResizableHeatmapGrid } from "~/components/resizable-heatmap-grid";
import {
  CritiqueDistributionChart,
  type CritiqueDistributionLayout,
  type CritiqueDistributionStyle,
  type CritiqueDistributionTrialsFilter,
  type CritiqueDistributionXMode,
  type CritiqueDistributionYMode,
} from "~/components/critique-distribution-chart";
import {
  fetchCritiqueHeatmap,
  fetchCritiqueItemFilters,
  fetchCritiqueItems,
  fetchCritiqueRun,
} from "~/lib/api";
import { cn } from "~/lib/utils";
import type {
  CritiqueHeatmapCell,
  CritiqueHeatmapColumnBy,
  CritiqueHeatmapRowBy,
  CritiqueHeatmapSourceTrialsFilter,
  CritiqueItemSummary,
} from "~/lib/types";

const PAGE_SIZE = 100;

function formatDateTime(date: string | null): string {
  if (!date) return "-";
  return new Date(date).toLocaleString();
}

function formatCostUSD(cost: number | null): string {
  if (cost === null) return "-";
  if (cost > 0 && cost < 0.01) return "<$0.01";
  return `$${cost.toFixed(2)}`;
}

function formatCritiqueStatus(status: string): string {
  return status
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function CritiqueStatusBadge({ status }: { status: string }) {
  const variant =
    status === "failed"
      ? "destructive"
      : status === "completed"
        ? "secondary"
        : "outline";

  return <Badge variant={variant}>{formatCritiqueStatus(status)}</Badge>;
}

function RewardText({ reward }: { reward: number }) {
  return (
    <span className="font-mono tabular-nums text-foreground">
      {reward.toFixed(2)}
    </span>
  );
}

function SourceOutcomeCell({ item }: { item: CritiqueItemSummary }) {
  if (item.source_error_type) {
    return (
      <div className="text-right text-destructive">
        {item.source_error_type}
      </div>
    );
  }
  if (item.source_reward === null) {
    return <div className="text-right text-muted-foreground">-</div>;
  }
  return (
    <div className="text-right">
      <RewardText reward={item.source_reward} />
    </div>
  );
}

function RatingBadge({ rating }: { rating: string | null }) {
  if (!rating) return <span className="text-muted-foreground">-</span>;

  const className =
    rating === "good"
      ? "text-green-700 dark:text-green-400"
      : rating === "bad"
        ? "text-destructive"
        : "";

  return (
    <span className={`font-mono tabular-nums ${className}`}>
      {rating}
    </span>
  );
}

function trialUrl(
  jobName: string,
  critiqueRunName: string,
  item: CritiqueItemSummary
): string | null {
  if (!item.task_name || !item.agent_name) return null;

  const params = new URLSearchParams({
    tab: "critiques",
    critique: critiqueRunName,
  });

  return `/jobs/${encodeURIComponent(jobName)}/tasks/${encodeURIComponent(item.source ?? "_")}/${encodeURIComponent(item.agent_name)}/${encodeURIComponent(item.model_provider ?? "_")}/${encodeURIComponent(item.model_name ?? "_")}/${encodeURIComponent(item.task_name)}/trials/${encodeURIComponent(item.source_trial_name)}?${params.toString()}`;
}

const columns: ColumnDef<CritiqueItemSummary>[] = [
  {
    accessorKey: "rating",
    header: ({ column }) => <SortableHeader column={column}>Rating</SortableHeader>,
    cell: ({ row }) => <RatingBadge rating={row.original.rating} />,
  },
  {
    accessorKey: "status",
    header: ({ column }) => <SortableHeader column={column}>State</SortableHeader>,
    cell: ({ row }) => <CritiqueStatusBadge status={row.original.status} />,
  },
  {
    accessorKey: "tags",
    header: ({ column }) => <SortableHeader column={column}>Tags</SortableHeader>,
    cell: ({ row }) =>
      row.original.tags.length > 0 ? (
        <div className="flex max-w-[24rem] flex-wrap gap-x-2 gap-y-1">
          {row.original.tags.map((tag, index) => (
            <span key={tag} className="font-mono">
              {tag}
              {index < row.original.tags.length - 1 ? "," : ""}
            </span>
          ))}
        </div>
      ) : (
        <span className="text-muted-foreground">-</span>
      ),
  },
  {
    id: "source_outcome",
    accessorFn: (row) => row.source_error_type ?? row.source_reward,
    header: ({ column }) => (
      <div className="text-right">
        <SortableHeader column={column}>Source Reward</SortableHeader>
      </div>
    ),
    cell: ({ row }) => <SourceOutcomeCell item={row.original} />,
  },
  {
    accessorKey: "source_trial_name",
    header: ({ column }) => <SortableHeader column={column}>Source Trial</SortableHeader>,
    cell: ({ row }) => (
      <span className="font-mono text-sm">{row.original.source_trial_name}</span>
    ),
  },
  {
    accessorKey: "task_name",
    header: ({ column }) => <SortableHeader column={column}>Task</SortableHeader>,
    cell: ({ row }) => row.original.task_name ?? "-",
  },
  {
    accessorKey: "source",
    header: ({ column }) => <SortableHeader column={column}>Dataset</SortableHeader>,
    cell: ({ row }) => row.original.source ?? "-",
  },
  {
    accessorKey: "cost_usd",
    header: ({ column }) => (
      <div className="text-right">
        <SortableHeader column={column}>Critique Cost</SortableHeader>
      </div>
    ),
    cell: ({ row }) => (
      <div className="text-right font-mono tabular-nums">
        {formatCostUSD(row.original.cost_usd)}
      </div>
    ),
  },
  {
    accessorKey: "feedback",
    header: ({ column }) => <SortableHeader column={column}>Feedback</SortableHeader>,
    cell: ({ row }) => (
      <span className="block max-w-[34rem] truncate" title={row.original.feedback ?? ""}>
        {row.original.feedback ?? "-"}
      </span>
    ),
  },
  {
    accessorKey: "error_type",
    header: ({ column }) => <SortableHeader column={column}>Error</SortableHeader>,
    cell: ({ row }) => row.original.error_type ?? "-",
  },
  {
    accessorKey: "has_result_json",
    header: ({ column }) => (
      <div className="text-right">
        <SortableHeader column={column}>JSON</SortableHeader>
      </div>
    ),
    cell: ({ row }) => (
      <div className="text-right">{row.original.has_result_json ? "yes" : "-"}</div>
    ),
  },
  {
    accessorKey: "has_result_md",
    header: ({ column }) => (
      <div className="text-right">
        <SortableHeader column={column}>Markdown</SortableHeader>
      </div>
    ),
    cell: ({ row }) => (
      <div className="text-right">{row.original.has_result_md ? "yes" : "-"}</div>
    ),
  },
  {
    accessorKey: "started_at",
    header: ({ column }) => <SortableHeader column={column}>Started</SortableHeader>,
    cell: ({ row }) => formatDateTime(row.original.started_at),
  },
  {
    accessorKey: "finished_at",
    header: ({ column }) => <SortableHeader column={column}>Finished</SortableHeader>,
    cell: ({ row }) => formatDateTime(row.original.finished_at),
  },
];

type CritiqueHeatmapStat = "bad_rate" | "good_rate" | "n_items" | "n_errors";

const CRITIQUE_HEATMAP_STATS: { value: CritiqueHeatmapStat; label: string }[] = [
  { value: "bad_rate", label: "Bad Rate" },
  { value: "good_rate", label: "Good Rate" },
  { value: "n_items", label: "Items" },
  { value: "n_errors", label: "Errors" },
];

const SOURCE_TRIAL_FILTER_OPTIONS = [
  { value: "all", label: "All source trials" },
  { value: "non_errored", label: "Exclude errored" },
  { value: "errored", label: "Only errored" },
  { value: "successful", label: "Only successful" },
];

interface CritiqueSourceFilters {
  searchQuery: string | null;
  agentFilter: string[];
  providerFilter: string[];
  modelFilter: string[];
  sourceFilter: string[];
  taskFilter: string[];
}

function PaginationFooter({
  page,
  setPage,
  total,
  totalPages,
  noun,
}: {
  page: number;
  setPage: (updater: number | ((page: number) => number)) => void;
  total: number;
  totalPages: number;
  noun: string;
}) {
  if (totalPages <= 1) return null;

  return (
    <div className="grid grid-cols-3 items-center mt-4">
      <div className="text-sm text-muted-foreground">
        Showing {(page - 1) * PAGE_SIZE + 1}-{Math.min(page * PAGE_SIZE, total)} of{" "}
        {total} {noun}
      </div>
      <Pagination>
        <PaginationContent>
          <PaginationItem>
            <PaginationPrevious
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className={page === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
            />
          </PaginationItem>
          {page > 2 && (
            <PaginationItem>
              <PaginationLink onClick={() => setPage(1)} className="cursor-pointer">
                1
              </PaginationLink>
            </PaginationItem>
          )}
          {page > 3 && (
            <PaginationItem>
              <PaginationEllipsis />
            </PaginationItem>
          )}
          {page > 1 && (
            <PaginationItem>
              <PaginationLink
                onClick={() => setPage(page - 1)}
                className="cursor-pointer"
              >
                {page - 1}
              </PaginationLink>
            </PaginationItem>
          )}
          <PaginationItem>
            <PaginationLink isActive>{page}</PaginationLink>
          </PaginationItem>
          {page < totalPages && (
            <PaginationItem>
              <PaginationLink
                onClick={() => setPage(page + 1)}
                className="cursor-pointer"
              >
                {page + 1}
              </PaginationLink>
            </PaginationItem>
          )}
          {page < totalPages - 2 && (
            <PaginationItem>
              <PaginationEllipsis />
            </PaginationItem>
          )}
          {page < totalPages - 1 && (
            <PaginationItem>
              <PaginationLink
                onClick={() => setPage(totalPages)}
                className="cursor-pointer"
              >
                {totalPages}
              </PaginationLink>
            </PaginationItem>
          )}
          <PaginationItem>
            <PaginationNext
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className={
                page === totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"
              }
            />
          </PaginationItem>
        </PaginationContent>
      </Pagination>
      <div />
    </div>
  );
}

function getCritiqueHeatmapValue(
  cell: CritiqueHeatmapCell,
  stat: CritiqueHeatmapStat
): number | null {
  return cell[stat];
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatCritiqueHeatmapValue(
  cell: CritiqueHeatmapCell,
  stat: CritiqueHeatmapStat
): string {
  const value = getCritiqueHeatmapValue(cell, stat);
  if (value === null) return "-";
  if (stat === "bad_rate" || stat === "good_rate") return formatPercent(value);
  return Math.round(value).toLocaleString();
}

function CritiqueHeatmap({
  jobName,
  critiqueRunName,
  sourceFilters,
  ratingOptions,
  tagOptions,
  itemStateOptions,
}: {
  jobName: string;
  critiqueRunName: string;
  sourceFilters: CritiqueSourceFilters;
  ratingOptions: ComboboxOption[];
  tagOptions: ComboboxOption[];
  itemStateOptions: ComboboxOption[];
}) {
  const [ratingFilter, setRatingFilter] = useQueryState(
    "critique_heatmap_rating",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [tagFilter, setTagFilter] = useQueryState(
    "critique_heatmap_tag",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [itemStateFilter, setItemStateFilter] = useQueryState(
    "critique_heatmap_state",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [sourceTrialsRaw, setSourceTrials] = useQueryState(
    "critique_heatmap_source_trials",
    parseAsString.withDefault("all")
  );
  const [rowByRaw, setRowBy] = useQueryState(
    "critique_heatmap_row",
    parseAsString.withDefault("tag")
  );
  const [columnByRaw, setColumnBy] = useQueryState(
    "critique_heatmap_col",
    parseAsString.withDefault("task")
  );
  const [statRaw, setStat] = useQueryState(
    "critique_heatmap_stat",
    parseAsString.withDefault("n_items")
  );
  const rowBy: CritiqueHeatmapRowBy = rowByRaw === "tag" ? "tag" : "rating";
  const columnBy: CritiqueHeatmapColumnBy =
    columnByRaw === "dataset" ? "dataset" : "task";
  const sourceTrials: CritiqueHeatmapSourceTrialsFilter =
    sourceTrialsRaw === "non_errored" ||
    sourceTrialsRaw === "errored" ||
    sourceTrialsRaw === "successful"
      ? sourceTrialsRaw
      : "all";
  const stat: CritiqueHeatmapStat = CRITIQUE_HEATMAP_STATS.some(
    (option) => option.value === statRaw
  )
    ? (statRaw as CritiqueHeatmapStat)
    : "bad_rate";

  const { data, isLoading, isPlaceholderData } = useQuery({
    queryKey: [
      "critique-heatmap",
      jobName,
      critiqueRunName,
      rowBy,
      columnBy,
      sourceTrials,
      ratingFilter,
      tagFilter,
      itemStateFilter,
      sourceFilters.searchQuery,
      sourceFilters.agentFilter,
      sourceFilters.providerFilter,
      sourceFilters.modelFilter,
      sourceFilters.sourceFilter,
      sourceFilters.taskFilter,
    ],
    queryFn: () =>
      fetchCritiqueHeatmap(jobName, critiqueRunName, {
        rowBy,
        columnBy,
        sourceTrials,
        search: sourceFilters.searchQuery || undefined,
        agents:
          sourceFilters.agentFilter.length > 0
            ? sourceFilters.agentFilter
            : undefined,
        providers:
          sourceFilters.providerFilter.length > 0
            ? sourceFilters.providerFilter
            : undefined,
        models:
          sourceFilters.modelFilter.length > 0
            ? sourceFilters.modelFilter
            : undefined,
        sources:
          sourceFilters.sourceFilter.length > 0
            ? sourceFilters.sourceFilter
            : undefined,
        tasks:
          sourceFilters.taskFilter.length > 0
            ? sourceFilters.taskFilter
            : undefined,
        ratings: ratingFilter.length > 0 ? ratingFilter : undefined,
        tags: tagFilter.length > 0 ? tagFilter : undefined,
        statuses: itemStateFilter.length > 0 ? itemStateFilter : undefined,
      }),
    placeholderData: (previous) => previous,
  });

  const numericValues: number[] = [];
  if (data && stat !== "good_rate" && stat !== "bad_rate") {
    for (const rowCells of Object.values(data.cells)) {
      for (const cell of Object.values(rowCells)) {
        const value = getCritiqueHeatmapValue(cell, stat);
        if (value !== null) numericValues.push(value);
      }
    }
  }
  const maxValue = numericValues.length > 0 ? Math.max(...numericValues) : 1;
  const autoRowLabelWidth =
    data && data.rows.length > 0
      ? Math.min(
          Math.max(220, Math.max(...data.rows.map((row) => row.label.length)) * 7 + 32),
          480
        )
      : 220;
  const autoColHeaderHeight =
    data && data.columns.length > 0
      ? Math.min(
          Math.max(
            120,
            Math.max(...data.columns.map((column) => column.label.length)) * 6 +
              32
          ),
          400
        )
      : 144;

  const colorFor = (cell: CritiqueHeatmapCell): string => {
    const value = getCritiqueHeatmapValue(cell, stat);
    if (value === null) return "transparent";
    const intensity =
      stat === "bad_rate" || stat === "good_rate"
        ? value
        : maxValue > 0
          ? value / maxValue
          : 0;
    const percent = (intensity * 85 + 8).toFixed(1);
    if (stat === "bad_rate") {
      return `color-mix(in oklch, var(--destructive) ${percent}%, transparent)`;
    }
    if (stat === "good_rate") {
      return `color-mix(in oklch, oklch(0.62 0.18 145) ${percent}%, transparent)`;
    }
    return `color-mix(in oklch, var(--foreground) ${percent}%, transparent)`;
  };

  const axisLabel =
    rowBy === "tag" ? "tags by" : "ratings by";
  const columnLabel = columnBy === "dataset" ? "dataset" : "task";

  const renderControls = ({
    isColumnOrderCustom,
    resetColumnOrder,
  }: {
    isColumnOrderCustom: boolean;
    resetColumnOrder: () => void;
  }) => (
    <ChartToolbar
      description={
        <>
          Critique {axisLabel} {columnLabel}.
        </>
      }
    >
      <ChartToolbarSelect
        label="Rows"
        value={rowBy}
        onValueChange={(value) => setRowBy(value)}
        options={[
          { value: "rating", label: "Rating" },
          { value: "tag", label: "Tag" },
        ]}
      />
      <ChartToolbarSelect
        label="Columns"
        value={columnBy}
        onValueChange={(value) => setColumnBy(value)}
        options={[
          { value: "task", label: "Task" },
          { value: "dataset", label: "Dataset" },
        ]}
      />
      <ChartToolbarSelect
        label="Color by"
        value={stat}
        onValueChange={(value) => setStat(value)}
        options={CRITIQUE_HEATMAP_STATS.map((option) => ({
          value: option.value,
          label: option.label,
        }))}
      />
      <ChartToolbarSelect
        label="Source trials"
        value={sourceTrials}
        onValueChange={(value) => setSourceTrials(value === "all" ? null : value)}
        options={SOURCE_TRIAL_FILTER_OPTIONS}
      />
      <Combobox
        options={ratingOptions}
        value={ratingFilter}
        onValueChange={setRatingFilter}
        placeholder="All ratings"
        searchPlaceholder="Search ratings..."
        emptyText="No ratings found."
        className="h-8 w-40 rounded-md text-xs"
      />
      <Combobox
        options={tagOptions}
        value={tagFilter}
        onValueChange={setTagFilter}
        placeholder="All tags"
        searchPlaceholder="Search tags..."
        emptyText="No tags found."
        className="h-8 w-48 rounded-md text-xs"
      />
      <Combobox
        options={itemStateOptions}
        value={itemStateFilter}
        onValueChange={setItemStateFilter}
        placeholder="All states"
        searchPlaceholder="Search states..."
        emptyText="No states found."
        className="h-8 w-40 rounded-md text-xs"
      />
      {isColumnOrderCustom && (
        <ChartToolbarAction onClick={resetColumnOrder}>
          Reset order
        </ChartToolbarAction>
      )}
    </ChartToolbar>
  );

  return (
    <ResizableHeatmapGrid
      rows={data?.rows}
      columns={data?.columns}
      getCell={(row, column) => data?.cells[row.key]?.[column.key]}
      renderControls={renderControls}
      renderRowHeader={(row) => (
        <span
          className={cn(
            "whitespace-nowrap text-right font-mono text-xs",
            row.kind === "rating" &&
              row.value === "good" &&
              "text-green-700 dark:text-green-400",
            row.kind === "rating" && row.value === "bad" && "text-destructive"
          )}
        >
          {row.label}
        </span>
      )}
      renderCell={(row, column, cell) => {
        if (!cell) {
          return (
            <div
              key={`${row.key}-${column.key}`}
              className="h-16 border-r border-b bg-muted/20"
            />
          );
        }
        const value = getCritiqueHeatmapValue(cell, stat);
        const intensity =
          value === null
            ? 0
            : stat === "bad_rate" || stat === "good_rate"
              ? value
              : maxValue > 0
                ? value / maxValue
                : 0;

        return (
          <HoverCard key={`${row.key}-${column.key}`} openDelay={150}>
            <HoverCardTrigger asChild>
              <div className="relative isolate flex h-16 items-center justify-center border-r border-b text-xs">
                <div
                  className="absolute inset-0"
                  style={{ background: colorFor(cell) }}
                />
                <span
                  className={cn(
                    "relative z-10 max-w-24 truncate px-2 font-mono tabular-nums",
                    intensity > 0.6 ? "text-background" : "text-foreground"
                  )}
                >
                  {formatCritiqueHeatmapValue(cell, stat)}
                </span>
              </div>
            </HoverCardTrigger>
            <HoverCardContent className="w-72 text-xs">
              <div className="space-y-3">
                <div>
                  <div className="text-muted-foreground">Row</div>
                  <div>{row.label}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Column</div>
                  <div>{column.label}</div>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <div>
                    <div className="text-muted-foreground">Items</div>
                    <div className="font-mono">{cell.n_items}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Bad</div>
                    <div className="font-mono">
                      {cell.n_bad}
                      {cell.bad_rate !== null
                        ? ` (${formatPercent(cell.bad_rate)})`
                        : ""}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Good</div>
                    <div className="font-mono">
                      {cell.n_good}
                      {cell.good_rate !== null
                        ? ` (${formatPercent(cell.good_rate)})`
                        : ""}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Errors</div>
                    <div className="font-mono">{cell.n_errors}</div>
                  </div>
                </div>
                {Object.keys(cell.tag_counts).length > 0 && (
                  <div>
                    <div className="mb-1 text-muted-foreground">Tags</div>
                    <div className="space-y-1">
                      {Object.entries(cell.tag_counts).map(([tag, count]) => (
                        <div key={tag} className="flex justify-between gap-4">
                          <span className="truncate font-mono">{tag}</span>
                          <span className="font-mono">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </HoverCardContent>
          </HoverCard>
        );
      }}
      isLoading={isLoading}
      isFetching={isPlaceholderData}
      emptyTitle="No critique heat map cells"
      emptyDescription="No critique results are available for this run."
      storageKeyPrefix="pier.critiqueHeatmap"
      autoRowLabelWidth={autoRowLabelWidth}
      autoColumnHeaderHeight={autoColHeaderHeight}
    />
  );
}

function CritiqueDistributionTab({
  jobName,
  critiqueRunName,
  nItems,
  sourceFilters,
  runFinished,
}: {
  jobName: string;
  critiqueRunName: string;
  nItems: number;
  sourceFilters: CritiqueSourceFilters;
  runFinished: boolean;
}) {
  const [xModeRaw, setXMode] = useQueryState(
    "dist_x",
    parseAsString.withDefault("count")
  );
  const [yModeRaw, setYMode] = useQueryState(
    "dist_y",
    parseAsString.withDefault("count")
  );
  const [trialsFilterRaw, setTrialsFilter] = useQueryState(
    "dist_trials",
    parseAsString.withDefault("all")
  );
  const [layoutRaw, setLayout] = useQueryState(
    "dist_layout",
    parseAsString.withDefault("overlay")
  );
  const [styleRaw, setStyle] = useQueryState(
    "dist_style",
    parseAsString.withDefault("bars")
  );
  const [binCountMode, setBinCountMode] = useQueryState(
    "dist_bins",
    parseAsString.withDefault("auto")
  );
  const [showMeansRaw, setShowMeansRaw] = useQueryState(
    "dist_means",
    parseAsString.withDefault("0")
  );

  const xMode: CritiqueDistributionXMode =
    xModeRaw === "percent" ? "percent" : "count";
  const yMode: CritiqueDistributionYMode =
    yModeRaw === "percent" ? "percent" : "count";
  const trialsFilter: CritiqueDistributionTrialsFilter =
    trialsFilterRaw === "non_errored" ||
    trialsFilterRaw === "errored" ||
    trialsFilterRaw === "successful"
      ? trialsFilterRaw
      : "all";
  const layout: CritiqueDistributionLayout =
    layoutRaw === "grid" ? "grid" : "overlay";
  const style: CritiqueDistributionStyle =
    styleRaw === "step" ? "step" : "bars";
  const showMeans = showMeansRaw === "1";
  const { data: itemsData, isLoading } = useQuery({
    queryKey: [
      "critique-items-distribution",
      jobName,
      critiqueRunName,
      sourceFilters.searchQuery,
      sourceFilters.agentFilter,
      sourceFilters.providerFilter,
      sourceFilters.modelFilter,
      sourceFilters.sourceFilter,
      sourceFilters.taskFilter,
      trialsFilter,
      nItems,
    ],
    queryFn: () =>
      fetchCritiqueItems(jobName, critiqueRunName, 1, Math.max(nItems, PAGE_SIZE), {
        search: sourceFilters.searchQuery || undefined,
        agents:
          sourceFilters.agentFilter.length > 0
            ? sourceFilters.agentFilter
            : undefined,
        providers:
          sourceFilters.providerFilter.length > 0
            ? sourceFilters.providerFilter
            : undefined,
        models:
          sourceFilters.modelFilter.length > 0
            ? sourceFilters.modelFilter
            : undefined,
        sources:
          sourceFilters.sourceFilter.length > 0
            ? sourceFilters.sourceFilter
            : undefined,
        tasks:
          sourceFilters.taskFilter.length > 0
            ? sourceFilters.taskFilter
            : undefined,
        sourceTrials: trialsFilter,
      }),
    refetchInterval: runFinished ? false : 2000,
    placeholderData: keepPreviousData,
  });

  if (isLoading && !itemsData) {
    return (
      <Card className="rounded-none border-t-0">
        <CardContent className="py-10 text-sm text-muted-foreground">
          <LoadingDots />
        </CardContent>
      </Card>
    );
  }

  return (
    <CritiqueDistributionChart
      items={itemsData?.items ?? []}
      xMode={xMode}
      setXMode={(v) => setXMode(v === "count" ? null : v)}
      yMode={yMode}
      setYMode={(v) => setYMode(v === "count" ? null : v)}
      trialsFilter={trialsFilter}
      setTrialsFilter={(v) => setTrialsFilter(v === "all" ? null : v)}
      layout={layout}
      setLayout={(v) => setLayout(v === "overlay" ? null : v)}
      style={style}
      setStyle={(v) => setStyle(v === "bars" ? null : v)}
      binCountMode={binCountMode}
      setBinCountMode={(v) => setBinCountMode(v === "auto" ? null : v)}
      showMeans={showMeans}
      setShowMeans={(v) => setShowMeansRaw(v ? "1" : null)}
    />
  );
}

export default function CritiqueRun() {
  const { jobName, critiqueRunName } = useParams();
  const navigate = useNavigate();
  const [tabParam, setTabParam] = useQueryState(
    "tab",
    parseAsString.withDefault("items")
  );
  const [searchQuery, setSearchQuery] = useQueryState("q", parseAsString);
  const [agentFilter, setAgentFilter] = useQueryState(
    "agent",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [providerFilter, setProviderFilter] = useQueryState(
    "provider",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [modelFilter, setModelFilter] = useQueryState(
    "model",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [sourceFilter, setSourceFilter] = useQueryState(
    "source",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [taskFilter, setTaskFilter] = useQueryState(
    "task",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [hiddenColumns, setHiddenColumns] = useQueryState(
    "hide",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [sortBy, setSortBy] = useQueryState("sort_by", parseAsString);
  const [sortOrder, setSortOrder] = useQueryState(
    "sort_order",
    parseAsString.withDefault("asc")
  );
  const [page, setPage] = useQueryState(
    "page",
    parseAsInteger.withDefault(1)
  );
  const searchInputRef = useRef<HTMLInputElement>(null);
  const hasMountedRef = useRef(false);
  const activeTab =
    tabParam === "heatmap" ||
    tabParam === "distribution" ||
    tabParam === "raw" ||
    tabParam === "config" ||
    tabParam === "result"
      ? tabParam
      : "items";
  const visibleTab =
    activeTab === "config" || activeTab === "result" ? "raw" : activeTab;

  const { data, isLoading } = useQuery({
    queryKey: ["critique-run", jobName, critiqueRunName],
    queryFn: () => fetchCritiqueRun(jobName!, critiqueRunName!, false),
    enabled: !!jobName && !!critiqueRunName,
  });

  const sorting: SortingState = sortBy
    ? [{ id: sortBy, desc: sortOrder === "desc" }]
    : [];

  const handleSortingChange = (newSorting: SortingState) => {
    if (newSorting.length === 0) {
      setSortBy(null);
      setSortOrder(null);
    } else {
      setSortBy(newSorting[0].id);
      setSortOrder(newSorting[0].desc ? "desc" : "asc");
    }
  };

  const itemQueryFilters = {
    search: searchQuery || undefined,
    agents: agentFilter.length > 0 ? agentFilter : undefined,
    providers: providerFilter.length > 0 ? providerFilter : undefined,
    models: modelFilter.length > 0 ? modelFilter : undefined,
    sources: sourceFilter.length > 0 ? sourceFilter : undefined,
    tasks: taskFilter.length > 0 ? taskFilter : undefined,
    sortBy: sortBy || undefined,
    sortOrder: sortOrder as "asc" | "desc" | undefined,
  };

  const { data: filtersData } = useQuery({
    queryKey: ["critique-item-filters", jobName, critiqueRunName],
    queryFn: () => fetchCritiqueItemFilters(jobName!, critiqueRunName!),
    enabled: !!jobName && !!critiqueRunName,
  });

  const { data: itemsData, isLoading: itemsLoading } = useQuery({
    queryKey: [
      "critique-items",
      jobName,
      critiqueRunName,
      page,
      searchQuery,
      agentFilter,
      providerFilter,
      modelFilter,
      sourceFilter,
      taskFilter,
      sortBy,
      sortOrder,
    ],
    queryFn: () =>
      fetchCritiqueItems(
        jobName!,
        critiqueRunName!,
        page,
        PAGE_SIZE,
        itemQueryFilters
      ),
    enabled: !!jobName && !!critiqueRunName,
    refetchInterval: data?.run.finished_at ? false : 2000,
    placeholderData: keepPreviousData,
  });

  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;
      return;
    }
    setPage(1);
  }, [
    agentFilter,
    modelFilter,
    providerFilter,
    searchQuery,
    setPage,
    sortBy,
    sortOrder,
    sourceFilter,
    taskFilter,
  ]);

  const columnOptions: ComboboxOption[] = useMemo(
    () => [
      { value: "rating", label: "Rating" },
      { value: "status", label: "State" },
      { value: "tags", label: "Tags" },
      { value: "source_outcome", label: "Source Reward" },
      { value: "source_trial_name", label: "Source Trial" },
      { value: "task_name", label: "Task" },
      { value: "source", label: "Dataset" },
      { value: "cost_usd", label: "Critique Cost" },
      { value: "feedback", label: "Feedback" },
      { value: "error_type", label: "Error" },
      { value: "has_result_json", label: "JSON" },
      { value: "has_result_md", label: "Markdown" },
      { value: "started_at", label: "Started" },
      { value: "finished_at", label: "Finished" },
    ],
    []
  );

  const columnVisibility = useMemo(() => {
    const visibility: VisibilityState = {};
    for (const col of hiddenColumns) {
      visibility[col] = false;
    }
    return visibility;
  }, [hiddenColumns]);

  const visibleColumns = useMemo(() => {
    return columnOptions
      .filter((col) => !hiddenColumns.includes(col.value))
      .map((col) => col.value);
  }, [columnOptions, hiddenColumns]);

  const handleColumnVisibilityChange = (selectedValues: string[]) => {
    const newHidden = columnOptions
      .filter((col) => !selectedValues.includes(col.value))
      .map((col) => col.value);
    setHiddenColumns(newHidden.length > 0 ? newHidden : null);
  };

  useHotkeys(
    "mod+k",
    (e) => {
      e.preventDefault();
      searchInputRef.current?.focus();
    },
    { enableOnFormTags: true }
  );

  const agentOptions = useMemo<ComboboxOption[]>(
    () =>
      (filtersData?.agents ?? []).map((option) => ({
        value: option.value,
        label: option.value,
        count: option.count,
      })),
    [filtersData?.agents]
  );
  const providerOptions = useMemo<ComboboxOption[]>(
    () =>
      (filtersData?.providers ?? []).map((option) => ({
        value: option.value,
        label: option.value,
        count: option.count,
      })),
    [filtersData?.providers]
  );
  const modelOptions = useMemo<ComboboxOption[]>(
    () =>
      (filtersData?.models ?? []).map((option) => ({
        value: option.value,
        label: option.value,
        count: option.count,
      })),
    [filtersData?.models]
  );
  const sourceOptions = useMemo<ComboboxOption[]>(
    () =>
      (filtersData?.sources ?? []).map((option) => ({
        value: option.value,
        label: option.value,
        count: option.count,
      })),
    [filtersData?.sources]
  );
  const taskOptions = useMemo<ComboboxOption[]>(
    () =>
      (filtersData?.tasks ?? []).map((option) => ({
        value: option.value,
        label: option.value,
        count: option.count,
      })),
    [filtersData?.tasks]
  );
  const ratingOptions = useMemo<ComboboxOption[]>(() => {
    const labels: Record<string, string> = {
      bad: "Bad",
      good: "Good",
      none: "No rating",
    };
    return (filtersData?.ratings ?? []).map((option) => ({
      value: option.value,
      label: labels[option.value] ?? option.value,
      count: option.count,
    }));
  }, [filtersData?.ratings]);
  const tagOptions = useMemo<ComboboxOption[]>(
    () =>
      (filtersData?.tags ?? []).map((option) => ({
        value: option.value,
        label: option.value,
        count: option.count,
      })),
    [filtersData?.tags]
  );
  const itemStateOptions = useMemo<ComboboxOption[]>(
    () =>
      (filtersData?.statuses ?? []).map((option) => ({
        value: option.value,
        label: formatCritiqueStatus(option.value),
        count: option.count,
      })),
    [filtersData?.statuses]
  );
  const items = itemsData?.items ?? [];
  const totalItems = itemsData?.total ?? 0;
  const totalPages = itemsData?.total_pages ?? 0;
  const safePage = Math.min(Math.max(page, 1), Math.max(totalPages, 1));
  const sourceFilters: CritiqueSourceFilters = {
    searchQuery,
    agentFilter,
    providerFilter,
    modelFilter,
    sourceFilter,
    taskFilter,
  };

  if (isLoading) {
    return (
      <div className="px-4 py-10">
        <Card>
          <CardHeader>
            <CardTitle>Critique Job</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground"><LoadingDots /></div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="px-4 py-10">
        <Empty className="bg-card border">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <FileText />
            </EmptyMedia>
            <EmptyTitle>Critique job not found</EmptyTitle>
            <EmptyDescription>
              The requested critique job could not be loaded.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    );
  }

  const run = data.run;
  const renderFilterControls = (showColumns: boolean) => (
    <div className={cn("grid -mb-px", showColumns ? "grid-cols-8" : "grid-cols-7")}>
      <div className="col-span-2 relative">
        <Input
          ref={searchInputRef}
          placeholder="Search critique items..."
          value={searchQuery ?? ""}
          onChange={(e) => setSearchQuery(e.target.value || null)}
          size="lg"
          variant="card"
          className="peer pl-9 pr-16 shadow-none"
        />
        <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-border transition-colors peer-focus-visible:text-ring" />
        {searchQuery ? (
          <button
            type="button"
            onClick={() => setSearchQuery(null)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        ) : (
          <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
            <Kbd>⌘</Kbd>
            <Kbd>K</Kbd>
          </div>
        )}
      </div>
      <Combobox
        options={agentOptions}
        value={agentFilter}
        onValueChange={setAgentFilter}
        placeholder="All source agents"
        searchPlaceholder="Search agents..."
        emptyText="No agents found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={providerOptions}
        value={providerFilter}
        onValueChange={setProviderFilter}
        placeholder="All source providers"
        searchPlaceholder="Search providers..."
        emptyText="No providers found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={modelOptions}
        value={modelFilter}
        onValueChange={setModelFilter}
        placeholder="All source models"
        searchPlaceholder="Search models..."
        emptyText="No models found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={sourceOptions}
        value={sourceFilter}
        onValueChange={setSourceFilter}
        placeholder="All datasets"
        searchPlaceholder="Search datasets..."
        emptyText="No datasets found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={taskOptions}
        value={taskFilter}
        onValueChange={setTaskFilter}
        placeholder="All tasks"
        searchPlaceholder="Search tasks..."
        emptyText="No tasks found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      {showColumns && (
        <Combobox
          options={columnOptions}
          value={visibleColumns}
          onValueChange={handleColumnVisibilityChange}
          placeholder="Columns"
          searchPlaceholder="Search columns..."
          emptyText="No columns."
          variant="card"
          className="w-full border-l-0 shadow-none"
          multiSelectLabel="columns"
        />
      )}
    </div>
  );

  return (
    <div className="px-4 py-10">
      <div className="mb-8">
        <Breadcrumb className="mb-4">
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link to="/">Jobs</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link to={`/jobs/${encodeURIComponent(jobName!)}`}>{jobName}</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link to={`/jobs/${encodeURIComponent(jobName!)}/critiques`}>
                  Critiques
                </Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>{critiqueRunName}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <div className="flex flex-col gap-4 min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-4xl font-normal tracking-tighter font-mono truncate">
              {critiqueRunName}
            </h1>
            <CritiqueStatusBadge status={run.status} />
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground">
            <span>
              {run.n_completed_items}/{run.n_items} items completed
            </span>
            <span>{run.n_failed_items} failures</span>
            {run.n_running_items > 0 && <span>{run.n_running_items} running</span>}
            {run.n_pending_items > 0 && <span>{run.n_pending_items} pending</span>}
            {run.n_missing_items > 0 && <span>{run.n_missing_items} missing</span>}
            <span>{run.agent_name ?? "-"} / {run.model_name ?? "-"}</span>
            <span>{run.environment_type ?? "-"}</span>
          </div>
        </div>
      </div>

      <Tabs
        value={visibleTab}
        onValueChange={(value) => setTabParam(value === "items" ? null : value)}
      >
        <TabsList className="bg-card border border-b-0 w-full">
          <TabsTrigger value="items">Items</TabsTrigger>
          <TabsTrigger value="heatmap">Heat Map</TabsTrigger>
          <TabsTrigger value="distribution">Distribution</TabsTrigger>
          <TabsTrigger value="raw">Raw</TabsTrigger>
        </TabsList>
        <TabsContent value="items" className="mt-0">
          {renderFilterControls(true)}
          {itemsLoading && !itemsData ? (
            <Card className="rounded-none border-t-0">
              <CardContent className="py-10 text-sm text-muted-foreground">
                <LoadingDots />
              </CardContent>
            </Card>
          ) : (
            <>
              <DataTable
                columns={columns}
                data={items}
                onRowClick={(item) => {
                  const url = trialUrl(jobName!, critiqueRunName!, item);
                  if (url) navigate(url);
                }}
                className="border-t-0"
                columnVisibility={columnVisibility}
                sorting={sorting}
                onSortingChange={handleSortingChange}
                manualSorting
                emptyState={
                  <Empty>
                    <EmptyHeader>
                      <EmptyMedia variant="icon">
                        <FileText />
                      </EmptyMedia>
                      <EmptyTitle>
                        {data.run.n_items === 0
                          ? "No critique items"
                          : "No critique items match those filters"}
                      </EmptyTitle>
                      <EmptyDescription>
                        {data.run.n_items === 0
                          ? "This critique job has not created any item directories yet."
                          : "Clear one or more item filters to broaden the result set."}
                      </EmptyDescription>
                    </EmptyHeader>
                  </Empty>
                }
              />
              <PaginationFooter
                page={safePage}
                setPage={setPage}
                total={totalItems}
                totalPages={totalPages}
                noun="items"
              />
            </>
          )}
        </TabsContent>
        <TabsContent value="heatmap" className="mt-0">
          {renderFilterControls(false)}
          <CritiqueHeatmap
            jobName={jobName!}
            critiqueRunName={critiqueRunName!}
            sourceFilters={sourceFilters}
            ratingOptions={ratingOptions}
            tagOptions={tagOptions}
            itemStateOptions={itemStateOptions}
          />
        </TabsContent>
        <TabsContent value="distribution" className="mt-0">
          {renderFilterControls(false)}
          <CritiqueDistributionTab
            jobName={jobName!}
            critiqueRunName={critiqueRunName!}
            nItems={run.n_items}
            sourceFilters={sourceFilters}
            runFinished={run.finished_at !== null}
          />
        </TabsContent>
        <TabsContent value="raw" className="mt-0 -mx-px">
          <div className="border bg-card">
            <div className="border-b px-4 py-3 text-sm font-medium">Config</div>
            <CodeBlock code={JSON.stringify(data.config ?? {}, null, 2)} lang="json" />
            <div className="border-y px-4 py-3 text-sm font-medium">Result</div>
            <CodeBlock code={JSON.stringify(data.result ?? {}, null, 2)} lang="json" />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
