//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { FileText, Search, X } from "lucide-react";
import { useEffect, useRef } from "react";
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
import {
  DataTable,
  SortableHeader,
} from "~/components/ui/data-table";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty";
import { Input } from "~/components/ui/input";
import { Kbd } from "~/components/ui/kbd";
import { Combobox, type ComboboxOption } from "~/components/ui/combobox";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "~/components/ui/pagination";
import { fetchCritiqueRuns } from "~/lib/api";
import { useDebouncedValue, useKeyboardTableNavigation } from "~/lib/hooks";
import type { CritiqueRunSummary } from "~/lib/types";

const PAGE_SIZE = 100;
const STATUS_OPTIONS: ComboboxOption[] = [
  { value: "pending", label: "Pending" },
  { value: "running", label: "Running" },
  { value: "completed", label: "Completed" },
];
const FAILURE_OPTIONS: ComboboxOption[] = [
  { value: "failed", label: "With failures" },
];

function formatDateTime(date: string | null): string {
  if (!date) return "-";
  return new Date(date).toLocaleString();
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

function agentModel(run: CritiqueRunSummary): string {
  const model = run.model_provider
    ? `${run.model_provider}/${run.model_name ?? ""}`
    : run.model_name;
  if (run.agent_name && model) return `${run.agent_name} / ${model}`;
  return run.agent_name ?? model ?? "-";
}

function PaginationFooter({
  page,
  setPage,
  total,
  totalPages,
}: {
  page: number;
  setPage: (updater: number | ((page: number) => number)) => void;
  total: number;
  totalPages: number;
}) {
  if (totalPages <= 1) return null;

  return (
    <div className="grid grid-cols-3 items-center mt-4">
      <div className="text-sm text-muted-foreground">
        Showing {(page - 1) * PAGE_SIZE + 1}-{Math.min(page * PAGE_SIZE, total)} of{" "}
        {total} critique jobs
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

const columns: ColumnDef<CritiqueRunSummary>[] = [
  {
    accessorKey: "status",
    header: ({ column }) => <SortableHeader column={column}>State</SortableHeader>,
    cell: ({ row }) => <CritiqueStatusBadge status={row.original.status} />,
  },
  {
    accessorKey: "name",
    header: ({ column }) => <SortableHeader column={column}>Critique Job</SortableHeader>,
    cell: ({ row }) => (
      <span className="font-mono text-xs">{row.original.name}</span>
    ),
  },
  {
    id: "agent_model",
    accessorFn: agentModel,
    header: ({ column }) => <SortableHeader column={column}>Agent / Model</SortableHeader>,
    cell: ({ row }) => agentModel(row.original),
  },
  {
    accessorKey: "environment_type",
    header: ({ column }) => <SortableHeader column={column}>Environment</SortableHeader>,
    cell: ({ row }) => row.original.environment_type ?? "-",
  },
  {
    accessorKey: "n_items",
    header: ({ column }) => (
      <div className="text-right">
        <SortableHeader column={column}>Items</SortableHeader>
      </div>
    ),
    cell: ({ row }) => {
      const run = row.original;
      return (
        <div className="text-right">
          {run.n_completed_items}/{run.n_items}
        </div>
      );
    },
  },
  {
    accessorKey: "n_failed_items",
    header: ({ column }) => (
      <div className="text-right">
        <SortableHeader column={column}>Failures</SortableHeader>
      </div>
    ),
    cell: ({ row }) => (
      <div className="text-right">{row.original.n_failed_items}</div>
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

export default function Critiques() {
  const { jobName } = useParams();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useQueryState("q", parseAsString);
  const [statusFilter, setStatusFilter] = useQueryState(
    "status",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [failureFilter, setFailureFilter] = useQueryState(
    "failures",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [page, setPage] = useQueryState(
    "page",
    parseAsInteger.withDefault(1)
  );
  const hasMountedRef = useRef(false);
  const debouncedSearch = useDebouncedValue(searchQuery ?? "", 300);
  const hasFailures = failureFilter.includes("failed");

  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;
      return;
    }
    setPage(1);
  }, [debouncedSearch, hasFailures, setPage, statusFilter]);

  const { data, isLoading } = useQuery({
    queryKey: [
      "critique-runs",
      jobName,
      page,
      debouncedSearch,
      statusFilter,
      hasFailures,
    ],
    queryFn: () =>
      fetchCritiqueRuns(jobName!, page, PAGE_SIZE, {
        search: debouncedSearch || undefined,
        statuses: statusFilter.length > 0 ? statusFilter : undefined,
        hasFailures,
      }),
    enabled: !!jobName,
    placeholderData: keepPreviousData,
  });

  const runs = data?.items ?? [];
  const totalPages = data?.total_pages ?? 0;
  const total = data?.total ?? 0;
  const safePage = Math.min(Math.max(page, 1), Math.max(totalPages, 1));
  const runningRuns = runs.filter((run) => run.status === "running").length;
  const failedRuns = runs.filter((run) => run.n_failed_items > 0).length;
  const { highlightedIndex } = useKeyboardTableNavigation({
    rows: runs,
    onNavigate: (run) =>
      navigate(
        `/jobs/${encodeURIComponent(jobName!)}/critiques/${encodeURIComponent(run.name)}`
      ),
    onEscapeUnhighlighted: () => navigate(`/jobs/${encodeURIComponent(jobName!)}`),
  });

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
              <BreadcrumbPage>Critiques</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <div className="flex flex-col xl:flex-row xl:justify-between gap-4">
          <div className="flex flex-col gap-4 justify-between min-w-0">
            <h1 className="text-4xl font-normal tracking-tighter font-mono">
              Critiques
            </h1>
            <div className="text-sm text-muted-foreground">
              {data?.total ?? 0} critique jobs
              {runningRuns > 0 && <> | {runningRuns} running</>}
              {failedRuns > 0 && <> | {failedRuns} with failures</>}
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground whitespace-nowrap mt-auto">
            <span className="flex items-center gap-1">
              <Kbd>j</Kbd>
              <Kbd>k</Kbd>
              <span>navigate</span>
            </span>
            <span className="flex items-center gap-1">
              <Kbd>Enter</Kbd>
              <span>open</span>
            </span>
            <span className="flex items-center gap-1">
              <Kbd>Esc</Kbd>
              <span>{highlightedIndex >= 0 ? "deselect" : "go back"}</span>
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 -mb-px">
        <div className="col-span-2 relative">
          <Input
            placeholder="Search critiques..."
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
          options={STATUS_OPTIONS}
          value={statusFilter}
          onValueChange={setStatusFilter}
          placeholder="All states"
          searchPlaceholder="Search states..."
          emptyText="No states found."
          variant="card"
          className="w-full border-l-0 shadow-none"
        />
        <Combobox
          options={FAILURE_OPTIONS}
          value={failureFilter}
          onValueChange={setFailureFilter}
          placeholder="All outcomes"
          searchPlaceholder="Search outcomes..."
          emptyText="No outcomes found."
          variant="card"
          className="w-full border-l-0 shadow-none"
        />
      </div>

      <DataTable
        columns={columns}
        data={runs}
        onRowClick={(run) =>
          navigate(
            `/jobs/${encodeURIComponent(jobName!)}/critiques/${encodeURIComponent(run.name)}`
          )
        }
        isLoading={isLoading}
        highlightedIndex={highlightedIndex}
        emptyState={
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <FileText />
              </EmptyMedia>
              <EmptyTitle>No critique jobs</EmptyTitle>
              <EmptyDescription>
                This job does not have a .critiques directory with runs yet.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        }
      />
      <PaginationFooter
        page={safePage}
        setPage={setPage}
        total={total}
        totalPages={totalPages}
      />
    </div>
  );
}
