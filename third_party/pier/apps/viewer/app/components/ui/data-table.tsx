//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import {
  type Column,
  type ColumnDef,
  type ColumnSizingState,
  type RowSelectionState,
  type SortingState,
  type VisibilityState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown, Info } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "~/components/ui/button";
import { Checkbox } from "~/components/ui/checkbox";
import { IndeterminateBar } from "~/components/ui/indeterminate-bar";
import { ScrollArea, ScrollBar } from "~/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/lib/utils";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";

type SortableHeaderProps<TData, TValue> = {
  column: Column<TData, TValue>;
  children: React.ReactNode;
} & Omit<React.ComponentProps<typeof Button>, "onClick" | "children">;

export function SortableHeader<TData, TValue>({
  column,
  children,
  className,
  ...rest
}: SortableHeaderProps<TData, TValue>) {
  const sorted = column.getIsSorted();
  return (
    <Button
      variant="ghost"
      size="sm"
      className={cn("-ml-3 h-8", className)}
      onClick={() => column.toggleSorting(sorted === "asc")}
      {...rest}
    >
      {children}
      {sorted === "asc" ? (
        <ArrowUp className="ml-2 h-4 w-4" />
      ) : sorted === "desc" ? (
        <ArrowDown className="ml-2 h-4 w-4" />
      ) : (
        <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
      )}
    </Button>
  );
}

/**
 * Small info-icon tooltip suitable for placement next to a column header
 * label. Renders a span (not a button) so it can sit alongside
 * `SortableHeader`'s button without violating button-in-button HTML.
 */
export function HeaderInfo({
  children,
  side = "top",
  className,
}: {
  children: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          aria-label="More info"
          className={cn(
            "inline-flex shrink-0 items-center justify-center text-muted-foreground/60 hover:text-foreground focus-visible:text-foreground cursor-help transition-colors outline-none",
            className
          )}
          onClick={(e) => e.stopPropagation()}
        >
          <Info className="h-3.5 w-3.5" />
        </span>
      </TooltipTrigger>
      <TooltipContent side={side} className="max-w-xs">
        <p className="text-xs leading-snug">{children}</p>
      </TooltipContent>
    </Tooltip>
  );
}

export function createSelectColumn<TData>(): ColumnDef<TData> {
  return {
    id: "select",
    header: ({ table }) => (
      <Checkbox
        checked={
          table.getIsAllPageRowsSelected() ||
          (table.getIsSomePageRowsSelected() && "indeterminate")
        }
        onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
        aria-label="Select all"
      />
    ),
    cell: ({ row }) => (
      <Checkbox
        checked={row.getIsSelected()}
        onCheckedChange={(value) => row.toggleSelected(!!value)}
        onClick={(e) => e.stopPropagation()}
        aria-label="Select row"
      />
    ),
    enableSorting: false,
    enableHiding: false,
  };
}

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  onRowClick?: (row: TData) => void;
  getRowStyle?: (row: TData) => React.CSSProperties | undefined;
  enableRowSelection?: boolean;
  onSelectionChange?: (selectedRows: TData[]) => void;
  rowSelection?: RowSelectionState;
  onRowSelectionChange?: (selection: RowSelectionState) => void;
  columnVisibility?: VisibilityState;
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
  sorting?: SortingState;
  onSortingChange?: (sorting: SortingState) => void;
  manualSorting?: boolean;
  getRowId?: (row: TData) => string;
  isLoading?: boolean;
  /**
   * Renders a thin indeterminate progress bar at the top of the table and
   * dims the body to indicate the rows on screen are stale. Ignored when
   * there are no rows yet — `isLoading` handles the empty initial state.
   *
   * Pass React Query's `isFetching` for normal queries, or
   * `isPlaceholderData` for queries that also use `keepPreviousData` with a
   * `refetchInterval` (so silent background polls don't flash the bar).
   */
  isFetching?: boolean;
  emptyState?: React.ReactNode;
  className?: string;
  highlightedIndex?: number;
  enableDragSelect?: boolean;
  selectedIndices?: Set<number>;
  onSelectedIndicesChange?: (indices: Set<number>) => void;
  onDragStart?: (startIndex: number) => void;
  /**
   * Enable user-resizable columns with truncation. When enabled, the table
   * switches to fixed layout, columns honor their `size`/`minSize`/`maxSize`
   * from the column def, and a resize handle appears on the right edge of
   * each header. Cell content is truncated with an ellipsis when it exceeds
   * the column width. Defaults to `false` for backwards compatibility.
   */
  enableColumnResizing?: boolean;
  /**
   * Stable identifier used to persist user-adjusted column widths across
   * sessions in localStorage. Only used when `enableColumnResizing` is true.
   */
  tableId?: string;
}

const COLUMN_SIZING_STORAGE_PREFIX = "pier.dataTable.colSizing.";

function readStoredColumnSizing(key: string): ColumnSizingState | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    const entries = Object.entries(parsed).filter(
      ([, v]) => typeof v === "number" && Number.isFinite(v)
    );
    return Object.fromEntries(entries) as ColumnSizingState;
  } catch {
    return null;
  }
}

export function DataTable<TData, TValue>({
  columns,
  data,
  onRowClick,
  getRowStyle,
  enableRowSelection = false,
  onSelectionChange,
  rowSelection: controlledRowSelection,
  onRowSelectionChange,
  columnVisibility: controlledColumnVisibility,
  onColumnVisibilityChange,
  sorting: controlledSorting,
  onSortingChange,
  manualSorting = false,
  getRowId,
  isLoading = false,
  isFetching = false,
  emptyState,
  className,
  highlightedIndex,
  enableDragSelect = false,
  selectedIndices: controlledSelectedIndices,
  onSelectedIndicesChange,
  onDragStart,
  enableColumnResizing = false,
  tableId,
}: DataTableProps<TData, TValue>) {
  const [internalRowSelection, setInternalRowSelection] =
    useState<RowSelectionState>({});
  const [internalColumnVisibility, setInternalColumnVisibility] =
    useState<VisibilityState>({});
  const [internalSorting, setInternalSorting] = useState<SortingState>([]);
  const [internalSelectedIndices, setInternalSelectedIndices] = useState<Set<number>>(new Set());

  const storageKey = useMemo(
    () =>
      enableColumnResizing && tableId
        ? COLUMN_SIZING_STORAGE_PREFIX + tableId
        : null,
    [enableColumnResizing, tableId]
  );
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>(() =>
    storageKey ? readStoredColumnSizing(storageKey) ?? {} : {}
  );

  // Drag select refs
  const dragStartIndex = useRef<number | null>(null);
  const didDragRef = useRef(false);

  const selectedIndices = controlledSelectedIndices ?? internalSelectedIndices;
  const setSelectedIndices = onSelectedIndicesChange ?? setInternalSelectedIndices;

  const handleRowMouseDown = useCallback((_rowIndex: number, e: React.MouseEvent) => {
    if (!enableDragSelect || e.button !== 0) return;
    if ((e.target as HTMLElement).closest('[role="checkbox"]')) return;
    dragStartIndex.current = _rowIndex;
    didDragRef.current = false;
    onDragStart?.(_rowIndex);
  }, [enableDragSelect, onDragStart]);

  const handleRowMouseEnter = useCallback((rowIndex: number) => {
    if (dragStartIndex.current === null) return;
    if (rowIndex === dragStartIndex.current && !didDragRef.current) return;
    // First move: prevent text selection for the rest of this drag
    if (!didDragRef.current) {
      didDragRef.current = true;
      window.getSelection()?.removeAllRanges();
    }
    const min = Math.min(dragStartIndex.current, rowIndex);
    const max = Math.max(dragStartIndex.current, rowIndex);
    const indices = new Set<number>();
    for (let i = min; i <= max; i++) {
      indices.add(i);
    }
    setSelectedIndices(indices);
  }, [setSelectedIndices]);

  // Prevent text selection while dragging & clear drag on mouseup
  useEffect(() => {
    if (!enableDragSelect) return;
    const onSelectStart = (e: Event) => {
      if (didDragRef.current) e.preventDefault();
    };
    const onMouseUp = () => { dragStartIndex.current = null; };
    document.addEventListener("selectstart", onSelectStart);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      document.removeEventListener("selectstart", onSelectStart);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [enableDragSelect]);

  const rowSelection = controlledRowSelection ?? internalRowSelection;
  const setRowSelection = onRowSelectionChange ?? setInternalRowSelection;
  const columnVisibility = controlledColumnVisibility ?? internalColumnVisibility;
  const setColumnVisibility = onColumnVisibilityChange ?? setInternalColumnVisibility;
  const sorting = controlledSorting ?? internalSorting;
  const setSorting = onSortingChange ?? setInternalSorting;

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: manualSorting ? undefined : getSortedRowModel(),
    manualSorting,
    enableRowSelection,
    enableColumnResizing,
    columnResizeMode: "onChange",
    // No `maxSize` so columns can be dragged arbitrarily wide by default.
    // Individual columns can still opt into a cap via `maxSize` in their
    // ColumnDef (e.g. the long Task hash column).
    defaultColumn: enableColumnResizing
      ? { minSize: 60, size: 160 }
      : undefined,
    onRowSelectionChange: (updaterOrValue) => {
      const newSelection =
        typeof updaterOrValue === "function"
          ? updaterOrValue(rowSelection)
          : updaterOrValue;
      setRowSelection(newSelection);
      if (onSelectionChange) {
        const selectedRows = Object.keys(newSelection)
          .filter((key) => newSelection[key])
          .map((key) => data[parseInt(key)]);
        onSelectionChange(selectedRows);
      }
    },
    onColumnVisibilityChange: (updaterOrValue) => {
      const newVisibility =
        typeof updaterOrValue === "function"
          ? updaterOrValue(columnVisibility)
          : updaterOrValue;
      setColumnVisibility(newVisibility);
    },
    onSortingChange: (updaterOrValue) => {
      const newSorting =
        typeof updaterOrValue === "function"
          ? updaterOrValue(sorting)
          : updaterOrValue;
      setSorting(newSorting);
    },
    onColumnSizingChange: (updaterOrValue) => {
      setColumnSizing((prev) => {
        const next =
          typeof updaterOrValue === "function"
            ? updaterOrValue(prev)
            : updaterOrValue;
        if (storageKey) {
          try {
            window.localStorage.setItem(storageKey, JSON.stringify(next));
          } catch {}
        }
        return next;
      });
    },
    state: {
      rowSelection,
      columnVisibility,
      sorting,
      columnSizing,
    },
    getRowId,
  });

  const isResizingColumn = !!table.getState().columnSizingInfo.isResizingColumn;

  const isBusy = isLoading || isFetching;
  const showStaleDim = isFetching && data.length > 0;

  return (
    <div className={cn("relative border bg-card", className)}>
      {/*
       * Top-of-table separator. Always rendered so the line above the
       * column headers stays visible even when the consumer overrides the
       * wrapper border (e.g. `border-t-0` to butt up against a filter row).
       */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 z-10 h-px bg-border"
      />
      {/*
       * Loading / refetch indicator. Sits over the existing 1px header
       * bottom border (between header and body) so the line stays exactly
       * 1px thick at rest; the 2px-thick bar centers on that border while
       * animating, no layout impact. Renders for both initial load and
       * background refetch — there's no separate "Loading..." placeholder.
       */}
      {isBusy && (
        <IndeterminateBar style={{ top: "calc(3rem - 1px)" }} />
      )}
      <ScrollArea className="size-full">
        <div
          className="relative"
          style={
            enableColumnResizing
              ? { width: table.getCenterTotalSize() }
              : undefined
          }
        >
        <Table
          style={
            enableColumnResizing
              ? {
                  width: table.getCenterTotalSize(),
                  tableLayout: "fixed",
                }
              : undefined
          }
        >
          {enableColumnResizing && (
            <colgroup>
              {table.getVisibleLeafColumns().map((col) => (
                <col key={col.id} style={{ width: col.getSize() }} />
              ))}
            </colgroup>
          )}
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead
                      key={header.id}
                      className={cn(
                        enableColumnResizing &&
                          "relative overflow-hidden truncate"
                      )}
                    >
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody
            aria-busy={isBusy}
            className={cn(
              "transition-opacity",
              showStaleDim && "opacity-60",
              isResizingColumn && "select-none"
            )}
          >
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row, rowIndex) => {
                const isSelected = selectedIndices.has(rowIndex);
                return (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && "selected"}
                    onClick={() => {
                      if (didDragRef.current) return;
                      onRowClick?.(row.original);
                    }}
                    onMouseDown={(e) => handleRowMouseDown(rowIndex, e)}
                    onMouseEnter={() => handleRowMouseEnter(rowIndex)}
                    className={cn(
                      onRowClick && "cursor-pointer",
                      rowIndex === highlightedIndex && "bg-muted",
                    )}
                    style={getRowStyle?.(row.original)}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        className={cn(
                          enableColumnResizing &&
                            "overflow-hidden text-ellipsis"
                        )}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center text-muted-foreground"
                >
                  {isBusy ? null : emptyState ?? "No results."}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        {enableColumnResizing && (() => {
          const headers = table.getHeaderGroups()[0]?.headers ?? [];
          let leftPx = 0;
          return headers.map((header, index) => {
            const size = header.column.getSize();
            leftPx += size;
            // No handle past the last column.
            if (index === headers.length - 1) return null;
            if (!header.column.getCanResize()) return null;
            const isResizing = header.column.getIsResizing();
            const boundary = leftPx;
            return (
              <div
                key={`resize-${header.id}`}
                role="separator"
                aria-orientation="vertical"
                aria-label="Resize column (double-click to reset)"
                onMouseDown={header.getResizeHandler()}
                onTouchStart={header.getResizeHandler()}
                onDoubleClick={() => header.column.resetSize()}
                className={cn(
                  "group/resize absolute top-0 bottom-0 z-20 w-2 -translate-x-1/2 cursor-col-resize select-none touch-none"
                )}
                style={{ left: boundary }}
              >
                <div
                  className={cn(
                    "pointer-events-none absolute inset-y-0 left-1/2 -translate-x-1/2 transition-all duration-150",
                    isResizing
                      ? "w-[2px] bg-ring"
                      : "w-px bg-transparent group-hover/resize:w-[2px] group-hover/resize:bg-ring/80"
                  )}
                />
              </div>
            );
          });
        })()}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
