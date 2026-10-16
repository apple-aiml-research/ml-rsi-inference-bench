//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { GripVertical, Search } from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent as ReactDragEvent,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from "react";

import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { IndeterminateBar } from "~/components/ui/indeterminate-bar";
import { cn } from "~/lib/utils";

const ROW_WIDTH_MIN = 80;
const ROW_WIDTH_MAX = 800;
const COL_HEIGHT_MIN = 40;
const COL_HEIGHT_MAX = 600;

export interface HeatmapGridColumn {
  key: string;
  label: string;
}

export interface HeatmapGridRow {
  key: string;
  label: string;
}

export interface HeatmapGridControlsState {
  isColumnOrderCustom: boolean;
  resetColumnOrder: () => void;
}

function arraysEqual<T>(a: T[], b: T[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function reconcileColumnOrder(
  currentOrder: string[],
  defaultOrder: string[]
): string[] {
  if (defaultOrder.length === 0) return [];
  if (currentOrder.length === 0) return defaultOrder;

  const defaultKeys = new Set(defaultOrder);
  const keptKeys = currentOrder.filter((key) => defaultKeys.has(key));
  const keptKeySet = new Set(keptKeys);
  const newKeys = defaultOrder.filter((key) => !keptKeySet.has(key));
  return [...keptKeys, ...newKeys];
}

function readStoredSize(key: string): number | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(key);
  if (!raw) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

export function ResizableHeatmapGrid<
  TRow extends HeatmapGridRow,
  TColumn extends HeatmapGridColumn,
  TCell,
>({
  rows,
  columns,
  getCell,
  renderControls,
  renderRowHeader,
  renderCell,
  renderColumnLabel,
  renderColumnTooltip,
  isLoading,
  isFetching = false,
  emptyTitle = "No heat map cells",
  emptyDescription = "No cells match the current filters.",
  storageKeyPrefix,
  autoRowLabelWidth,
  autoColumnHeaderHeight,
  rowHeight = 64,
  minColumnWidth = 72,
}: {
  rows: TRow[] | undefined;
  columns: TColumn[] | undefined;
  getCell: (row: TRow, column: TColumn) => TCell | undefined;
  renderControls: (state: HeatmapGridControlsState) => ReactNode;
  renderRowHeader: (row: TRow) => ReactNode;
  renderCell: (
    row: TRow,
    column: TColumn,
    cell: TCell | undefined
  ) => ReactNode;
  renderColumnLabel?: (column: TColumn) => ReactNode;
  renderColumnTooltip?: (column: TColumn) => ReactNode;
  isLoading: boolean;
  isFetching?: boolean;
  emptyTitle?: string;
  emptyDescription?: string;
  storageKeyPrefix: string;
  autoRowLabelWidth: number;
  autoColumnHeaderHeight: number;
  rowHeight?: number;
  minColumnWidth?: number;
}) {
  const rowWidthStorageKey = `${storageKeyPrefix}.rowLabelWidth`;
  const colHeightStorageKey = `${storageKeyPrefix}.colHeaderHeight`;
  const [rowLabelWidthOverride, setRowLabelWidthOverride] = useState<
    number | null
  >(() => readStoredSize(rowWidthStorageKey));
  const [colHeaderHeightOverride, setColHeaderHeightOverride] = useState<
    number | null
  >(() => readStoredSize(colHeightStorageKey));
  const [draggingKind, setDraggingKind] = useState<"row" | "col" | null>(null);
  const [columnOrder, setColumnOrder] = useState<string[]>([]);
  const [draggingColumnKey, setDraggingColumnKey] = useState<string | null>(
    null
  );

  const defaultColumnOrder = useMemo(
    () => columns?.map((column) => column.key) ?? [],
    [columns]
  );

  useEffect(() => {
    setColumnOrder((currentOrder) => {
      if (currentOrder.length === 0) return currentOrder;
      const nextOrder = reconcileColumnOrder(currentOrder, defaultColumnOrder);
      if (arraysEqual(nextOrder, defaultColumnOrder)) return [];
      return arraysEqual(currentOrder, nextOrder) ? currentOrder : nextOrder;
    });
  }, [defaultColumnOrder]);

  const effectiveColumnOrder = useMemo(
    () =>
      columnOrder.length > 0
        ? reconcileColumnOrder(columnOrder, defaultColumnOrder)
        : defaultColumnOrder,
    [columnOrder, defaultColumnOrder]
  );

  const orderedColumns = useMemo(() => {
    if (!columns) return [];
    const columnsByKey = new Map<string, TColumn>(
      columns.map((column) => [column.key, column])
    );
    return effectiveColumnOrder
      .map((key) => columnsByKey.get(key))
      .filter((column): column is TColumn => !!column);
  }, [columns, effectiveColumnOrder]);

  const isColumnOrderCustom =
    columnOrder.length > 0 &&
    !arraysEqual(effectiveColumnOrder, defaultColumnOrder);

  const rowLabelWidth = rowLabelWidthOverride ?? autoRowLabelWidth;
  const colHeaderHeight = colHeaderHeightOverride ?? autoColumnHeaderHeight;

  const dragRef = useRef<{
    kind: "row" | "col";
    start: number;
    initial: number;
  } | null>(null);

  useEffect(() => {
    function onMove(e: MouseEvent) {
      const drag = dragRef.current;
      if (!drag) return;
      if (drag.kind === "row") {
        const next = Math.min(
          Math.max(ROW_WIDTH_MIN, drag.initial + (e.clientX - drag.start)),
          ROW_WIDTH_MAX
        );
        setRowLabelWidthOverride(Math.round(next));
      } else {
        const next = Math.min(
          Math.max(COL_HEIGHT_MIN, drag.initial + (e.clientY - drag.start)),
          COL_HEIGHT_MAX
        );
        setColHeaderHeightOverride(Math.round(next));
      }
    }

    function onUp() {
      const drag = dragRef.current;
      if (!drag) return;
      try {
        if (drag.kind === "row") {
          window.localStorage.setItem(
            rowWidthStorageKey,
            String(Math.round(rowLabelWidth))
          );
        } else {
          window.localStorage.setItem(
            colHeightStorageKey,
            String(Math.round(colHeaderHeight))
          );
        }
      } catch {}
      dragRef.current = null;
      setDraggingKind(null);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape" || !dragRef.current) return;
      const drag = dragRef.current;
      if (drag.kind === "row") {
        setRowLabelWidthOverride(readStoredSize(rowWidthStorageKey) ?? null);
      } else {
        setColHeaderHeightOverride(readStoredSize(colHeightStorageKey) ?? null);
      }
      dragRef.current = null;
      setDraggingKind(null);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("keydown", onKey);
    };
  }, [rowLabelWidth, colHeaderHeight, rowWidthStorageKey, colHeightStorageKey]);

  const startRowDrag = (e: ReactMouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragRef.current = {
      kind: "row",
      start: e.clientX,
      initial: rowLabelWidth,
    };
    setDraggingKind("row");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  const startColDrag = (e: ReactMouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragRef.current = {
      kind: "col",
      start: e.clientY,
      initial: colHeaderHeight,
    };
    setDraggingKind("col");
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  };

  const resetRowWidth = () => {
    setRowLabelWidthOverride(null);
    try {
      window.localStorage.removeItem(rowWidthStorageKey);
    } catch {}
  };

  const resetColHeight = () => {
    setColHeaderHeightOverride(null);
    try {
      window.localStorage.removeItem(colHeightStorageKey);
    } catch {}
  };

  const moveColumnToTarget = (movingKey: string, targetKey: string) => {
    if (movingKey === targetKey) return;
    setColumnOrder((currentOrder) => {
      const baseOrder = reconcileColumnOrder(currentOrder, defaultColumnOrder);
      const movingIndex = baseOrder.indexOf(movingKey);
      const targetIndex = baseOrder.indexOf(targetKey);
      if (movingIndex < 0 || targetIndex < 0) return currentOrder;

      const nextOrder = [...baseOrder];
      const [moving] = nextOrder.splice(movingIndex, 1);
      nextOrder.splice(targetIndex, 0, moving);
      if (arraysEqual(nextOrder, defaultColumnOrder)) return [];
      return arraysEqual(currentOrder, nextOrder) ? currentOrder : nextOrder;
    });
  };

  const startColumnDrag = (
    e: ReactDragEvent<HTMLDivElement>,
    columnKey: string
  ) => {
    setDraggingColumnKey(columnKey);
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", columnKey);
  };

  const allowColumnDrop = (e: ReactDragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  const enterColumnDropTarget = (
    e: ReactDragEvent<HTMLDivElement>,
    columnKey: string
  ) => {
    allowColumnDrop(e);
    const movingKey =
      draggingColumnKey || e.dataTransfer.getData("text/plain");
    if (movingKey) moveColumnToTarget(movingKey, columnKey);
  };

  const endColumnDrag = () => {
    setDraggingColumnKey(null);
  };

  const resetColumnOrder = () => {
    setColumnOrder([]);
  };

  const isBusy = isLoading || isFetching;
  const showStaleDim = isFetching && !!rows && !!columns;
  const controls = renderControls({ isColumnOrderCustom, resetColumnOrder });
  const indicator = isBusy ? <IndeterminateBar className="-top-px" /> : null;

  if (isLoading) {
    return (
      <div className="border bg-card">
        {controls}
        <div className="relative min-h-80">{indicator}</div>
      </div>
    );
  }

  if (!rows || !columns || rows.length === 0 || columns.length === 0) {
    return (
      <div className="border bg-card">
        {controls}
        <div className="relative">
          {indicator}
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <Search />
              </EmptyMedia>
              <EmptyTitle>{emptyTitle}</EmptyTitle>
              <EmptyDescription>{emptyDescription}</EmptyDescription>
            </EmptyHeader>
          </Empty>
        </div>
      </div>
    );
  }

  return (
    <div className="border bg-card">
      {controls}
      <div className="relative">
        {indicator}
        <div
          className={cn(
            "overflow-auto transition-opacity",
            showStaleDim && "opacity-60"
          )}
        >
          <div
            className="grid w-fit min-w-full border-r"
            style={{
              gridTemplateColumns: `${rowLabelWidth}px repeat(${orderedColumns.length}, minmax(${minColumnWidth}px, max-content))`,
            }}
          >
            <div className="sticky top-0 left-0 z-30 border-r border-b bg-background" />
            {orderedColumns.map((column) => (
              <Tooltip key={column.key}>
                <TooltipTrigger asChild>
                  <div
                    draggable
                    onDragStart={(e) => startColumnDrag(e, column.key)}
                    onDragOver={allowColumnDrop}
                    onDragEnter={(e) => enterColumnDropTarget(e, column.key)}
                    onDrop={endColumnDrag}
                    onDragEnd={endColumnDrag}
                    className={cn(
                      "group/col sticky top-0 z-10 flex cursor-grab items-end justify-center overflow-hidden border-r border-b bg-background transition-colors active:cursor-grabbing",
                      draggingColumnKey === column.key && "bg-accent/60"
                    )}
                    style={{ height: `${colHeaderHeight}px` }}
                    aria-label={`Drag ${column.label} to reorder columns`}
                  >
                    <GripVertical className="absolute top-2 left-1/2 size-3 -translate-x-1/2 text-muted-foreground/50 opacity-0 transition-opacity group-hover/col:opacity-100 group-focus/col:opacity-100" />
                    <span
                      className="px-2 py-3 text-xs whitespace-nowrap text-muted-foreground"
                      style={{
                        writingMode: "sideways-lr",
                        textOrientation: "mixed",
                      }}
                    >
                      {renderColumnLabel?.(column) ?? column.label}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  {renderColumnTooltip?.(column) ?? (
                    <>
                      <p className="text-xs">{column.label}</p>
                      <p className="text-xs text-muted-foreground">
                        Drag to reorder
                      </p>
                    </>
                  )}
                </TooltipContent>
              </Tooltip>
            ))}
            {rows.map((row) => (
              <div key={row.key} className="contents">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div
                      className="sticky left-0 z-10 flex items-center justify-end overflow-hidden border-r border-b bg-background px-3"
                      style={{ height: `${rowHeight}px` }}
                    >
                      {renderRowHeader(row)}
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="text-xs">{row.label}</p>
                  </TooltipContent>
                </Tooltip>
                {orderedColumns.map((column) => (
                  <div key={`${row.key}-${column.key}`} className="contents">
                    {renderCell(row, column, getCell(row, column))}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize row label column (double-click to reset)"
          aria-valuenow={Math.round(rowLabelWidth)}
          aria-valuemin={ROW_WIDTH_MIN}
          aria-valuemax={ROW_WIDTH_MAX}
          tabIndex={-1}
          onMouseDown={startRowDrag}
          onDoubleClick={resetRowWidth}
          className="group/resize-row absolute top-0 bottom-0 z-40 -translate-x-1/2 cursor-col-resize select-none"
          style={{ left: `${rowLabelWidth}px`, width: 11 }}
        >
          <div
            className={cn(
              "pointer-events-none absolute inset-y-0 left-1/2 -translate-x-1/2 transition-all duration-150",
              draggingKind === "row"
                ? "w-[2px] bg-ring"
                : "w-px bg-transparent group-hover/resize-row:w-[2px] group-hover/resize-row:bg-ring/80"
            )}
          />
          <div
            className={cn(
              "pointer-events-none absolute inset-y-0 left-1/2 -translate-x-1/2 transition-opacity duration-150",
              draggingKind === "row"
                ? "w-[10px] bg-ring/15 opacity-100"
                : "w-[10px] bg-ring/15 opacity-0 group-hover/resize-row:opacity-100"
            )}
          />
        </div>
        <div
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize column header height (double-click to reset)"
          aria-valuenow={Math.round(colHeaderHeight)}
          aria-valuemin={COL_HEIGHT_MIN}
          aria-valuemax={COL_HEIGHT_MAX}
          tabIndex={-1}
          onMouseDown={startColDrag}
          onDoubleClick={resetColHeight}
          className="group/resize-col absolute left-0 right-0 z-40 -translate-y-1/2 cursor-row-resize select-none"
          style={{ top: `${colHeaderHeight}px`, height: 11 }}
        >
          <div
            className={cn(
              "pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 transition-all duration-150",
              draggingKind === "col"
                ? "h-[2px] bg-ring"
                : "h-px bg-transparent group-hover/resize-col:h-[2px] group-hover/resize-col:bg-ring/80"
            )}
          />
          <div
            className={cn(
              "pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 transition-opacity duration-150",
              draggingKind === "col"
                ? "h-[10px] bg-ring/15 opacity-100"
                : "h-[10px] bg-ring/15 opacity-0 group-hover/resize-col:opacity-100"
            )}
          />
        </div>
      </div>
    </div>
  );
}
