//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import * as React from "react";
import { InfoIcon } from "lucide-react";

import { Checkbox } from "~/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { cn } from "~/lib/utils";

/**
 * Standard toolbar used above chart components. Renders a two-row layout:
 * an info description on top and a wrapping row of chip-styled controls
 * below. The description never competes with controls for layout space.
 */
export function ChartToolbar({
  description,
  children,
  className,
}: {
  description: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("border-b", className)}>
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-2.5">
        {children}
      </div>
      <div className="flex items-start gap-2 px-4 py-2.5">
        <InfoIcon className="size-3.5 shrink-0 translate-y-0.5 text-muted-foreground" />
        <span className="text-xs leading-relaxed text-muted-foreground">
          {description}
        </span>
      </div>
    </div>
  );
}

interface ChartToolbarSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export function ChartToolbarSelect({
  label,
  value,
  onValueChange,
  options,
  placeholder,
  className,
}: {
  label: string;
  value: string | undefined;
  onValueChange: (value: string) => void;
  options: ChartToolbarSelectOption[];
  placeholder?: string;
  className?: string;
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        size="sm"
        className={cn(
          "h-8 rounded-md border-border bg-background text-xs hover:bg-accent",
          className
        )}
      >
        <span className="text-muted-foreground">{label}</span>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem
            key={option.value}
            value={option.value}
            disabled={option.disabled}
          >
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function ChartToolbarToggle({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
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

/**
 * Range slider with label and a value readout, styled to match the
 * surrounding chip controls.
 */
export function ChartToolbarSlider({
  label,
  value,
  onValueChange,
  min = 0,
  max = 100,
  step = 5,
  formatValue,
  ariaLabel,
}: {
  label: string;
  value: number;
  onValueChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  formatValue?: (value: number) => string;
  ariaLabel?: string;
}) {
  const formatted = formatValue ? formatValue(value) : `${value}%`;
  return (
    <div className="inline-flex h-8 items-center gap-2 rounded-md border border-border bg-background px-3 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <input
        aria-label={ariaLabel ?? label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onValueChange(Number(e.target.value))}
        className="w-20 accent-foreground"
      />
      <span className="min-w-8 text-right tabular-nums">{formatted}</span>
    </div>
  );
}

/**
 * Inline button that shares its baseline with toolbar chips but reads as a
 * subtle text action (used for "Reset", "Swap", etc).
 */
export function ChartToolbarAction({
  onClick,
  children,
  className,
}: {
  onClick: () => void;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex h-8 items-center px-2 text-xs text-foreground underline-offset-2 hover:underline",
        className
      )}
    >
      {children}
    </button>
  );
}
