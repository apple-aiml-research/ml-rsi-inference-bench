//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { useState } from "react";

import { Button } from "~/components/ui/button";
import {
  DEFAULT_DEMO_STATE,
  LONG_DESCRIPTION,
  MockChartBody,
  MockComboboxFilterRow,
  SHORT_DESCRIPTION,
  VariantA,
  VariantB,
  VariantC,
  VariantCurrent,
  VariantD,
  WidthClamp,
  type DemoState,
} from "~/components/chart-toolbar-variants";
import { cn } from "~/lib/utils";

const WIDTHS: { label: string; value: number | "full" }[] = [
  { label: "Full", value: "full" },
  { label: "1400", value: 1400 },
  { label: "1100", value: 1100 },
  { label: "880", value: 880 },
  { label: "720", value: 720 },
];

export default function ChartToolbarPrototypes() {
  const [width, setWidth] = useState<number | "full">("full");
  const [longDescription, setLongDescription] = useState(false);
  const [showFilterRow, setShowFilterRow] = useState(true);
  const description = longDescription ? LONG_DESCRIPTION : SHORT_DESCRIPTION;

  return (
    <div className="bg-background p-6 text-foreground min-h-screen">
      <div className="mx-auto max-w-screen-2xl space-y-8">
        <header className="space-y-3">
          <h1 className="text-lg font-semibold">Chart toolbar prototypes</h1>
          <p className="text-sm text-muted-foreground max-w-2xl">
            Comparing patterns for the description-plus-controls bar used by
            the slope, scaling, scatter, and heatmap charts. Toggle the
            description length and container width to test reflow.
          </p>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted-foreground">Container width:</span>
            {WIDTHS.map((w) => (
              <Button
                key={String(w.value)}
                size="sm"
                variant={width === w.value ? "default" : "outline"}
                onClick={() => setWidth(w.value)}
              >
                {w.label}
              </Button>
            ))}
            <Button
              size="sm"
              variant={longDescription ? "default" : "outline"}
              onClick={() => setLongDescription((v) => !v)}
            >
              {longDescription ? "Long description" : "Short description"}
            </Button>
            <Button
              size="sm"
              variant={showFilterRow ? "default" : "outline"}
              onClick={() => setShowFilterRow((v) => !v)}
            >
              {showFilterRow ? "Filter row on" : "Filter row off"}
            </Button>
          </div>
        </header>

        <Section
          label="Current"
          note="What we have today. No visible separators between controls; description wraps and pushes controls onto the next line on narrow widths."
        >
          <PreviewFrame width={width} showFilterRow={showFilterRow}>
            {(state, setState) => (
              <VariantCurrent
                description={description}
                state={state}
                setState={setState}
              />
            )}
          </PreviewFrame>
        </Section>

        <Section
          label="Variant A — Filter-bar style"
          note="Each control is a card-styled cell with its own column heading; mirrors the combobox filter row above. Description on its own row so it can never push controls."
        >
          <PreviewFrame width={width} showFilterRow={showFilterRow}>
            {(state, setState) => (
              <VariantA
                description={description}
                state={state}
                setState={setState}
              />
            )}
          </PreviewFrame>
        </Section>

        <Section
          label="Variant B — Bordered toolbar group"
          note="Description on the left, single-line truncated with tooltip; controls on the right grouped into one bordered pill with vertical dividers."
        >
          <PreviewFrame width={width} showFilterRow={showFilterRow}>
            {(state, setState) => (
              <VariantB
                description={description}
                state={state}
                setState={setState}
              />
            )}
          </PreviewFrame>
        </Section>

        <Section
          label="Variant C — Description on top + chips"
          note="Description on its own row with an info icon; each control becomes a bordered chip. Predictable baseline; wraps cleanly at narrow widths."
        >
          <PreviewFrame width={width} showFilterRow={showFilterRow}>
            {(state, setState) => (
              <VariantC
                description={description}
                state={state}
                setState={setState}
              />
            )}
          </PreviewFrame>
        </Section>

        <Section
          label="Variant D — Two-column grid"
          note="CSS grid with a fixed description column; controls live in a divided pill on the right. Description and controls never compete for the same row."
        >
          <PreviewFrame width={width} showFilterRow={showFilterRow}>
            {(state, setState) => (
              <VariantD
                description={description}
                state={state}
                setState={setState}
              />
            )}
          </PreviewFrame>
        </Section>
      </div>
    </div>
  );
}

function Section({
  label,
  note,
  children,
}: {
  label: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-2">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-sm font-semibold">{label}</h2>
        <p className="max-w-2xl text-xs text-muted-foreground">{note}</p>
      </div>
      {children}
    </section>
  );
}

function PreviewFrame({
  width,
  showFilterRow,
  children,
}: {
  width: number | "full";
  showFilterRow: boolean;
  children: (
    state: DemoState,
    setState: React.Dispatch<React.SetStateAction<DemoState>>
  ) => React.ReactNode;
}) {
  const [state, setState] = useState<DemoState>(DEFAULT_DEMO_STATE);
  return (
    <div
      className={cn(
        "rounded-md border border-dashed border-muted-foreground/30 p-4",
        "bg-muted/20"
      )}
    >
      <WidthClamp width={width}>
        <div className="space-y-0">
          {showFilterRow && <MockComboboxFilterRow />}
          {children(state, setState)}
          <MockChartBody />
        </div>
      </WidthClamp>
    </div>
  );
}
