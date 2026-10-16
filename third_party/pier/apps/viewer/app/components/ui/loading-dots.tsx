//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { cn } from "~/lib/utils";

function LoadingDots({
  className,
  text = "Loading",
}: {
  className?: string;
  text?: string;
}) {
  return (
    <span className={cn("inline-flex", className)}>
      {text}
      <span className="inline-flex">
        <span className="animate-dot-1">.</span>
        <span className="animate-dot-2 -ml-[0.1em]">.</span>
        <span className="animate-dot-3 -ml-[0.1em]">.</span>
      </span>
    </span>
  );
}

export { LoadingDots };
