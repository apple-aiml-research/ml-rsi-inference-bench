//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * Inline image rendering for trajectory content + observation parts.
 *
 * Pier's API exposes a trial's working directory at
 *   `/api/jobs/{job}/trials/{trial}/files/agent/{path}?step={stepName}`
 * — a stable URL shape, so the viewer just builds it directly instead of
 * asking the host for a renderer callback.
 *
 * `<TrajectoryImage>` consumes the trial coordinates from Context, lazy-
 * loads the URL into an `<img>`, and falls back to a small "Image
 * unavailable" panel that surfaces the API's error detail. This is a
 * close port of harbor's `ImageWithFallback` (apps/viewer/app/components/
 * trajectory/content-renderer.tsx) but lives inside the trajectory-viewer
 * package so leaf components can drop it in directly without prop
 * drilling.
 */
import { createContext, useContext, useState } from "react";
import { ImageOff } from "lucide-react";

export interface TrajectoryImageContext {
  jobName: string;
  trialName: string;
  /** ATIF step identifier when the trial is being viewed at a specific
   *  step (preview-style). Pier serializes intermediate snapshots that
   *  way; live trials leave it null. */
  stepName?: string | null;
}

const Ctx = createContext<TrajectoryImageContext | null>(null);

export function TrajectoryImageProvider({
  value,
  children,
}: {
  value: TrajectoryImageContext;
  children: React.ReactNode;
}) {
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

interface ImageError {
  status: number;
  message: string;
}

export interface TrajectoryImageProps {
  /** Path relative to the trial's `agent/` directory — the value that
   *  ATIF emits in `image.source.path`. */
  path: string;
}

/** Inline image rendered from the current trial's agent directory. */
export function TrajectoryImage({ path }: TrajectoryImageProps) {
  const ctx = useContext(Ctx);
  const [error, setError] = useState<ImageError | null>(null);

  if (!ctx) {
    // No provider — render a quiet placeholder rather than throwing,
    // so embedding the viewer outside pier (e.g. unit tests) stays safe.
    return (
      <div className="my-2 rounded border border-dashed border-muted-foreground/40 bg-muted/40 p-2 text-xs text-muted-foreground">
        <ImageOff className="mr-1 inline h-3 w-3" />
        {path}
      </div>
    );
  }

  const stepQuery = ctx.stepName ? `?step=${encodeURIComponent(ctx.stepName)}` : "";
  const src = `/api/jobs/${encodeURIComponent(ctx.jobName)}/trials/${encodeURIComponent(ctx.trialName)}/files/agent/${path}${stepQuery}`;

  // On <img> error, refetch the URL once to surface the API's JSON error
  // detail (e.g. "file not found", "step out of range") in the fallback.
  const handleError = async () => {
    try {
      const response = await fetch(src);
      let message = response.statusText || "Failed to load image";
      if (!response.ok) {
        try {
          const json = (await response.json()) as { detail?: string };
          message = json.detail ?? message;
        } catch {
          // body wasn't JSON; keep statusText
        }
      }
      setError({ status: response.status, message });
    } catch {
      setError({ status: 0, message: "Network error" });
    }
  };

  if (error) {
    return (
      <div className="my-2 rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 p-3 text-sm">
        <div className="mb-1 flex items-center gap-2 text-muted-foreground">
          <ImageOff className="h-4 w-4" />
          <span className="font-medium">Image unavailable</span>
          {error.status > 0 && (
            <span className="rounded bg-muted px-1.5 py-0.5 text-xs">{error.status}</span>
          )}
        </div>
        <div className="break-all font-mono text-xs text-muted-foreground/80">{path}</div>
        <div className="mt-2 text-xs text-muted-foreground/60">{error.message}</div>
      </div>
    );
  }

  return (
    <div className="my-2">
      <img
        src={src}
        alt={`Image: ${path}`}
        loading="lazy"
        onError={handleError}
        className="h-auto max-w-full rounded-md border border-border"
        style={{ maxHeight: 400 }}
      />
      <div className="mt-1 font-mono text-xs text-muted-foreground">{path}</div>
    </div>
  );
}
