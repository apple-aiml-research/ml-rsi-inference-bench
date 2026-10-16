//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("compare", "routes/compare.tsx"),
  route("jobs/:jobName", "routes/job.tsx"),
  route("jobs/:jobName/critiques", "routes/critiques.tsx"),
  route(
    "jobs/:jobName/critiques/:critiqueRunName",
    "routes/critique-run.tsx"
  ),
  route(
    "jobs/:jobName/tasks/:source/:agent/:modelProvider/:modelName/:taskName",
    "routes/task.tsx"
  ),
  route(
    "jobs/:jobName/tasks/:source/:agent/:modelProvider/:modelName/:taskName/trials/:trialName",
    "routes/trial.tsx"
  ),
  route("task-definitions", "routes/task-definitions.tsx"),
  route("task-definitions/:taskName", "routes/task-definition.tsx"),
  route("prototypes/chart-toolbar", "routes/chart-toolbar-prototypes.tsx"),
] satisfies RouteConfig;
