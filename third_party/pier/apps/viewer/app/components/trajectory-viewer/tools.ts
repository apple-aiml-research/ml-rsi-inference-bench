//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
// Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
/**
 * Heuristics that turn a generic tool call into a single one-line summary.
 *
 *   `Read app/foo.tsx`
 *   `Ran git status`
 *   `Edited types.ts`
 *   `Searched for "TODO"`
 *
 * Most agents (Claude Code, OpenCode, Codex, Gemini) use overlapping but
 * subtly different tool names; we map them all by lowercase substring so
 * "read", "Read", "view_file", "fs_read" all land on the same shape.
 */
import type { ResolvedToolCall } from "./types";

export type ToolCategory =
  | "read"
  | "write"
  | "execute"
  | "search"
  | "list"
  | "directory"
  | "network"
  | "subagent"
  | "other";

export function getToolCategory(name: string): ToolCategory {
  const lower = name.toLowerCase();
  if (
    lower === "task" ||
    lower === "agent" ||
    lower.includes("subagent") ||
    lower.includes("delegate") ||
    lower.startsWith("launch_agent")
  )
    return "subagent";
  if (lower.includes("read") || lower.includes("cat") || lower.includes("view")) return "read";
  if (
    lower.includes("write") ||
    lower.includes("edit") ||
    lower.includes("create") ||
    lower.includes("replace") ||
    lower.includes("insert") ||
    lower.includes("patch")
  )
    return "write";
  if (lower === "todowrite" || lower === "todo_write" || lower === "update_plan") return "other";
  if (
    lower.includes("bash") ||
    lower.includes("exec") ||
    lower === "run_command" ||
    lower.includes("shell") ||
    lower.includes("terminal") ||
    (lower.includes("command") && !lower.includes("write"))
  )
    return "execute";
  if (lower.includes("grep") || lower.includes("search") || lower.includes("find")) return "search";
  if (lower.includes("glob") || lower.includes("ls") || lower === "list_files") return "list";
  if (lower.includes("dir") || lower.includes("tree")) return "directory";
  if (
    lower.includes("fetch") ||
    lower.includes("http") ||
    lower.includes("web") ||
    lower.includes("url") ||
    lower.includes("curl")
  )
    return "network";
  return "other";
}

export interface ToolLabel {
  /** "Read", "Ran", "Edited", … */
  prefix: string;
  /** The salient argument (path, command, pattern). */
  code?: string;
}

export function getToolLabel(toolCall: ResolvedToolCall): ToolLabel {
  const args = toolCall.call.arguments ?? {};
  const name = toolCall.call.function_name.toLowerCase();
  const a = args as Record<string, unknown>;

  if (
    name.includes("bash") ||
    name === "exec_command" ||
    name === "execute_bash" ||
    name === "run_command" ||
    name.includes("shell") ||
    name.includes("terminal") ||
    name.includes("command") ||
    name === "run"
  ) {
    const cmd =
      String(a.command ?? a.cmd ?? "")
        .split("\n")[0]
        ?.slice(0, 200) ?? "";
    return { prefix: "Ran", code: cmd || undefined };
  }
  if (name === "todowrite" || name === "todo_write" || name === "update_plan") {
    const todos = (a.todos ?? a.plan) as Array<{ content?: string; step?: string }> | undefined;
    if (Array.isArray(todos) && todos.length > 0) {
      return {
        prefix: name === "update_plan" ? "Updated plan" : "Updated todos",
        code: `${todos.length} ${todos.length === 1 ? "item" : "items"}`,
      };
    }
    return { prefix: name === "update_plan" ? "Updated plan" : "Updated todos" };
  }
  if (name === "apply_patch") {
    const patch = String(a.input ?? a.patch ?? "");
    const m = patch.match(/\*\*\* (?:Add File|Update File|Delete File): (.+)/);
    if (m) {
      return { prefix: "Applied patch", code: m[1] };
    }
    return { prefix: "Applied patch" };
  }
  if (name.includes("read") || name === "view") {
    return {
      prefix: "Read",
      code: (a.file_path as string) ?? (a.path as string) ?? "file",
    };
  }
  if (name === "write_file" || name === "write" || name.includes("write")) {
    const target = (a.file_path as string) ?? (a.path as string);
    if (target) return { prefix: "Wrote", code: target };
    // Fall through if no path — name probably isn't really a file write.
  }
  if (name.includes("edit") || name.includes("replace") || name.includes("patch")) {
    return {
      prefix: "Edited",
      code: (a.file_path as string) ?? (a.path as string) ?? "file",
    };
  }
  if (name.includes("grep")) {
    return { prefix: "Searched for", code: String(a.pattern ?? "pattern") };
  }
  if (name.includes("glob") || name.includes("find")) {
    return {
      prefix: "Searched for files",
      code: String(a.pattern ?? a.path ?? "*"),
    };
  }
  if (name.includes("search")) {
    return {
      prefix: "Searched",
      code: String(a.query ?? a.pattern ?? "") || undefined,
    };
  }
  if (name.includes("list") || name === "ls") {
    return {
      prefix: "Listed",
      code: String(a.path ?? a.directory ?? "") || undefined,
    };
  }
  if (name === "task" || name.includes("subagent") || name.includes("delegate")) {
    return {
      prefix: "Delegated",
      code:
        String(a.description ?? a.prompt ?? "")
          .split("\n")[0]
          ?.slice(0, 200) || undefined,
    };
  }
  if (typeof a.file_path === "string")
    return { prefix: toolCall.call.function_name, code: a.file_path };
  if (typeof a.path === "string") return { prefix: toolCall.call.function_name, code: a.path };
  if (typeof a.command === "string")
    return { prefix: toolCall.call.function_name, code: a.command.slice(0, 200) };
  if (typeof a.pattern === "string")
    return { prefix: toolCall.call.function_name, code: a.pattern };
  if (typeof a.query === "string") return { prefix: toolCall.call.function_name, code: a.query };
  if (typeof a.url === "string") return { prefix: toolCall.call.function_name, code: a.url };
  return { prefix: toolCall.call.function_name };
}
