#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from enum import Enum


class AgentName(str, Enum):
    ORACLE = "oracle"
    NOP = "nop"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    CURSOR_CLI = "cursor-cli"
    GEMINI_CLI = "gemini-cli"
    MINI_SWE_AGENT = "mini-swe-agent"
    SWE_AGENT = "swe-agent"
    OPENCODE = "opencode"
    TAIGA = "taiga"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}
