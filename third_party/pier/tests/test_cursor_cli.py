#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from pathlib import Path

import pytest

from pier.agents.installed.cursor_cli import CursorCli
from pier.agents.factory import AgentFactory
from pier.models.agent.name import AgentName


def test_cursor_cli_is_registered(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.CURSOR_CLI,
        logs_dir=tmp_path,
        model_name="cursor/composer-2.5",
    )

    assert isinstance(agent, CursorCli)
    assert agent.name() == "cursor-cli"


def test_cursor_cli_install_spec_uses_official_installer(tmp_path: Path):
    agent = CursorCli(logs_dir=tmp_path, model_name="cursor/composer-2.5")

    spec = agent.install_spec()

    assert spec.agent_name == "cursor-cli"
    assert "curl https://cursor.com/install -fsS | bash" in spec.steps[1].run
    assert spec.verification_command is not None
    assert "cursor-agent --version" in spec.verification_command


def test_cursor_cli_reports_cursor_domains(tmp_path: Path):
    agent = CursorCli(logs_dir=tmp_path, model_name="cursor/composer-2.5")

    domains = set(agent.network_allowlist().domains)

    assert {"cursor.com", "api2.cursor.sh", ".cursor.sh"} <= domains


def test_cursor_cli_no_internet_config_allows_coding_tools_but_not_web_fetch():
    config = CursorCli._no_internet_cli_config()

    assert config["permissions"]["allow"] == [
        "Shell(**)",
        "Read(**)",
        "Write(**)",
        "Mcp(**)",
    ]
    assert config["approvalMode"] == "allowlist"
    assert config["webFetchDomainAllowlist"] == []


def test_cursor_cli_no_internet_config_command_writes_home_config(tmp_path: Path):
    agent = CursorCli(logs_dir=tmp_path, model_name="cursor/composer-2.5")

    command = agent._build_no_internet_cli_config_command()

    assert "mkdir -p ~/.cursor" in command
    assert "> ~/.cursor/cli-config.json" in command
    assert "Shell(**)" in command
    assert "webFetchDomainAllowlist" in command


def test_cursor_cli_converts_stream_json_to_atif(tmp_path: Path):
    agent = CursorCli(logs_dir=tmp_path, model_name="cursor/composer-2.5")
    events = [
        {
            "type": "system",
            "subtype": "init",
            "apiKeySource": "env",
            "cwd": "/workspace",
            "session_id": "session-1",
            "model": "composer-2.5",
            "permissionMode": "default",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Fix the tests"}],
            },
            "session_id": "session-1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I'll inspect the repo."}],
            },
            "session_id": "session-1",
            "model_call_id": "call-1",
            "timestamp_ms": 1000,
        },
        {
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "tool-1",
            "tool_call": {
                "readToolCall": {
                    "args": {"path": "README.md"},
                    "result": {"success": {"content": "# Pier"}},
                }
            },
            "session_id": "session-1",
            "model_call_id": "call-1",
            "timestamp_ms": 1001,
        },
        {
            "type": "result",
            "subtype": "success",
            "duration_ms": 5000,
            "duration_api_ms": 4500,
            "is_error": False,
            "result": "done",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 20,
                "cacheReadTokens": 10,
                "cacheWriteTokens": 5,
            },
            "session_id": "session-1",
            "request_id": "request-1",
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.session_id == "session-1"
    assert trajectory.agent.name == "cursor-cli"
    assert trajectory.steps[0].source == "user"
    assert trajectory.steps[1].source == "agent"
    assert trajectory.steps[1].tool_calls is not None
    assert trajectory.steps[1].tool_calls[0].function_name == "readToolCall"
    assert trajectory.steps[1].observation is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 115
    assert trajectory.final_metrics.total_completion_tokens == 20
    assert trajectory.final_metrics.total_cached_tokens == 10
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.0001045)
    assert trajectory.final_metrics.extra is not None
    assert trajectory.final_metrics.extra["cost_source"] == "cursor_pricing"


def test_cursor_cli_preserves_thinking_before_tool_call_and_timeout(tmp_path: Path):
    agent = CursorCli(logs_dir=tmp_path, model_name="cursor/composer-2.5")
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-1",
        },
        {
            "type": "thinking",
            "subtype": "delta",
            "text": "Need to inspect",
            "session_id": "session-1",
        },
        {
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "tool-1",
            "tool_call": {
                "grepToolCall": {
                    "args": {"pattern": "audit"},
                    "result": {"success": {"matches": []}},
                }
            },
            "session_id": "session-1",
            "model_call_id": "call-1",
        },
        {
            "type": "thinking",
            "subtype": "delta",
            "text": "Trailing thought",
            "session_id": "session-1",
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.steps[0].reasoning_content == "Need to inspect"
    assert trajectory.steps[0].tool_calls is not None
    assert trajectory.steps[1].reasoning_content == "Trailing thought"
