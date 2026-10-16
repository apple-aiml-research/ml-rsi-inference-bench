#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import logging
import socket
import asyncio

from pier.environments.daytona import (
    DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS,
    DaytonaEnvironment,
    _DaytonaDinD,
    resolve_network_allowlist_to_daytona_cidrs,
)
from pier.models.agent.network import NetworkAllowlist


def _addr(ip: str):
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))


def test_daytona_resolves_domains_to_ipv4_cidrs(monkeypatch):
    def fake_getaddrinfo(host, *_args, **_kwargs):
        assert host == "api.openai.com"
        return [_addr("203.0.113.10"), _addr("203.0.113.10")]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    resolution, cidrs = resolve_network_allowlist_to_daytona_cidrs(
        ["api.openai.com", ".anthropic.com"]
    )

    assert resolution == {"api.openai.com": ["203.0.113.10"]}
    assert cidrs == ["203.0.113.10/32"]


def test_daytona_collapses_resolved_cidrs_to_daytona_limit(monkeypatch):
    def fake_getaddrinfo(_host, *_args, **_kwargs):
        return [_addr(f"203.0.113.{i}") for i in range(1, 18)]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    _resolution, cidrs = resolve_network_allowlist_to_daytona_cidrs(
        ["api.openai.com"]
    )

    assert len(cidrs) <= DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS
    assert all(cidr.endswith(("/32", "/31", "/30", "/29", "/28")) for cidr in cidrs)


def test_daytona_network_params_use_resolved_allowlist(monkeypatch):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env._resolved_network_allow_list = None
    env._network_resolution_debug = {}
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    env.network_allowlist = NetworkAllowlist(domains=["api.openai.com"])
    env.logger = logging.getLogger("test")

    monkeypatch.setattr(
        "pier.environments.daytona.resolve_network_allowlist_to_daytona_cidrs",
        lambda domains: (
            {"api.openai.com": ["203.0.113.10"]},
            ["203.0.113.10/32"],
        ),
    )

    assert DaytonaEnvironment._network_params(env) == {
        "network_block_all": False,
        "network_allow_list": "203.0.113.10/32",
    }


def test_daytona_network_params_block_when_no_cidrs(monkeypatch):
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env._resolved_network_allow_list = None
    env._network_resolution_debug = {}
    env.task_env_config = type("TaskEnv", (), {"allow_internet": False})()
    env.network_allowlist = NetworkAllowlist(domains=[".anthropic.com"])
    env.logger = logging.getLogger("test")

    monkeypatch.setattr(
        "pier.environments.daytona.resolve_network_allowlist_to_daytona_cidrs",
        lambda domains: ({}, []),
    )

    assert DaytonaEnvironment._network_params(env) == {"network_block_all": True}


def test_daytona_compose_keeps_main_network_when_sandbox_allowlist_is_active():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env.environment_name = "task"
    env.session_id = "task.1"
    env._persistent_env = {}
    env.task_env_config = type(
        "TaskEnv",
        (),
        {
            "allow_internet": False,
            "env": {},
            "cpus": 1,
            "memory_mb": 1024,
            "docker_image": None,
        },
    )()
    env._compose_should_block_main_network = lambda: False

    strategy = _DaytonaDinD(env)

    assert not any("no-network" in flag for flag in strategy._compose_file_flags())


def test_daytona_compose_does_not_advertise_agent_preinstall():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = True

    assert env.capabilities.filtered_egress is True
    assert env.capabilities.preinstall_agents is False


def test_daytona_pins_resolved_hosts():
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._compose_mode = False
    env._network_resolution_debug = {
        "domain_resolution": {"api.example.com": ["203.0.113.10"]}
    }
    captured = {}

    async def sandbox_exec(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return type("Result", (), {"return_code": 0, "stdout": "", "stderr": ""})()

    env._sandbox_exec = sandbox_exec

    asyncio.run(DaytonaEnvironment._pin_resolved_hosts(env))

    assert "203.0.113.10 api.example.com" in captured["command"]
    assert captured["kwargs"]["shell"] == "bash -c"
