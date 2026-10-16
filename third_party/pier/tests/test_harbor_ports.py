#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import json

import pytest
from pydantic import ValidationError

from pier.agents.installed.opencode import OpenCode
from pier.environments.capabilities import EnvironmentResourceCapabilities
from pier.environments.resource_policies import validate_resource_values
from pier.models.task.config import (
    EnvironmentConfig as TaskEnvironmentConfig,
)
from pier.models.task.config import (
    TaskConfig,
    VerifierEnvironmentMode,
)
from pier.models.task.verifier_mode import resolve_task_verifier_mode
from pier.models.trial.config import ResourceMode


def test_task_toml_dump_uses_blank_lines_between_sections():
    config = TaskConfig(
        environment=TaskEnvironmentConfig(cpus=2, memory_mb=4096),
        source="local",
    )

    dumped = config.model_dump_toml()

    assert "\n\n[verifier]\n" in dumped
    assert "\n\n[environment]\n" in dumped


def test_legacy_memory_conflict_is_rejected():
    with pytest.raises(ValidationError, match="Conflicting 'memory' and 'memory_mb'"):
        TaskEnvironmentConfig.model_validate({"memory": "1G", "memory_mb": 2048})


def test_verifier_environment_implies_separate_mode():
    config = TaskConfig.model_validate(
        {
            "verifier": {"environment": {"cpus": 1}},
            "environment": {},
        }
    )

    assert config.verifier.environment_mode is None
    assert config.verifier.environment is not None
    assert resolve_task_verifier_mode(config) == VerifierEnvironmentMode.SEPARATE


def test_verifier_shared_mode_rejects_environment():
    with pytest.raises(ValidationError, match="environment_mode='shared'"):
        TaskConfig.model_validate(
            {
                "verifier": {
                    "environment_mode": VerifierEnvironmentMode.SHARED,
                    "environment": {},
                },
                "environment": {},
            }
        )


def test_explicit_resource_mode_requires_task_value():
    with pytest.raises(ValueError, match="CPU resource mode 'limit' requires"):
        validate_resource_values(
            cpu_enforcement_policy=ResourceMode.LIMIT,
            memory_enforcement_policy=ResourceMode.AUTO,
            cpus=None,
            memory_mb=None,
        )


def test_resource_capabilities_reject_unsupported_request():
    caps = EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    with pytest.raises(ValueError, match="CPU resource requests"):
        from pier.environments.resource_policies import validate_resource_capabilities

        validate_resource_capabilities(
            environment_label="docker",
            resource_capabilities=caps,
            cpu_enforcement_policy=ResourceMode.REQUEST,
            memory_enforcement_policy=ResourceMode.AUTO,
        )


def test_opencode_extracts_json_error_events(tmp_path):
    agent = OpenCode(logs_dir=tmp_path, model_name="provider/model")
    (tmp_path / "opencode.txt").write_text(
        json.dumps(
            {
                "type": "error",
                "error": {"name": "ProviderError", "data": {"message": "bad key"}},
            }
        )
        + "\n"
    )

    assert agent._error_messages() == ["bad key"]
