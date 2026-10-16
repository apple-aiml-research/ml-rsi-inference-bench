#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
"""Tests for the optional task-level pre_artifacts.sh hook.

The hook runs inside the agent environment after the agent finishes and
immediately before artifact collection, so tasks can materialize artifacts
(e.g. capture the agent's change set as /logs/artifacts/model.patch for a
separate verifier).
"""

import asyncio
import functools
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pier.models.task.paths import TaskPaths
from pier.trial.trial import Trial


def run_async(fn):
    """Drive an async test with asyncio.run (pier has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _fake_trial(task_dir: Path) -> SimpleNamespace:
    environment = AsyncMock()
    environment.exec.return_value = SimpleNamespace(return_code=0)
    return SimpleNamespace(
        _task=SimpleNamespace(paths=TaskPaths(task_dir)),
        _environment=environment,
        _logger=logging.getLogger(__name__),
    )


@run_async
async def test_no_script_is_a_noop(tmp_path: Path) -> None:
    trial = _fake_trial(tmp_path)
    await Trial._run_pre_artifacts_script(trial)
    trial._environment.upload_file.assert_not_awaited()
    trial._environment.exec.assert_not_awaited()


@run_async
async def test_script_is_uploaded_and_executed(tmp_path: Path) -> None:
    (tmp_path / "pre_artifacts.sh").write_text("#!/bin/bash\necho hi\n")
    trial = _fake_trial(tmp_path)
    await Trial._run_pre_artifacts_script(trial)
    trial._environment.upload_file.assert_awaited_once_with(
        source_path=tmp_path / "pre_artifacts.sh",
        target_path="/tmp/.pier-pre-artifacts.sh",
    )
    trial._environment.exec.assert_awaited_once_with(
        command="bash /tmp/.pier-pre-artifacts.sh",
        timeout_sec=300,
    )


@run_async
async def test_nonzero_exit_does_not_raise(tmp_path: Path) -> None:
    (tmp_path / "pre_artifacts.sh").write_text("#!/bin/bash\nexit 3\n")
    trial = _fake_trial(tmp_path)
    trial._environment.exec.return_value = SimpleNamespace(return_code=3)
    await Trial._run_pre_artifacts_script(trial)  # must not raise


@run_async
async def test_exec_exception_is_swallowed(tmp_path: Path) -> None:
    (tmp_path / "pre_artifacts.sh").write_text("#!/bin/bash\n")
    trial = _fake_trial(tmp_path)
    trial._environment.exec.side_effect = RuntimeError("env died")
    await Trial._run_pre_artifacts_script(trial)  # must not raise
