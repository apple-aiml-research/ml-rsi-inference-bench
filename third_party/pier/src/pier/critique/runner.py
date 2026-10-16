#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import shlex
import shutil
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from shortuuid import ShortUUID

from pier.agents.installed.base import BaseInstalledAgent
from pier.critique.models import (
    CritiqueConfig,
    CritiqueItem,
    CritiqueItemResult,
    CritiqueJobResult,
)
from pier.models.agent.context import AgentContext
from pier.models.job.lock import LOCK_FILENAME, JobLock, TrialLock
from pier.models.task.task import Task
from pier.models.trial.config import TaskConfig
from pier.models.trial.paths import TrialPaths
from pier.models.trial.result import ExceptionInfo, TimingInfo, TrialResult
from pier.trial.execution import TrialExecution
from pier.utils.logger import logger as global_logger

CRITIQUE_DIRNAME = "critique"
CRITIQUE_INPUT_DIRNAME = "input"
CRITIQUE_TASK_DIRNAME = "task"
CRITIQUE_TRIAL_DIRNAME = "trial"
CRITIQUE_RESULT_FILENAME = "critique-result.json"
CRITIQUE_MARKDOWN_FILENAME = "critique-result.md"
CRITIQUE_RUNS_DIRNAME = ".critiques"
REDACTED_CRITIQUE_PROMPT = "[redacted critique prompt]"


@dataclass(frozen=True)
class CritiquePaths:
    critique_dir: Path

    @property
    def mount_paths(self) -> TrialPaths:
        return TrialPaths(self.critique_dir)

    def mkdir(self) -> None:
        self.mount_paths.mkdir()

    @property
    def config_path(self) -> Path:
        return self.critique_dir / "critique-config.json"

    @property
    def result_path(self) -> Path:
        return self.critique_dir / "critique-metadata.json"

    @property
    def log_path(self) -> Path:
        return self.critique_dir / "critique.log"

    @property
    def agent_dir(self) -> Path:
        return self.mount_paths.agent_dir

    @property
    def artifacts_dir(self) -> Path:
        return self.mount_paths.artifacts_dir

    @property
    def artifacts_manifest_path(self) -> Path:
        return self.mount_paths.artifacts_manifest_path

    @property
    def exception_message_path(self) -> Path:
        return self.critique_dir / "exception.txt"


def is_job_dir(path: Path) -> bool:
    return (path / "job.log").exists()


def is_trial_dir(path: Path) -> bool:
    return (path / "trial.log").exists() and (path / "result.json").exists()


def critique_run_dir(source_job_dir: Path, run_name: str) -> Path:
    return source_job_dir / CRITIQUE_RUNS_DIRNAME / run_name


def _replace_template_vars(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _prefixed_digest(digest: str) -> str:
    return digest if digest.startswith("sha256:") else f"sha256:{digest}"


def _stable_json_key(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _trial_result_task_order_key(task: TaskConfig) -> tuple[Any, ...]:
    return (
        "package"
        if task.is_package_task()
        else ("git" if task.is_git_task() else "local"),
        task.name,
        _prefixed_digest(task.ref) if task.ref and task.is_package_task() else None,
        task.source,
        task.path.as_posix() if task.path is not None else None,
        task.git_url,
        task.git_commit_id,
    )


def _trial_record_attempt_key(record: tuple[Path, TrialResult]) -> tuple[Any, ...]:
    trial_dir, result = record
    return (
        result.started_at is None,
        result.started_at.isoformat() if result.started_at is not None else "",
        trial_dir.name,
    )


def _is_passing_trial(result: TrialResult) -> bool:
    has_reward_one = (
        result.verifier_result is not None
        and result.verifier_result.rewards is not None
        and result.verifier_result.rewards.get("reward", 0) == 1.0
    )
    return has_reward_one and result.exception_info is None


def _candidate_names(*values: str | None) -> set[str]:
    names: set[str] = set()
    for value in values:
        if not value:
            continue
        names.add(value)
        if "/" in value:
            names.add(value.split("/", 1)[1])
            names.add(value.rsplit("/", 1)[1])
    return names


def _matches_name_filter(candidates: set[str], wanted: list[str] | None) -> bool:
    if not wanted:
        return True
    normalized_candidates = {candidate.lower() for candidate in candidates}
    return any(name.lower() in normalized_candidates for name in wanted)


def _source_agent_name_candidates(result: TrialResult) -> set[str]:
    return _candidate_names(result.agent_info.name, result.config.agent.name)


def _source_model_name_candidates(result: TrialResult) -> set[str]:
    model_info = result.agent_info.model_info
    model_name = model_info.name if model_info is not None else None
    provider_model_name = (
        f"{model_info.provider}/{model_info.name}"
        if model_info is not None and model_info.provider is not None
        else None
    )
    return _candidate_names(
        model_name,
        provider_model_name,
        result.config.agent.model_name,
    )


class CritiqueRunner:
    def __init__(self, config: CritiqueConfig):
        self.config = config
        self.source_job_dir = config.source_job_dir.resolve()
        self.critique_dir = critique_run_dir(self.source_job_dir, config.run_name)

    def collect_items(self) -> list[CritiqueItem]:
        if not is_job_dir(self.source_job_dir):
            raise ValueError(f"Not a Pier job directory: {self.source_job_dir}")

        trial_records = [
            (
                trial_dir,
                TrialResult.model_validate_json(
                    (trial_dir / "result.json").read_text(encoding="utf-8")
                ),
            )
            for trial_dir in sorted(
                d
                for d in self.source_job_dir.iterdir()
                if d.is_dir() and is_trial_dir(d)
            )
        ]

        if self.config.trial_names:
            selected = set(self.config.trial_names)
            trial_records = [
                record for record in trial_records if record[0].name in selected
            ]
            missing = selected - {record[0].name for record in trial_records}
            if missing:
                raise ValueError(
                    f"Trial(s) not found in {self.source_job_dir}: "
                    f"{', '.join(sorted(missing))}"
                )
            selected_order = {
                trial_name: index
                for index, trial_name in enumerate(self.config.trial_names)
            }
            trial_records.sort(key=lambda record: selected_order[record[0].name])
        else:
            trial_records = self._order_trial_records_by_source_job(trial_records)

        items: list[CritiqueItem] = []
        for trial_dir, source_result in trial_records:
            if not _matches_name_filter(
                _source_agent_name_candidates(source_result),
                self.config.source_agent_names,
            ):
                continue

            if not _matches_name_filter(
                _source_model_name_candidates(source_result),
                self.config.source_model_names,
            ):
                continue

            if self.config.filter_passing is not None:
                is_passing = _is_passing_trial(source_result)
                if self.config.filter_passing != is_passing:
                    continue

            task_dir = source_result.config.task.get_local_path()
            if not task_dir.exists():
                raise FileNotFoundError(
                    f"Task directory for trial {trial_dir.name} does not exist: "
                    f"{task_dir}"
                )
            items.append(
                CritiqueItem(
                    source_trial_dir=trial_dir,
                    source_trial_name=trial_dir.name,
                    task_dir=task_dir,
                )
            )

        if self.config.sample_seed is not None:
            random.Random(self.config.sample_seed).shuffle(items)

        if self.config.limit is not None:
            items = items[: self.config.limit]

        if not items:
            raise ValueError(f"No trial directories selected in {self.source_job_dir}")

        return items

    def _order_trial_records_by_source_job(
        self, trial_records: list[tuple[Path, TrialResult]]
    ) -> list[tuple[Path, TrialResult]]:
        job_lock = self._read_source_job_lock()
        if job_lock is None:
            return trial_records

        records_by_key: dict[tuple[Any, ...], deque[tuple[Path, TrialResult]]] = (
            defaultdict(deque)
        )
        for record in sorted(trial_records, key=_trial_record_attempt_key):
            records_by_key[self._trial_result_order_key(record[1])].append(record)

        ordered_records: list[tuple[Path, TrialResult]] = []
        for trial_lock in job_lock.trials:
            queue = records_by_key.get(self._trial_lock_order_key(trial_lock))
            if queue:
                ordered_records.append(queue.popleft())

        if not ordered_records:
            return trial_records

        ordered_trial_names = {record[0].name for record in ordered_records}
        remaining_records = [
            record
            for record in trial_records
            if record[0].name not in ordered_trial_names
        ]
        return ordered_records + remaining_records

    def _read_source_job_lock(self) -> JobLock | None:
        lock_path = self.source_job_dir / LOCK_FILENAME
        if not lock_path.exists():
            return None
        try:
            return JobLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
        except Exception as e:
            global_logger.warning(
                f"Could not read source job lock {lock_path}; "
                f"falling back to trial directory order: {e}"
            )
            return None

    @staticmethod
    def _trial_result_order_key(result: TrialResult) -> tuple[Any, ...]:
        config = result.config
        return (
            _trial_result_task_order_key(config.task),
            _stable_json_key(config.agent),
            _stable_json_key(config.environment),
            _stable_json_key(config.verifier),
            config.timeout_multiplier,
            config.agent_timeout_multiplier,
            config.verifier_timeout_multiplier,
            config.agent_setup_timeout_multiplier,
            config.environment_build_timeout_multiplier,
        )

    @staticmethod
    def _trial_lock_order_key(trial_lock: TrialLock) -> tuple[Any, ...]:
        return (
            (
                trial_lock.task.type,
                trial_lock.task.name if trial_lock.task.type == "package" else None,
                trial_lock.task.digest if trial_lock.task.type == "package" else None,
                trial_lock.task.source,
                trial_lock.task.path.as_posix()
                if trial_lock.task.path is not None
                else None,
                trial_lock.task.git_url,
                trial_lock.task.git_commit_id,
            ),
            _stable_json_key(trial_lock.agent),
            _stable_json_key(trial_lock.environment),
            _stable_json_key(trial_lock.verifier),
            trial_lock.timeout_multiplier,
            trial_lock.agent_timeout_multiplier,
            trial_lock.verifier_timeout_multiplier,
            trial_lock.agent_setup_timeout_multiplier,
            trial_lock.environment_build_timeout_multiplier,
        )

    async def run_job(
        self,
        on_total: Callable[[int], None] | None = None,
        on_item_complete: Callable[[], None] | None = None,
    ) -> CritiqueJobResult:
        self._prepare_critique_dir()
        items = self.collect_items()
        if on_total is not None:
            on_total(len(items))

        (self.critique_dir / "config.json").write_text(
            self.config.model_dump_json(indent=4), encoding="utf-8"
        )

        job_result = CritiqueJobResult(
            run_name=self.config.run_name,
            source_job_dir=self.source_job_dir,
            critique_dir=self.critique_dir,
            config=self.config,
            started_at=datetime.now(timezone.utc),
        )

        semaphore = asyncio.Semaphore(self.config.n_concurrent)

        async def _run_one(item: CritiqueItem) -> None:
            try:
                async with semaphore:
                    result = await self.run_item(item)
                job_result.item_results.append(result)
            except Exception as e:
                job_result.failed_items.append(f"{item.source_trial_name}: {e}")
            finally:
                if on_item_complete is not None:
                    on_item_complete()
                self._write_job_result(job_result)

        async with asyncio.TaskGroup() as tg:
            for item in items:
                tg.create_task(_run_one(item))

        job_result.item_results.sort(key=lambda r: r.source_trial_name)
        job_result.failed_items.sort()
        job_result.finished_at = datetime.now(timezone.utc)
        self._write_job_result(job_result)
        return job_result

    def _prepare_critique_dir(self) -> None:
        config_path = self.critique_dir / "config.json"

        if self.config.overwrite and self.critique_dir.exists():
            shutil.rmtree(self.critique_dir)

        if config_path.exists():
            existing_config = CritiqueConfig.model_validate_json(
                config_path.read_text(encoding="utf-8")
            )
            if self._resume_key(existing_config) != self._resume_key(self.config):
                raise FileExistsError(
                    f"Critique run {self.critique_dir} already exists and cannot "
                    "be resumed with a different config. Use --overwrite to replace it."
                )
            return

        self.critique_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resume_key(config: CritiqueConfig) -> dict[str, Any]:
        return config.model_dump(
            exclude={"overwrite", "prompt_path", "n_concurrent", "limit"}
        )

    async def run_item(self, item: CritiqueItem) -> CritiqueItemResult:
        paths = CritiquePaths(self.critique_dir / item.source_trial_name)
        cached = paths.result_path
        if not self.config.overwrite and cached.exists():
            cached_result = self._load_cached_item_result(paths)
            if self._should_reuse_cached_result(cached_result):
                return cached_result

        if self.config.overwrite and paths.critique_dir.exists():
            shutil.rmtree(paths.critique_dir)

        trial = CritiqueTrial(self.config, item, self.critique_dir)
        return await trial.run()

    @staticmethod
    def _should_reuse_cached_result(result: CritiqueItemResult) -> bool:
        return result.exception_info is None and CritiqueTrial.is_valid_critique_result(
            result.critique_result
        )

    @staticmethod
    def _load_cached_item_result(paths: CritiquePaths) -> CritiqueItemResult:
        result = CritiqueItemResult.model_validate_json(
            paths.result_path.read_text(encoding="utf-8")
        )
        result_path = paths.artifacts_dir / CRITIQUE_RESULT_FILENAME
        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            parsed = None
        result.critique_result = parsed if isinstance(parsed, dict) else None
        return result

    def _write_job_result(self, result: CritiqueJobResult) -> None:
        (self.critique_dir / "result.json").write_text(
            result.model_dump_json(indent=4, exclude={"item_results"}),
            encoding="utf-8",
        )


class CritiqueTrial:
    _AGENT_SETUP_TIMEOUT_SEC = 360.0

    def __init__(
        self,
        config: CritiqueConfig,
        item: CritiqueItem,
        critique_dir: Path,
    ):
        self.config = config
        self.item = item
        self.critique_dir = critique_dir
        self.source_result = TrialResult.model_validate_json(
            (item.source_trial_dir / "result.json").read_text(encoding="utf-8")
        )
        self._task = Task(item.task_dir)
        self._critique_paths = CritiquePaths(critique_dir / item.source_trial_name)
        self._trial_paths = self._critique_paths.mount_paths
        self._critique_paths.mkdir()
        self._are_agent_logs_downloaded = False
        self._log_handler: logging.Handler | None = None
        self._init_logger()

        self._execution = TrialExecution.create(
            task=self._task,
            agent_config=config.agent,
            environment_config=config.environment,
            trial_paths=self._trial_paths,
            session_id=self._session_id(),
            logger=self._logger,
            timeout_multiplier=config.timeout_multiplier,
            agent_timeout_multiplier=config.agent_timeout_multiplier,
            agent_setup_timeout_multiplier=config.agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=config.environment_build_timeout_multiplier,
            default_agent_setup_timeout_sec=self._AGENT_SETUP_TIMEOUT_SEC,
        )
        self._agent = self._execution.agent
        self._environment = self._execution.environment
        self._rendered_instruction: str | None = None

        self._result = CritiqueItemResult(
            source_trial_name=item.source_trial_name,
            critique_trial_name=item.source_trial_name,
            task_name=self._task.name,
            task_id=self.source_result.task_id,
            task_checksum=self._task.checksum,
            source_trial_uri=item.source_trial_dir.expanduser().resolve().as_uri(),
            critique_trial_uri=self._critique_paths.critique_dir.expanduser()
            .resolve()
            .as_uri(),
            agent_info=self._agent.to_agent_info(),
        )

    @property
    def result(self) -> CritiqueItemResult:
        return self._result

    def _session_id(self) -> str:
        raw = f"crit__{self.config.run_name}__{self.item.source_trial_name}"
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        suffix = ShortUUID().random(length=5)
        prefix = cleaned[:20].rstrip("_.-") or "crit"
        return f"{prefix}__{digest}_{suffix}"

    def _init_logger(self) -> None:
        self._logger = global_logger.getChild(
            f"{__name__}.{self.config.run_name}.{self.item.source_trial_name}"
        )
        file_handler = logging.FileHandler(self._critique_paths.log_path)
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)
        self._log_handler = file_handler

    def _close_logger_handler(self) -> None:
        if self._log_handler is not None:
            self._logger.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    async def run(self) -> CritiqueItemResult:
        self._critique_paths.critique_dir.mkdir(parents=True, exist_ok=True)
        self._critique_paths.config_path.write_text(
            self.config.model_dump_json(indent=4), encoding="utf-8"
        )
        self.result.started_at = datetime.now(timezone.utc)

        try:
            await self._setup_environment()
            await self._environment.run_healthcheck()
            self._environment.default_user = self._task.config.agent.user
            try:
                await self._setup_agent()
                self.result.agent_info = self._agent.to_agent_info()
                await self._upload_critique_inputs()
                await self._execute_agent()
            finally:
                self._environment.default_user = None

            await self._maybe_download_logs(
                source_dir=self._environment.env_paths.agent_dir.as_posix(),
                target_dir=self._critique_paths.agent_dir,
            )
            self._redact_rendered_instruction_from_local_logs()
            self._maybe_populate_agent_context(self.result.agent_result)
            await self._download_artifacts()
            self._parse_critique_result()

        except asyncio.CancelledError as e:
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)
                self._critique_paths.exception_message_path.write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            await self._best_effort_download_outputs()
            raise

        except Exception as e:
            self._logger.debug(f"Critique trial failed: {e}")
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)
                self._critique_paths.exception_message_path.write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            await self._best_effort_download_outputs()

        finally:
            await self._cleanup_and_finalize()
            self._close_logger_handler()

        return self.result

    async def _setup_environment(self) -> None:
        self.result.environment_setup = TimingInfo(
            started_at=datetime.now(timezone.utc)
        )
        try:
            await self._execution.start_environment(
                force_build=self.config.environment.force_build
            )
        finally:
            self.result.environment_setup.finished_at = datetime.now(timezone.utc)

    async def _setup_agent(self) -> None:
        self.result.agent_setup = TimingInfo(started_at=datetime.now(timezone.utc))
        try:
            await self._execution.setup_agent()
        finally:
            self.result.agent_setup.finished_at = datetime.now(timezone.utc)

    async def _upload_critique_inputs(self) -> None:
        task_dir = self._critique_task_dir()
        trial_dir = self._critique_trial_dir()
        await self._environment.exec(
            "mkdir -p "
            f"{shlex.quote(task_dir.as_posix())} "
            f"{shlex.quote(trial_dir.as_posix())} "
            f"{shlex.quote(self._environment.env_paths.artifacts_dir.as_posix())}",
            user="root",
            timeout_sec=30,
        )
        await self._environment.upload_dir(self.item.task_dir, task_dir.as_posix())
        await self._environment.upload_dir(
            self.item.source_trial_dir, trial_dir.as_posix()
        )

    async def _execute_agent(self) -> None:
        self.result.agent_execution = TimingInfo(started_at=datetime.now(timezone.utc))
        try:
            self.result.agent_result = AgentContext()
            instruction = self._render_instruction()
            self._rendered_instruction = instruction
            await self._execution.run_agent(
                instruction=instruction,
                context=self.result.agent_result,
            )
        finally:
            self.result.agent_execution.finished_at = datetime.now(timezone.utc)

    def _render_instruction(self) -> str:
        critique_task_dir = self._critique_task_dir()
        critique_trial_dir = self._critique_trial_dir()
        critique_result_path = self._critique_result_path()
        critique_markdown_path = self._critique_markdown_path()
        values = {
            "task_dir": critique_task_dir.as_posix(),
            "trial_dir": critique_trial_dir.as_posix(),
            "source_trial_name": self.item.source_trial_name,
            "task_name": self._task.name,
            "critique_result_path": critique_result_path.as_posix(),
            "critique_markdown_path": critique_markdown_path.as_posix(),
        }
        prompt = _replace_template_vars(
            self.config.prompt_path.read_text(encoding="utf-8"), values
        )

        sections = [
            prompt.rstrip(),
            "Critique runtime contract:",
            f"- Task files are available at `{critique_task_dir.as_posix()}`.",
            f"- Source trial files are available at `{critique_trial_dir.as_posix()}`.",
            f"- Write a valid JSON object to `{critique_result_path.as_posix()}`.",
            f"- You may also write Markdown notes to `{critique_markdown_path.as_posix()}`.",
        ]
        return "\n\n".join(sections) + "\n"

    def _critique_input_dir(self) -> PurePosixPath:
        return (
            self._environment.env_paths.logs_dir.parent
            / CRITIQUE_DIRNAME
            / CRITIQUE_INPUT_DIRNAME
        )

    def _critique_task_dir(self) -> PurePosixPath:
        return self._critique_input_dir() / CRITIQUE_TASK_DIRNAME

    def _critique_trial_dir(self) -> PurePosixPath:
        return self._critique_input_dir() / CRITIQUE_TRIAL_DIRNAME

    def _critique_result_path(self) -> PurePosixPath:
        return self._environment.env_paths.artifacts_dir / CRITIQUE_RESULT_FILENAME

    def _critique_markdown_path(self) -> PurePosixPath:
        return self._environment.env_paths.artifacts_dir / CRITIQUE_MARKDOWN_FILENAME

    async def _maybe_download_logs(self, source_dir: str, target_dir: Path) -> None:
        if self._are_agent_logs_downloaded:
            return
        if self._environment.capabilities.mounted:
            await self._environment.prepare_logs_for_host()
            self._are_agent_logs_downloaded = True
            return
        try:
            await self._environment.download_dir(
                source_dir=source_dir, target_dir=target_dir
            )
        except Exception:
            self._logger.error(f"Failed to download logs to {target_dir}")
        self._are_agent_logs_downloaded = True

    def _maybe_populate_agent_context(self, agent_result: AgentContext | None) -> None:
        if (
            agent_result is None
            or not agent_result.is_empty()
            or not isinstance(self._agent, BaseInstalledAgent)
        ):
            return
        self._agent.populate_context_post_run(agent_result)

    def _redact_rendered_instruction_from_local_logs(self) -> None:
        instruction = self._rendered_instruction
        if not instruction:
            return

        candidate_paths = [self._critique_paths.log_path]
        if self._critique_paths.agent_dir.exists():
            candidate_paths.extend(
                path
                for path in self._critique_paths.agent_dir.rglob("*")
                if path.is_file()
            )

        for path in candidate_paths:
            if path.suffix == ".jsonl":
                self._redact_instruction_from_jsonl(path, instruction)
            else:
                self._redact_instruction_from_text(path, instruction)

    def _redact_instruction_from_text(self, path: Path, instruction: str) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        if instruction not in text:
            return
        path.write_text(
            text.replace(instruction, REDACTED_CRITIQUE_PROMPT),
            encoding="utf-8",
        )

    def _redact_instruction_from_jsonl(self, path: Path, instruction: str) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return

        changed = False
        redacted_lines: list[str] = []
        for line in lines:
            if not line.strip():
                redacted_lines.append(line)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                redacted_line = line.replace(instruction, REDACTED_CRITIQUE_PROMPT)
                changed = changed or redacted_line != line
                redacted_lines.append(redacted_line)
                continue

            redacted_payload, did_change = self._redact_instruction_value(
                payload, instruction
            )
            changed = changed or did_change
            redacted_lines.append(
                json.dumps(redacted_payload, sort_keys=True, separators=(",", ":"))
            )

        if changed:
            path.write_text("\n".join(redacted_lines) + "\n", encoding="utf-8")

    def _redact_instruction_value(
        self, value: Any, instruction: str
    ) -> tuple[Any, bool]:
        if isinstance(value, str):
            if instruction in value:
                return value.replace(instruction, REDACTED_CRITIQUE_PROMPT), True
            return value, False
        if isinstance(value, list):
            changed = False
            redacted_items = []
            for item in value:
                redacted_item, did_change = self._redact_instruction_value(
                    item, instruction
                )
                changed = changed or did_change
                redacted_items.append(redacted_item)
            return redacted_items, changed
        if isinstance(value, dict):
            changed = False
            redacted_dict = {}
            for key, item in value.items():
                redacted_item, did_change = self._redact_instruction_value(
                    item, instruction
                )
                changed = changed or did_change
                redacted_dict[key] = redacted_item
            return redacted_dict, changed
        return value, False

    async def _download_artifacts(self) -> None:
        self._critique_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not self._environment.capabilities.mounted:
            try:
                await self._environment.download_dir(
                    source_dir=self._environment.env_paths.artifacts_dir.as_posix(),
                    target_dir=self._critique_paths.artifacts_dir,
                )
            except Exception:
                self._logger.warning("Failed to download critique artifacts")
        self._write_artifacts_manifest()

    def _write_artifacts_manifest(self) -> None:
        entries = []
        for name, source in (
            (
                CRITIQUE_RESULT_FILENAME,
                self._critique_result_path().as_posix(),
            ),
            (
                CRITIQUE_MARKDOWN_FILENAME,
                self._critique_markdown_path().as_posix(),
            ),
        ):
            path = self._critique_paths.artifacts_dir / name
            entries.append(
                {
                    "source": source,
                    "destination": f"artifacts/{name}",
                    "type": "file",
                    "status": "ok" if path.exists() else "missing",
                }
            )
        self._critique_paths.artifacts_manifest_path.write_text(
            json.dumps(entries, indent=2), encoding="utf-8"
        )

    def _parse_critique_result(self) -> None:
        result_path = self._critique_paths.artifacts_dir / CRITIQUE_RESULT_FILENAME
        if not result_path.exists():
            raise FileNotFoundError(
                "Critique result was not written to "
                f"{self._critique_result_path().as_posix()}"
            )
        parsed = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("critique-result.json must contain a JSON object")
        errors = CritiqueTrial.critique_result_errors(parsed)
        if errors:
            raise ValueError(
                "critique-result.json is missing required first-party field(s): "
                + ", ".join(errors)
            )
        self.result.critique_result = parsed

    @staticmethod
    def is_valid_critique_result(result: dict[str, Any] | None) -> bool:
        return not CritiqueTrial.critique_result_errors(result)

    @staticmethod
    def critique_result_tags(result: dict[str, Any] | None) -> list[str]:
        if not isinstance(result, dict):
            return []

        tags: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value and value not in tags:
                tags.append(value)

        add(result.get("tag"))
        raw_tags = result.get("tags")
        if isinstance(raw_tags, str):
            add(raw_tags)
        elif isinstance(raw_tags, list):
            for tag in raw_tags:
                add(tag)
        return tags

    @staticmethod
    def critique_result_errors(result: dict[str, Any] | None) -> list[str]:
        if not isinstance(result, dict):
            return ["object"]

        errors: list[str] = []
        if not isinstance(result.get("feedback"), str) or not result.get("feedback"):
            errors.append("feedback")
        if result.get("rating") not in {"good", "bad"}:
            errors.append("rating")
        if not CritiqueTrial.critique_result_tags(result):
            errors.append("tag or tags")
        return errors

    async def _best_effort_download_outputs(self) -> None:
        try:
            await self._maybe_download_logs(
                source_dir=self._environment.env_paths.agent_dir.as_posix(),
                target_dir=self._critique_paths.agent_dir,
            )
            self._redact_rendered_instruction_from_local_logs()
            self._maybe_populate_agent_context(self.result.agent_result)
        except Exception:
            pass
        try:
            await self._download_artifacts()
            if self.result.critique_result is None:
                self._parse_critique_result()
        except Exception:
            pass

    async def _cleanup_and_finalize(self) -> None:
        try:
            await asyncio.shield(
                self._environment.stop(delete=self.config.environment.delete)
            )
        except asyncio.CancelledError:
            self._logger.warning(
                "Cleanup interrupted, but environment stop is shielded"
            )
        except Exception as e:
            self._logger.warning(f"Critique environment cleanup failed: {e}")
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)

        self.result.finished_at = datetime.now(timezone.utc)
        self._critique_paths.result_path.write_text(
            self.result.model_dump_json(indent=4, exclude={"critique_result"}),
            encoding="utf-8",
        )
