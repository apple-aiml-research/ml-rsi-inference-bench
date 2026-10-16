#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
import json
from datetime import datetime
from pathlib import Path

from pier.models.job.config import RetryConfig
from pier.models.job.lock import JobLock, TaskLock, TrialLock
from pier.critique.models import CritiqueConfig, CritiqueItemResult, CritiqueJobResult
from pier.critique.runner import CritiquePaths, CritiqueRunner, CritiqueTrial
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from pier.models.trial.result import AgentInfo, ExceptionInfo, TrialResult


def _critique_item_result(
    *,
    critique_result: dict | None,
    exception_info: ExceptionInfo | None = None,
) -> CritiqueItemResult:
    return CritiqueItemResult(
        source_trial_name="trial",
        critique_trial_name="trial",
        task_name="task",
        task_id=LocalTaskId(path=Path("/tmp/task")),
        task_checksum="checksum",
        source_trial_uri="file:///tmp/source",
        critique_trial_uri="file:///tmp/critique",
        agent_info=AgentInfo(name="agent", version="1"),
        critique_result=critique_result,
        exception_info=exception_info,
    )


def _write_source_trial(
    job_dir: Path,
    *,
    trial_name: str,
    task_dir: Path,
    task_checksum: str,
    agent: AgentConfig,
    started_at: datetime | None = None,
) -> None:
    task_config = TaskConfig(path=task_dir, source="test")
    trial_config = TrialConfig(
        task=task_config,
        trial_name=trial_name,
        trials_dir=job_dir,
        agent=agent,
        environment=EnvironmentConfig(),
        verifier=VerifierConfig(),
    )
    trial_result = TrialResult(
        task_name=task_dir.name,
        trial_name=trial_name,
        trial_uri=(job_dir / trial_name).resolve().as_uri(),
        task_id=LocalTaskId(path=task_dir),
        source="test",
        task_checksum=task_checksum,
        config=trial_config,
        agent_info=AgentInfo(name=agent.name or "agent", version="1"),
        started_at=started_at,
    )
    trial_dir = job_dir / trial_name
    trial_dir.mkdir()
    (trial_dir / "trial.log").write_text("", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        trial_result.model_dump_json(indent=4), encoding="utf-8"
    )


def _trial_lock(
    *,
    task_dir: Path,
    task_checksum: str,
    agent: AgentConfig,
) -> TrialLock:
    return TrialLock(
        task=TaskLock(
            name=task_dir.name,
            type="local",
            digest=f"sha256:{task_checksum}",
            path=task_dir,
            source="test",
        ),
        agent=agent,
        environment=EnvironmentConfig(),
        verifier=VerifierConfig(),
    )


def test_reuse_clean_cached_critique_result():
    result = _critique_item_result(
        critique_result={
            "rating": "good",
            "tag": "PASS_LEGITIMATE",
            "feedback": "Looks good.",
        }
    )

    assert CritiqueRunner._should_reuse_cached_result(result)


def test_retry_cached_critique_result_with_exception():
    result = _critique_item_result(
        critique_result={
            "rating": "good",
            "tag": "PASS_LEGITIMATE",
            "feedback": "Looks good.",
        },
        exception_info=ExceptionInfo(
            exception_type="CancelledError",
            exception_message="",
            exception_traceback="traceback",
            occurred_at=datetime.now(),
        ),
    )

    assert not CritiqueRunner._should_reuse_cached_result(result)


def test_retry_cached_metadata_without_critique_result():
    result = _critique_item_result(critique_result=None)

    assert not CritiqueRunner._should_reuse_cached_result(result)


def test_retry_cached_metadata_without_required_critique_result_fields():
    result = _critique_item_result(critique_result={"ok": True})

    assert not CritiqueRunner._should_reuse_cached_result(result)


def test_validate_critique_result_contract():
    assert CritiqueTrial.is_valid_critique_result(
        {
            "rating": "good",
            "tag": "FAIL_WRONG_LOGIC",
            "feedback": "The implementation is wrong.",
        }
    )
    assert CritiqueTrial.is_valid_critique_result(
        {
            "rating": "bad",
            "tag": "FAIL_TEST_BROKEN",
            "tags": ["task_unfair"],
            "feedback": "The task is broken.",
        }
    )
    assert not CritiqueTrial.is_valid_critique_result({"feedback": "Old shape."})
    assert not CritiqueTrial.is_valid_critique_result(
        {"rating": "good", "tag": "FAIL_WRONG_LOGIC"}
    )
    assert not CritiqueTrial.is_valid_critique_result(
        {"rating": "unknown", "tag": "FAIL_WRONG_LOGIC", "feedback": "No."}
    )


def test_prepare_critique_dir_does_not_copy_prompt(tmp_path):
    source_job_dir = tmp_path / "job"
    prompt_path = tmp_path / "prompts" / "critique.md"
    prompt_path.parent.mkdir()
    prompt_path.write_text("Prompt body", encoding="utf-8")

    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=prompt_path,
            agent=AgentConfig(),
        )
    )

    runner._prepare_critique_dir()

    assert runner.config.prompt_path == prompt_path
    assert not (runner.critique_dir / "prompt.md").exists()


def test_write_job_result_does_not_embed_item_results(tmp_path):
    source_job_dir = tmp_path / "job"
    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=tmp_path / "prompt.md",
            agent=AgentConfig(),
        )
    )
    runner._prepare_critique_dir()
    result = CritiqueJobResult(
        run_name="run",
        source_job_dir=source_job_dir,
        critique_dir=runner.critique_dir,
        config=runner.config,
        item_results=[
            _critique_item_result(
                critique_result={
                    "rating": "good",
                    "tag": "PASS_LEGITIMATE",
                    "feedback": "Looks good.",
                }
            )
        ],
    )

    runner._write_job_result(result)

    payload = json.loads((runner.critique_dir / "result.json").read_text())
    assert "item_results" not in payload
    assert payload["n_items"] == 1
    assert payload["n_failed"] == 0


def test_cached_item_result_reads_artifact_as_source_of_truth(tmp_path):
    paths = CritiquePaths(tmp_path / "trial")
    paths.critique_dir.mkdir(parents=True)
    paths.artifacts_dir.mkdir(parents=True)
    metadata = _critique_item_result(critique_result=None)
    paths.result_path.write_text(
        metadata.model_dump_json(indent=4, exclude={"critique_result"}),
        encoding="utf-8",
    )
    artifact = {
        "rating": "bad",
        "tag": "TASK_UNFAIR",
        "feedback": "The task is unfair.",
    }
    (paths.artifacts_dir / "critique-result.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )

    result = CritiqueRunner._load_cached_item_result(paths)

    assert result.critique_result == artifact
    assert CritiqueRunner._should_reuse_cached_result(result)
    assert "critique_result" not in json.loads(paths.result_path.read_text())


def test_resume_uses_current_prompt_path(tmp_path):
    source_job_dir = tmp_path / "job"
    original_prompt = tmp_path / "prompts" / "original.md"
    current_prompt = tmp_path / "prompts" / "current.md"
    original_prompt.parent.mkdir()
    original_prompt.write_text("Original prompt", encoding="utf-8")
    current_prompt.write_text("Current prompt", encoding="utf-8")

    original_runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=original_prompt,
            agent=AgentConfig(),
        )
    )
    original_runner._prepare_critique_dir()
    (original_runner.critique_dir / "config.json").write_text(
        original_runner.config.model_dump_json(indent=4), encoding="utf-8"
    )

    resumed_runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=current_prompt,
            agent=AgentConfig(),
        )
    )
    resumed_runner._prepare_critique_dir()

    assert resumed_runner.config.prompt_path == current_prompt


def test_resume_allows_changing_concurrency(tmp_path):
    source_job_dir = tmp_path / "job"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Prompt", encoding="utf-8")

    original_runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=prompt_path,
            agent=AgentConfig(),
            n_concurrent=1,
        )
    )
    original_runner._prepare_critique_dir()
    (original_runner.critique_dir / "config.json").write_text(
        original_runner.config.model_dump_json(indent=4), encoding="utf-8"
    )

    resumed_runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=prompt_path,
            agent=AgentConfig(),
            n_concurrent=100,
        )
    )
    resumed_runner._prepare_critique_dir()

    assert resumed_runner.config.n_concurrent == 100


def test_resume_allows_changing_limit(tmp_path):
    source_job_dir = tmp_path / "job"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Prompt", encoding="utf-8")

    original_runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=prompt_path,
            agent=AgentConfig(),
            sample_seed=13,
            limit=100,
        )
    )
    original_runner._prepare_critique_dir()
    (original_runner.critique_dir / "config.json").write_text(
        original_runner.config.model_dump_json(indent=4), encoding="utf-8"
    )

    resumed_runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=source_job_dir,
            prompt_path=prompt_path,
            agent=AgentConfig(),
            sample_seed=13,
            limit=None,
        )
    )
    resumed_runner._prepare_critique_dir()

    assert resumed_runner.config.limit is None


def test_collect_items_prefers_source_job_lock_order(tmp_path):
    job_dir = tmp_path / "job"
    task_a = tmp_path / "tasks" / "a"
    task_b = tmp_path / "tasks" / "b"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)
    job_dir.mkdir()
    (job_dir / "job.log").write_text("", encoding="utf-8")

    checksum_a = "a" * 64
    checksum_b = "b" * 64
    agent = AgentConfig(name="codex", model_name="gpt-5.5")
    _write_source_trial(
        job_dir,
        trial_name="trial-a",
        task_dir=task_a,
        task_checksum=checksum_a,
        agent=agent,
    )
    _write_source_trial(
        job_dir,
        trial_name="trial-z",
        task_dir=task_b,
        task_checksum=checksum_b,
        agent=agent,
    )
    job_lock = JobLock(
        n_concurrent_trials=1,
        retry=RetryConfig(),
        trials=[
            _trial_lock(task_dir=task_b, task_checksum="c" * 64, agent=agent),
            _trial_lock(task_dir=task_a, task_checksum="d" * 64, agent=agent),
        ],
    )
    (job_dir / "lock.json").write_text(
        job_lock.model_dump_json(indent=4), encoding="utf-8"
    )

    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=job_dir,
            prompt_path=tmp_path / "prompt.md",
            agent=AgentConfig(),
        )
    )

    assert [item.source_trial_name for item in runner.collect_items()] == [
        "trial-z",
        "trial-a",
    ]


def test_collect_items_orders_duplicate_attempts_by_start_time(tmp_path):
    job_dir = tmp_path / "job"
    task_dir = tmp_path / "tasks" / "task"
    task_dir.mkdir(parents=True)
    job_dir.mkdir()
    (job_dir / "job.log").write_text("", encoding="utf-8")

    checksum = "a" * 64
    agent = AgentConfig(name="codex", model_name="gpt-5.5")
    _write_source_trial(
        job_dir,
        trial_name="trial-a",
        task_dir=task_dir,
        task_checksum=checksum,
        agent=agent,
        started_at=datetime(2026, 1, 2),
    )
    _write_source_trial(
        job_dir,
        trial_name="trial-z",
        task_dir=task_dir,
        task_checksum=checksum,
        agent=agent,
        started_at=datetime(2026, 1, 1),
    )
    job_lock = JobLock(
        n_concurrent_trials=1,
        retry=RetryConfig(),
        trials=[
            _trial_lock(task_dir=task_dir, task_checksum=checksum, agent=agent),
            _trial_lock(task_dir=task_dir, task_checksum=checksum, agent=agent),
        ],
    )
    (job_dir / "lock.json").write_text(
        job_lock.model_dump_json(indent=4), encoding="utf-8"
    )

    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=job_dir,
            prompt_path=tmp_path / "prompt.md",
            agent=AgentConfig(),
        )
    )

    assert [item.source_trial_name for item in runner.collect_items()] == [
        "trial-z",
        "trial-a",
    ]


def test_collect_items_falls_back_to_trial_dir_order_without_lock(tmp_path):
    job_dir = tmp_path / "job"
    task_a = tmp_path / "tasks" / "a"
    task_b = tmp_path / "tasks" / "b"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)
    job_dir.mkdir()
    (job_dir / "job.log").write_text("", encoding="utf-8")

    agent = AgentConfig(name="codex", model_name="gpt-5.5")
    _write_source_trial(
        job_dir,
        trial_name="trial-z",
        task_dir=task_b,
        task_checksum="b" * 64,
        agent=agent,
    )
    _write_source_trial(
        job_dir,
        trial_name="trial-a",
        task_dir=task_a,
        task_checksum="a" * 64,
        agent=agent,
    )

    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=job_dir,
            prompt_path=tmp_path / "prompt.md",
            agent=AgentConfig(),
        )
    )

    assert [item.source_trial_name for item in runner.collect_items()] == [
        "trial-a",
        "trial-z",
    ]


def test_collect_items_filters_by_source_model_without_matching_prefixes(tmp_path):
    job_dir = tmp_path / "job"
    task_a = tmp_path / "tasks" / "a"
    task_b = tmp_path / "tasks" / "b"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)
    job_dir.mkdir()
    (job_dir / "job.log").write_text("", encoding="utf-8")

    _write_source_trial(
        job_dir,
        trial_name="trial-a",
        task_dir=task_a,
        task_checksum="a" * 64,
        agent=AgentConfig(name="mini-swe-agent", model_name="openai/gpt-5.4"),
    )
    _write_source_trial(
        job_dir,
        trial_name="trial-b",
        task_dir=task_b,
        task_checksum="b" * 64,
        agent=AgentConfig(name="mini-swe-agent", model_name="openai/gpt-5.4-mini"),
    )

    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=job_dir,
            prompt_path=tmp_path / "prompt.md",
            agent=AgentConfig(),
            source_model_names=["gpt-5.4"],
        )
    )

    assert [item.source_trial_name for item in runner.collect_items()] == ["trial-a"]


def test_collect_items_filters_by_source_agent(tmp_path):
    job_dir = tmp_path / "job"
    task_a = tmp_path / "tasks" / "a"
    task_b = tmp_path / "tasks" / "b"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)
    job_dir.mkdir()
    (job_dir / "job.log").write_text("", encoding="utf-8")

    _write_source_trial(
        job_dir,
        trial_name="trial-a",
        task_dir=task_a,
        task_checksum="a" * 64,
        agent=AgentConfig(name="mini-swe-agent", model_name="gpt-5.5"),
    )
    _write_source_trial(
        job_dir,
        trial_name="trial-b",
        task_dir=task_b,
        task_checksum="b" * 64,
        agent=AgentConfig(name="codex", model_name="gpt-5.5"),
    )

    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=job_dir,
            prompt_path=tmp_path / "prompt.md",
            agent=AgentConfig(),
            source_agent_names=["codex"],
        )
    )

    assert [item.source_trial_name for item in runner.collect_items()] == ["trial-b"]


def test_collect_items_applies_sample_seed_before_limit(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.log").write_text("", encoding="utf-8")

    agent = AgentConfig(name="mini-swe-agent", model_name="gpt-5.5")
    for suffix in ["a", "b", "c", "d"]:
        task_dir = tmp_path / "tasks" / suffix
        task_dir.mkdir(parents=True)
        _write_source_trial(
            job_dir,
            trial_name=f"trial-{suffix}",
            task_dir=task_dir,
            task_checksum=suffix * 64,
            agent=agent,
        )

    runner = CritiqueRunner(
        CritiqueConfig(
            run_name="run",
            source_job_dir=job_dir,
            prompt_path=tmp_path / "prompt.md",
            agent=AgentConfig(),
            sample_seed=1,
            limit=2,
        )
    )

    assert [item.source_trial_name for item in runner.collect_items()] == [
        "trial-d",
        "trial-a",
    ]


def test_redacts_rendered_instruction_from_codex_jsonl(tmp_path):
    instruction = "private critique prompt\nwith runtime paths"
    session_path = tmp_path / "session.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": instruction}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    trial = CritiqueTrial.__new__(CritiqueTrial)
    trial._redact_instruction_from_jsonl(session_path, instruction)

    redacted = session_path.read_text(encoding="utf-8")
    assert instruction not in redacted
    assert "[redacted critique prompt]" in redacted
    assert "done" in redacted
