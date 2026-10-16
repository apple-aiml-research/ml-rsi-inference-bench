#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for regression_multimodal_prefill_ttft: mean TTFT over a served workload matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import py_compile
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib.bench_stats import usable_round_arms  # noqa: E402
from _verifier_lib.metrics import normalize_engine_metrics  # noqa: E402
from _verifier_lib.regression_scoring import (  # noqa: E402
    CONTRACT as SCORING_CONTRACT,
    compute_regression_reward,
    resolve_rounds_ratio,
)



TARGET_REL = "repo/sglang/kernels/ops/attention/fla/l2norm.py"

BASELINE_REPO_REL = "baseline_repo"
CANDIDATE_REPO_REL = "repo"

REQUIRED_TERMS = (
    'do_not_specialize=["T"]',
)

SCORED_METRIC_KEY = "latency_ms.ttft.mean"


def _scored_value(summary: object) -> float | None:
    """SCORED_METRIC_KEY off one normalised summary: finite and positive, else None."""
    node = summary
    for part in SCORED_METRIC_KEY.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    return float(node) if math.isfinite(node) and node > 0 else None

REGIME_MIN_DISTINCT_PROMPT_TOKEN_COUNTS = 110
REGIME_MIN_DISTINCT_T = 150
REGIME_MIN_FIRST_SIGHT_IN_WINDOW = 100

PRESTAGE_MAX_CUBINS_AT_T0_VS_PRISTINE = 3.0

REPAIR_MAX_CANDIDATE_CUBIN_FRACTION = 0.5


def _candidate_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _py_shas(root: Path) -> tuple[dict[str, str], list[str]] | None:
    """``({relpath: sha256}, sorted unreadable relpaths)`` for every `.py` under `root`."""
    if not root.is_dir():
        return None
    out: dict[str, str] = {}
    unreadable: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        rel = str(path.relative_to(root))
        try:
            out[rel] = _candidate_sha(path)
        except OSError:
            unreadable.append(rel)
    if not out and not unreadable:
        return None
    return out, sorted(unreadable)


def _edited_paths(workspace: Path) -> list[str] | None:
    """Every `.py` under `repo/` that differs from the freshly-staged pristine tree."""
    return _tree_diff(workspace)[0]


def _tree_diff(workspace: Path) -> tuple[list[str] | None, list[str]]:
    """``(edited paths or None, arm-qualified unreadable paths)`` -- ONE walk of both trees."""
    baseline = _py_shas(workspace / BASELINE_REPO_REL)
    candidate = _py_shas(workspace / CANDIDATE_REPO_REL)
    if baseline is None or candidate is None:
        return None, []
    baseline_shas, baseline_unreadable = baseline
    candidate_shas, candidate_unreadable = candidate
    unreadable = sorted(
        {f"{BASELINE_REPO_REL}/{rel}" for rel in baseline_unreadable}
        | {f"{CANDIDATE_REPO_REL}/{rel}" for rel in candidate_unreadable}
    )
    if unreadable:
        return None, unreadable
    edited = sorted(
        rel for rel in set(baseline_shas) | set(candidate_shas)
        if baseline_shas.get(rel) != candidate_shas.get(rel)
    )
    return edited, unreadable


def check_candidate(workspace: Path) -> dict[str, object]:
    candidate = workspace / TARGET_REL
    checks: dict[str, object] = {
        "target_path": TARGET_REL,
        "target_exists": candidate.exists(),
    }

    edited, unreadable = _tree_diff(workspace)
    checks["edited_paths"] = edited
    checks["unreadable_paths"] = unreadable
    checks["file_edited"] = None if (unreadable or edited is None) else bool(edited)

    if candidate.exists():
        try:
            py_compile.compile(str(candidate), doraise=True)
            checks["target_compiles"] = True
        except py_compile.PyCompileError as exc:
            checks["target_compiles"] = False
            checks["compile_error"] = str(exc)
        text = candidate.read_text(encoding="utf-8", errors="replace")
        checks["required_terms"] = {term: term in text for term in REQUIRED_TERMS}
        checks["candidate_sha256"] = _candidate_sha(candidate)
    else:
        checks["target_compiles"] = False
        checks["required_terms"] = {term: False for term in REQUIRED_TERMS}

    return checks


def score(checks: dict[str, object]) -> tuple[float, float]:
    booleans: list[bool] = [
        bool(checks.get("target_exists")),
        bool(checks.get("target_compiles")),
    ]
    booleans.append(checks.get("file_edited") is True)

    correctness = sum(1 for b in booleans if b) / len(booleans)
    reward = 1.0 if correctness == 1.0 else 0.0
    return reward, correctness


REQUIRED_ENGINE_PATHS = (
    "pyproject.toml",
    "sglang",
    "sglang/srt/server_args.py",
    "sglang/srt/managers/scheduler.py",
    "sglang/srt/model_executor/model_runner.py",
    "sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
)

DEFAULT_PERFORMANCE_THRESHOLDS = {
    "latency": {
        "max_e2e_mean_ms": 600_000.0,
    },
    "throughput": {
        "min_request_per_sec": 0.01,
        "min_total_tokens_per_sec": 1.0,
    },
    "serve": {
        "min_request_per_sec": 0.01,
        "min_total_tokens_per_sec": 1.0,
        "max_ttft_mean_ms": 60_000.0,
    },
}

NOISE_FLOOR = 1.03
SCORED_LOWER_IS_BETTER = True


def locate_candidate_repo(workspace: Path) -> Path:
    env_path = os.environ.get("INFB_CANDIDATE_REPO_DIR")
    if env_path:
        return Path(env_path).resolve()
    return workspace / "repo"


def repo_commit(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def check_real_engine_repo(repo: Path) -> dict[str, object]:
    checks: dict[str, object] = {
        "repo_path": str(repo),
        "repo_exists": repo.exists(),
        "git_commit": repo_commit(repo),
        "required_paths": {},
    }
    checks["required_paths"] = {rel: (repo / rel).exists() for rel in REQUIRED_ENGINE_PATHS}
    pyproject = repo / "pyproject.toml"
    checks["pyproject_declares_engine"] = pyproject.exists() and 'name = "sglang"' in (
        pyproject.read_text(encoding="utf-8", errors="replace")
    )
    return checks


def all_real_checks_pass(checks: dict[str, object]) -> bool:
    required = checks.get("required_paths", {})
    return (
        bool(checks.get("repo_exists"))
        and bool(checks.get("pyproject_declares_engine"))
        and isinstance(required, dict)
        and all(bool(value) for value in required.values())
    )


RAW_RECORD_NAMES = ("sglang_bench_serving.json", "sglang_bench_throughput.json")
RAW_MANIFEST_NAME = "sglang_bench_manifest.json"


def _first_record(directory: Path) -> Path:
    """The bench record in `directory`: the first of RAW_RECORD_NAMES that exists, else the first name
    (so a missing record reads as a missing file rather than as a KeyError)."""
    for name in RAW_RECORD_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return directory / RAW_RECORD_NAMES[0]


def _expected_request_count() -> int | None:
    """The anchor's `num_prompts` from the trusted tests/workloads.json beside this file."""
    path = Path(__file__).resolve().parent / "workloads.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    primary = doc.get("primary_workload")
    for entry in doc.get("workloads", []):
        if isinstance(entry, dict) and entry.get("name") == primary:
            value = entry.get("num_prompts")
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return None


def check_upstream_benchmark(raw_dir: Path) -> dict[str, object]:
    raw_throughput = _first_record(raw_dir)
    manifest = raw_dir / RAW_MANIFEST_NAME
    checks: dict[str, object] = {
        "raw_throughput_path": str(raw_throughput),
        "manifest_path": str(manifest),
        "raw_exists": raw_throughput.exists(),
        "manifest_exists": manifest.exists(),
    }
    if raw_throughput.exists():
        checks["raw_path"] = str(raw_throughput)
        try:
            raw = json.loads(raw_throughput.read_text(encoding="utf-8"))
            checks["raw_json_object"] = isinstance(raw, dict)
            checks["raw_keys"] = sorted(raw) if isinstance(raw, dict) else []
            if isinstance(raw, dict):
                checks["engine_package"] = raw.get("engine_package")
                checks["engine_kwargs"] = raw.get("engine_kwargs") or raw.get("engine_args")
                expected = _expected_request_count()
                completed, failed = raw.get("completed"), raw.get("failed")
                checks["expected_requests"] = expected
                checks["completed_requests"] = completed
                checks["failed_requests"] = failed
                checks["workload_delivered"] = (
                    isinstance(expected, int) and isinstance(completed, int)
                    and not isinstance(completed, bool) and completed == expected
                    and (failed is None or failed == 0)
                )
                regime = raw.get("regime")
                checks["regime"] = regime
                if isinstance(regime, dict):
                    calls = regime.get("l2norm_calls")
                    distinct_t = regime.get("l2norm_distinct_T")
                    first_sight = regime.get("l2norm_first_sight_in_scored_window")
                    distinct_prompts = regime.get("distinct_prompt_token_counts")
                    checks["regime_path_exercised"] = isinstance(calls, int) and calls > 0
                    checks["regime_distinct_T_ok"] = (
                        isinstance(distinct_t, int)
                        and distinct_t >= REGIME_MIN_DISTINCT_T
                    )
                    checks["regime_first_sight_ok"] = (
                        isinstance(first_sight, int)
                        and first_sight >= REGIME_MIN_FIRST_SIGHT_IN_WINDOW
                    )
                    checks["regime_distinct_prompts_ok"] = (
                        isinstance(distinct_prompts, int)
                        and distinct_prompts >= REGIME_MIN_DISTINCT_PROMPT_TOKEN_COUNTS
                    )
                    checks["regime_floors"] = {
                        "distinct_T": REGIME_MIN_DISTINCT_T,
                        "first_sight_in_scored_window": REGIME_MIN_FIRST_SIGHT_IN_WINDOW,
                        "distinct_prompt_token_counts":
                            REGIME_MIN_DISTINCT_PROMPT_TOKEN_COUNTS,
                    }
                else:
                    checks["regime_path_exercised"] = False
                    checks["regime_distinct_T_ok"] = False
                    checks["regime_first_sight_ok"] = False
                    checks["regime_distinct_prompts_ok"] = False
        except json.JSONDecodeError as exc:
            checks["raw_json_object"] = False
            checks["raw_error"] = str(exc)
    if manifest.exists():
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            checks["manifest_json_object"] = isinstance(manifest_data, dict)
            if isinstance(manifest_data, dict):
                checks["manifest_returncode"] = manifest_data.get("returncode")
                checks["manifest_timed_out"] = manifest_data.get("timed_out")
                command = manifest_data.get("command")
                checks["manifest_command"] = command
                checks["manifest_mentions_engine_bench"] = (
                    isinstance(command, str) and "sglang" in command and "bench" in command
                )
        except json.JSONDecodeError as exc:
            checks["manifest_json_object"] = False
            checks["manifest_error"] = str(exc)
    artifacts = raw_dir.parent
    candidate_at_t0 = _arm_regime_int_with_fallback(
        raw_dir, artifacts / "rounds", "candidate", "triton_cubins_at_t0")
    pristine_at_t0 = _arm_regime_int_with_fallback(
        artifacts / "baseline" / "raw", artifacts / "rounds", "pristine", "triton_cubins_at_t0")
    checks["candidate_cubins_at_t0"] = candidate_at_t0
    checks["pristine_cubins_at_t0"] = pristine_at_t0
    checks["prestage_max_ratio_vs_pristine"] = PRESTAGE_MAX_CUBINS_AT_T0_VS_PRISTINE
    checks["prestage_ok"] = (
        isinstance(candidate_at_t0, int) and isinstance(pristine_at_t0, int) and pristine_at_t0 > 0
        and candidate_at_t0 <= PRESTAGE_MAX_CUBINS_AT_T0_VS_PRISTINE * pristine_at_t0
    )
    return checks


def upstream_benchmark_ready(checks: dict[str, object]) -> bool:
    return (
        bool(checks.get("raw_exists"))
        and bool(checks.get("manifest_exists"))
        and bool(checks.get("raw_json_object"))
        and bool(checks.get("manifest_json_object"))
        and checks.get("manifest_returncode") == 0
        and bool(checks.get("manifest_mentions_engine_bench"))
        and bool(checks.get("workload_delivered"))
        and bool(checks.get("regime_distinct_prompts_ok"))
        and bool(checks.get("prestage_ok"))
    )


def _arm_regime_int(raw_dir: Path, key: str) -> int | None:
    """`regime.<key>` for one arm's published raw, or None when it cannot be read."""
    path = _first_record(raw_dir)
    if not path.is_file():
        return None
    try:
        regime = json.loads(path.read_text(encoding="utf-8")).get("regime")
    except (json.JSONDecodeError, OSError):
        return None
    value = regime.get(key) if isinstance(regime, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _arm_cubins(raw_dir: Path) -> int | None:
    return _arm_regime_int(raw_dir, "triton_cubins")


def _arm_regime_int_with_fallback(published: Path, rounds_dir: Path, arm: str, key: str) -> int | None:
    """`_arm_regime_int` on the published copy, then the per-round copies of the same arm, newest first."""
    value = _arm_regime_int(published, key)
    if value is not None:
        return value
    if not rounds_dir.is_dir():
        return None
    for round_dir in sorted(rounds_dir.iterdir(), reverse=True):
        value = _arm_regime_int(round_dir / arm / "raw", key)
        if value is not None:
            return value
    return None


def _arm_cubins_with_fallback(published: Path, rounds_dir: Path, arm: str) -> int | None:
    """`_arm_cubins` on the published copy, falling back to the per-round copies of the same arm."""
    return _arm_regime_int_with_fallback(published, rounds_dir, arm, "triton_cubins")


def check_regression_repaired(raw_dir: Path) -> dict[str, object]:
    """Did the CANDIDATE arm compile materially fewer kernels than the PRISTINE arm?"""
    artifacts = raw_dir.parent
    pristine_dir = artifacts / "baseline" / "raw"
    rounds_dir = artifacts / "rounds"
    candidate = _arm_cubins_with_fallback(raw_dir, rounds_dir, "candidate")
    pristine = _arm_cubins_with_fallback(pristine_dir, rounds_dir, "pristine")
    checks: dict[str, object] = {
        "candidate_triton_cubins": candidate,
        "pristine_triton_cubins": pristine,
        "max_candidate_fraction": REPAIR_MAX_CANDIDATE_CUBIN_FRACTION,
        "candidate_raw": str(_first_record(raw_dir)),
        "pristine_raw": str(_first_record(pristine_dir)),
    }
    if candidate is None or pristine is None or pristine <= 0:
        checks["cubin_fraction"] = None
        checks["repaired"] = False
        return checks
    fraction = candidate / pristine
    checks["cubin_fraction"] = fraction
    checks["candidate_compiled_something"] = candidate > 0
    checks["repaired"] = candidate > 0 and fraction <= REPAIR_MAX_CANDIDATE_CUBIN_FRACTION
    return checks


def regression_repaired(checks: dict[str, object]) -> bool:
    return bool(checks.get("repaired"))


CORRECTNESS_ATOL = 0.05
CORRECTNESS_RTOL = 0.05
CORRECTNESS_MIN_PASS_RATE = 0.95


def _correctness_block(directory: Path) -> dict | None:
    path = _first_record(directory)
    if not path.is_file():
        return None
    try:
        block = json.loads(path.read_text(encoding="utf-8")).get("correctness_check")
    except (json.JSONDecodeError, OSError):
        return None
    return block if isinstance(block, dict) else None


def _arm_logprobs(published: Path, rounds_dir: Path, arm: str) -> tuple[list | None, str | None]:
    for directory in [published] + (
            [d / arm / "raw" for d in sorted(rounds_dir.iterdir(), reverse=True)]
            if rounds_dir.is_dir() else []):
        block = _correctness_block(directory)
        if block is None:
            continue
        lp = block.get("logprobs")
        if isinstance(lp, list) and lp:
            return lp, block.get("error")
    return None, "no correctness_check logprobs in this arm's raw"


def check_outputs_agree(raw_dir: Path) -> dict[str, object]:
    """Does the candidate assign the same logprob to the same token in the same context?"""
    artifacts = raw_dir.parent
    rounds = artifacts / "rounds"
    cand, cand_err = _arm_logprobs(raw_dir, rounds, "candidate")
    prist, prist_err = _arm_logprobs(artifacts / "baseline" / "raw", rounds, "pristine")
    checks: dict[str, object] = {
        "atol": CORRECTNESS_ATOL, "rtol": CORRECTNESS_RTOL,
        "min_pass_rate": CORRECTNESS_MIN_PASS_RATE,
        "candidate_error": cand_err, "pristine_error": prist_err,
        "candidate_measured": isinstance(cand, list) and bool(cand),
        "pristine_measured": isinstance(prist, list) and bool(prist),
    }
    if not isinstance(cand, list) or not isinstance(prist, list) or not cand or not prist:
        checks["compared"] = 0
        checks["pass_rate"] = None
        checks["outputs_agree"] = False
        return checks
    compared = agreed = 0
    worst = 0.0
    for a, b in zip(cand, prist):
        if not isinstance(a, list) or not isinstance(b, list):
            continue
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            compared += 1
            delta = abs(float(x) - float(y))
            worst = max(worst, delta)
            if delta <= CORRECTNESS_ATOL + CORRECTNESS_RTOL * abs(float(y)):
                agreed += 1
    checks["compared"] = compared
    checks["agreed"] = agreed
    checks["max_abs_delta"] = worst
    if compared == 0:
        checks["pass_rate"] = None
        checks["outputs_agree"] = False
        return checks
    rate = agreed / compared
    checks["pass_rate"] = rate
    checks["outputs_agree"] = rate >= CORRECTNESS_MIN_PASS_RATE


    return checks


def outputs_agree(checks: dict[str, object]) -> bool:
    return bool(checks.get("outputs_agree"))


def _unmeasured_metrics(metrics_path: Path) -> dict[str, object]:
    """Metrics for a run that produced NO bench output."""
    metrics: dict[str, object] = {
        "schema_version": 1,
        "metric_source": "sglang",
        "source_path": None,
        "provenance": {"unmeasured": True},
        "summary": {},
        "records": [],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def normalize_measured_or_unmeasured(raw_dir: Path, metrics_path: Path) -> dict[str, object]:
    raw_throughput = _first_record(raw_dir)
    if raw_throughput.exists():
        return normalize_engine_metrics(raw_throughput, metrics_path, metric_source="sglang")
    return _unmeasured_metrics(metrics_path)


def load_baseline_tps(tests_dir: Path, raw_dir: Path | None = None) -> float | None:
    """The regressed-state baseline value on the scored key: mean TTFT in ms (the `*_tps` name is the
    shared resolver's)."""
    if raw_dir is None:
        return None
    rounds_mean = _round_arm_mean(load_bench_rounds(raw_dir), "pristine")
    if rounds_mean is not None:
        return rounds_mean
    bench_file = _first_record(raw_dir.parent / "baseline" / "raw")
    if not bench_file.exists():
        return None
    try:
        tmp_metrics = normalize_engine_metrics(
            bench_file,
            bench_file.parent / "_baseline_metrics.json",
            metric_source="sglang",
        )
        return _scored_value(tmp_metrics.get("summary", {}))
    except Exception:  # noqa: BLE001 -- a malformed baseline is "no baseline", not a crash
        return None


def _task_toml_path() -> Path | None:
    """Locate the task's task.toml."""
    source_task = os.environ.get("INFB_SOURCE_TASK_DIR")
    if source_task:
        candidate = Path(source_task) / "task.toml"
        if candidate.exists():
            return candidate
    candidate = Path(__file__).resolve().parent.parent / "task.toml"
    return candidate if candidate.exists() else None


def load_performance_thresholds() -> dict[str, dict[str, float]]:
    """The ABSOLUTE floors (req/s, tok/s, latency caps): the family defaults, refined by task.toml
    when the task tree is shipped whole (Pier ships it, tests/ one level below task.toml)."""
    thresholds = {name: dict(values) for name, values in DEFAULT_PERFORMANCE_THRESHOLDS.items()}
    task_toml = _task_toml_path()
    if task_toml is not None:
        try:
            config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            config = {}
        metadata = config.get("metadata", {})
        infb = metadata.get("inference_bench", {}) if isinstance(metadata, dict) else {}
        raw = infb.get("real_engine_thresholds", {}) if isinstance(infb, dict) else {}
        if isinstance(raw, dict):
            for benchmark_name, values in raw.items():
                if not isinstance(values, dict):
                    continue
                target = thresholds.setdefault(str(benchmark_name), {})
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        target[str(key)] = float(value)
    return thresholds


SCORED_BENCHMARK_KINDS = ("serve",)

MIN_ROUNDS_PER_ARM = 2
MATRIX_KEYS = ("speedup_per_workload", "worst_workload_speedup", "best_workload_speedup",
               "scored_workloads", "guard_workloads", "anchor_workload", "guard_ratios",
               "guards_ok", "guard_unmeasured", "unusable_workloads")


def resolve_scored_arms(
    tests_dir: Path,
    raw_dir: Path | None,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Both arms of the scored ratio, from ONE estimator or from NEITHER."""
    rounds = load_bench_rounds(raw_dir)
    if rounds is not None:
        return resolve_rounds_ratio(
            rounds, usable=usable_round_arms(rounds, MIN_ROUNDS_PER_ARM),
            lower_is_better=SCORED_LOWER_IS_BETTER,
        )

    baseline = load_baseline_tps(tests_dir, raw_dir)
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    candidate = _scored_value(summary)
    if baseline is None or candidate is None:
        return {
            "speedup": None, "paired_round_ratios": [],
            "baseline_tps": None, "candidate_tps": None, "estimator": None,
            "reason": "legacy_single_bench_incomplete",
            "pristine_arm_ok": baseline is not None,
            "candidate_arm_ok": candidate is not None,
        }
    return {
        "speedup": (baseline / candidate) if SCORED_LOWER_IS_BETTER else (candidate / baseline),
        "paired_round_ratios": [],
        "baseline_tps": baseline, "candidate_tps": candidate,
        "estimator": "single_bench_legacy", "reason": None,
        "pristine_arm_ok": True, "candidate_arm_ok": True,
    }


def load_bench_rounds(raw_dir: Path | None) -> dict[str, object] | None:
    """What `tests/_bench_rounds.py` aggregated this run, or None if it never ran."""
    if raw_dir is None:
        return None
    path = raw_dir.parent / "bench_rounds.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _round_arm_mean(rounds: dict[str, object] | None, arm: str) -> float | None:
    if not isinstance(rounds, dict):
        return None
    arms = rounds.get("arms")
    if not isinstance(arms, dict):
        return None
    entry = arms.get(arm)
    if not isinstance(entry, dict):
        return None
    mean = entry.get("mean")
    if not isinstance(mean, (int, float)) or isinstance(mean, bool) or not mean > 0:
        return None
    return float(mean) if math.isfinite(mean) else None




def check_performance(
    metrics: dict[str, object],
    *,
    thresholds_by_benchmark: dict[str, dict[str, float]] | None = None,
    baseline_tps: float | None = None,
    scored: dict[str, object] | None = None,
    bench_rounds: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = metrics.get("summary", {})
    if not isinstance(summary, dict):
        return {"benchmark": None, "passed": False, "reason": "missing summary"}
    benchmark = summary.get("benchmark", {})
    benchmark_name = benchmark.get("name") if isinstance(benchmark, dict) else None
    thresholds_by_benchmark = thresholds_by_benchmark or load_performance_thresholds()
    thresholds = thresholds_by_benchmark.get(str(benchmark_name))
    checks: dict[str, object] = {
        "benchmark": benchmark_name,
        "thresholds": thresholds or {},
        "passed": False,
    }
    if not thresholds:
        checks["reason"] = "unknown benchmark kind"
        return checks
    throughput = summary.get("throughput", {})
    latency_ms = summary.get("latency_ms", {})
    failures: list[str] = []
    if str(benchmark_name) not in SCORED_BENCHMARK_KINDS:
        failures.append("benchmark_kind_not_scored")
        checks["benchmark_kind_not_scored"] = str(benchmark_name)
    if "max_e2e_mean_ms" in thresholds:
        value = _latency_stat(latency_ms, "e2e", "mean")
        checks["e2e_mean_ms"] = value
        if value is None or value > thresholds["max_e2e_mean_ms"]:
            failures.append("e2e_mean_ms")
    if "max_ttft_mean_ms" in thresholds:
        value = _latency_stat(latency_ms, "ttft", "mean")
        checks["ttft_mean_ms"] = value
        if value is None or value > thresholds["max_ttft_mean_ms"]:
            failures.append("ttft_mean_ms")
    if "min_request_per_sec" in thresholds:
        value = _nested_number(throughput, "request_per_sec")
        checks["request_per_sec"] = value
        if value is None or value < thresholds["min_request_per_sec"]:
            failures.append("request_per_sec")
    if "min_total_tokens_per_sec" in thresholds:
        value = _nested_number(throughput, "total_tokens_per_sec")
        checks["total_tokens_per_sec"] = value
        if value is None or value < thresholds["min_total_tokens_per_sec"]:
            failures.append("total_tokens_per_sec")
    scored = scored or {}
    agent_tps = scored.get("candidate_tps")
    baseline_tps = scored.get("baseline_tps")
    estimator = scored.get("estimator")
    speedup = scored.get("speedup")
    if speedup is None and isinstance(agent_tps, (int, float)) and baseline_tps:
        speedup = (baseline_tps / agent_tps) if SCORED_LOWER_IS_BETTER else (agent_tps / baseline_tps)
    if speedup is not None and not (isinstance(speedup, (int, float)) and math.isfinite(speedup)):
        speedup = None
    checks["scored_metric"] = SCORED_METRIC_KEY
    checks["scored_lower_is_better"] = SCORED_LOWER_IS_BETTER
    checks["speedup_metric"] = "mean_ttft_ms (pristine/candidate; a lower first-token latency is faster)"
    checks["candidate_tps"] = agent_tps
    checks["speedup_vs_baseline"] = speedup
    checks["paired_round_ratios"] = scored.get("paired_round_ratios")
    checks["scored_arms_reason"] = scored.get("reason")
    checks["speedup_estimator"] = estimator
    checks["candidate_tps_scored"] = agent_tps
    checks["baseline_tps"] = baseline_tps
    checks["pristine_arm_ok"] = scored.get("pristine_arm_ok")
    for _key in MATRIX_KEYS:
        if _key in scored:
            checks[_key] = scored[_key]
    checks["candidate_arm_ok"] = scored.get("candidate_arm_ok")
    checks["noise_floor"] = NOISE_FLOOR
    if isinstance(bench_rounds, dict):
        checks["rounds"] = {
            key: bench_rounds.get(key)
            for key in (
                "ratio", "ratio_stdev", "arm_order", "warmup_runs", "warmup_plateau",
                "first_scored_run_vs_warmup_plateau", "elapsed_sec", "notes",
            )
        }
        arms_summary = bench_rounds.get("arms")
        if isinstance(arms_summary, dict):
            checks["rounds"]["arms"] = arms_summary
    if estimator is None or baseline_tps is None or agent_tps is None:
        checks["baseline_tps_missing"] = baseline_tps is None
        if not scored.get("pristine_arm_ok", baseline_tps is not None):
            failures.append("baseline_tps_unavailable")
        else:
            failures.append("candidate_tps_unavailable")
    elif speedup is None:
        failures.append("ratio_unavailable")
    elif speedup < NOISE_FLOOR:
        failures.append("speedup_vs_baseline")
    checks["above_noise_floor"] = speedup is not None and speedup >= NOISE_FLOOR
    if scored.get("guards_ok") is False:
        failures.append("guard_regressed")
    checks["failed_checks"] = failures
    checks["passed"] = not failures
    return checks


def _latency_stat(latency_ms: object, group: str, stat: str) -> float | int | None:
    if not isinstance(latency_ms, dict):
        return None
    values = latency_ms.get(group)
    if not isinstance(values, dict):
        return None
    return _nested_number(values, stat)


def _nested_number(values: object, key: str) -> float | int | None:
    """A finite number at `values[key]`, else None.

    THE FINITE CHECK IS A GATE, NOT HYGIENE: `NaN` compares False against everything, so a NaN
    throughput would clear every `min_*` floor at once with an empty `failed_checks`, and
    `json.loads` accepts a bare `NaN` token from an engine's own output. None routes it into the
    "could not be evaluated" path, which fails closed.
    """
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _reward_reason(
    *,
    reward: float,
    static_reward: float,
    upstream_ready: bool,
    performance_checks: dict[str, object],
    outputs_ok: bool = True,
    outputs_measured_ok: bool = True,
    reference_ok: bool = True,
    edited_ok: bool = True,
    diffable_ok: bool = True,
) -> str | None:
    """The verdict token for a zero, or None when the reward stands."""
    if reward > 0:
        return None
    failed = (
        performance_checks.get("failed_checks", [])
        if isinstance(performance_checks, dict)
        else []
    )
    if "baseline_tps_unavailable" in failed:
        return "pristine_failed"
    if not reference_ok:
        return "pristine_failed"
    if not diffable_ok:
        return "static_checks"
    if not edited_ok:
        return "no_patch"
    if "candidate_tps_unavailable" in failed:
        return "candidate_failed"
    if not upstream_ready:
        return "candidate_failed"
    if not outputs_measured_ok:
        return "correctness_failed"
    if not outputs_ok:
        return "correctness_failed"
    if "ratio_unavailable" in failed:
        return "candidate_failed"
    if "guard_regressed" in failed:
        return "guard_regressed"
    if "speedup_vs_baseline" in failed:
        return "no_speedup"
    return "candidate_failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--tests-dir", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    candidate_checks = check_candidate(workspace)
    static_reward, static_fraction = score(candidate_checks)
    repo = locate_candidate_repo(workspace)
    real_checks = check_real_engine_repo(repo)
    real_ready = all_real_checks_pass(real_checks)
    raw_dir = Path(args.raw_dir)
    upstream_checks = check_upstream_benchmark(raw_dir)
    upstream_ready = upstream_benchmark_ready(upstream_checks)
    repair_checks = check_regression_repaired(raw_dir)
    repaired = regression_repaired(repair_checks)
    output_checks = check_outputs_agree(raw_dir)
    outputs_ok = outputs_agree(output_checks)

    metrics = normalize_measured_or_unmeasured(raw_dir, Path(args.metrics))
    bench_rounds = load_bench_rounds(raw_dir)
    performance_checks = check_performance(
        metrics,
        thresholds_by_benchmark=load_performance_thresholds(),
        scored=resolve_scored_arms(Path(args.tests_dir), raw_dir, metrics),
        bench_rounds=bench_rounds,
    )
    performance_ready = upstream_ready and bool(performance_checks.get("passed"))
    ratio = performance_checks.get("speedup_vs_baseline")
    edited = candidate_checks.get("file_edited")
    candidate_measured = output_checks.get("candidate_measured") is True
    pristine_measured = output_checks.get("pristine_measured") is True

    stands = (
        real_ready and upstream_ready and performance_ready and outputs_ok
        and candidate_measured and pristine_measured and edited is True
    )
    reason = _reward_reason(
        reward=1.0 if stands else 0.0,
        static_reward=static_reward,
        upstream_ready=upstream_ready,
        performance_checks=performance_checks,
        outputs_ok=outputs_ok,
        outputs_measured_ok=candidate_measured,
        reference_ok=pristine_measured,
        edited_ok=edited is True,
        diffable_ok=edited is not None,
    )
    verdict = compute_regression_reward(
        matrix={key: performance_checks[key] for key in MATRIX_KEYS if key in performance_checks},
        reason=reason,
        ratio=ratio if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) else None,
        noise_floor=NOISE_FLOOR,
        correctness_ok=outputs_ok and candidate_measured and pristine_measured,
        oracle_issue={
            "note": "diagnostic only, never a vote: did the gain come from the planted cause",
            "candidate_triton_cubins": repair_checks.get("candidate_triton_cubins"),
            "pristine_triton_cubins": repair_checks.get("pristine_triton_cubins"),
            "cubin_fraction": repair_checks.get("cubin_fraction"),
            "cubin_fraction_at_most_for_repair": REPAIR_MAX_CANDIDATE_CUBIN_FRACTION,
            "repaired": repaired,
        },
        scored_metric=SCORED_METRIC_KEY,
        higher_is_better=not SCORED_LOWER_IS_BETTER,
    )
    metrics["candidate_checks"] = candidate_checks
    metrics["static_check_fraction"] = static_fraction
    metrics["real_engine_checks"] = real_checks
    metrics["upstream_benchmark_checks"] = upstream_checks
    metrics["upstream_benchmark_ready"] = upstream_ready
    metrics["regression_repair_checks"] = repair_checks
    metrics["regression_repaired"] = repaired
    metrics["correctness_checks"] = output_checks
    metrics["outputs_agree"] = outputs_ok
    metrics["bench_rounds"] = bench_rounds
    metrics["performance_checks"] = performance_checks
    metrics["performance_ready"] = performance_ready
    metrics["score_profile"] = SCORING_CONTRACT
    metrics["real_engine_ready"] = real_ready and upstream_ready and performance_ready
    metrics.update(verdict["metrics"])
    metrics["reason"] = reason
    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rewards = dict(verdict["rewards"])
    rewards["real_engine_ready"] = 1.0 if (real_ready and upstream_ready and performance_ready) else 0.0
    rewards["upstream_benchmark_ready"] = 1.0 if upstream_ready else 0.0
    rewards["regression_repaired"] = 1.0 if repaired else 0.0
    rewards["outputs_agree"] = 1.0 if outputs_ok else 0.0
    Path(args.reward).write_text(
        json.dumps(rewards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
