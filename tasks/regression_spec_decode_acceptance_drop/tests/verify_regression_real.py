#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for regression_spec_decode_acceptance_drop."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib.metrics import normalize_engine_metrics  # noqa: E402
from _verifier_lib.bench_stats import usable_round_arms  # noqa: E402
from _verifier_lib.regression_scoring import (  # noqa: E402
    CONTRACT as SCORING_CONTRACT,
    compute_regression_reward,
    resolve_rounds_ratio,
)

from verify_regression import (
    check_candidate,
    normalize_sample,
    score,
)


REQUIRED_VLLM_PATHS = (
    "pyproject.toml",
    "vllm",
    "benchmarks/benchmark_throughput.py",
    "vllm/model_executor/models/registry.py",
    "vllm/entrypoints/openai/api_server.py",
    "vllm/v1/engine/core.py",
)

DEFAULT_PERFORMANCE_THRESHOLDS = {
    "latency": {
        "max_e2e_mean_ms": 600_000.0,
    },
    "throughput": {
        "min_request_per_sec": 0.05,
        "min_total_tokens_per_sec": 1.0,
    },
    "serve": {
        "min_request_per_sec": 0.05,
        "min_total_tokens_per_sec": 1.0,
        "max_ttft_mean_ms": 30_000.0,
    },
}

NOISE_FLOOR = 1.03
SCORED_THROUGHPUT_KEY = "total_tokens_per_sec"
SCORED_LOWER_IS_BETTER = False


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


def check_real_vllm_repo(repo: Path) -> dict[str, object]:
    checks: dict[str, object] = {
        "repo_path": str(repo),
        "repo_exists": repo.exists(),
        "git_commit": repo_commit(repo),
        "required_paths": {},
    }
    required = {}
    for rel in REQUIRED_VLLM_PATHS:
        required[rel] = (repo / rel).exists()
    checks["required_paths"] = required
    pyproject = repo / "pyproject.toml"
    declares = pyproject.exists() and 'name = "vllm"' in pyproject.read_text(
        encoding="utf-8",
        errors="replace",
    )
    if not declares:
        setup_py = repo / "setup.py"
        if setup_py.exists():
            declares = "vllm" in setup_py.read_text(encoding="utf-8", errors="replace")
    checks["pyproject_declares_vllm"] = declares
    return checks


def all_real_checks_pass(checks: dict[str, object]) -> bool:
    return (
        bool(checks.get("repo_exists"))
    )


def check_upstream_benchmark(raw_dir: Path) -> dict[str, object]:
    raw_latency = raw_dir / "vllm_bench_latency.json"
    raw_throughput = raw_dir / "vllm_bench_throughput.json"
    raw_serve = raw_dir / "vllm_bench_serve.json"
    manifest = raw_dir / "vllm_bench_manifest.json"
    selected = next(
        (path for path in (raw_latency, raw_throughput, raw_serve) if path.exists()),
        raw_latency,
    )
    checks: dict[str, object] = {
        "raw_latency_path": str(raw_latency),
        "raw_throughput_path": str(raw_throughput),
        "raw_serve_path": str(raw_serve),
        "manifest_path": str(manifest),
        "raw_exists": selected.exists(),
        "manifest_exists": manifest.exists(),
    }
    if selected.exists():
        checks["raw_path"] = str(selected)
        try:
            raw = json.loads(selected.read_text(encoding="utf-8"))
            checks["raw_json_object"] = isinstance(raw, dict)
            checks["raw_keys"] = sorted(raw) if isinstance(raw, dict) else []
        except json.JSONDecodeError as exc:
            checks["raw_json_object"] = False
            checks["raw_error"] = str(exc)
    if manifest.exists():
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            checks["manifest_json_object"] = isinstance(manifest_data, dict)
            if isinstance(manifest_data, dict):
                checks["manifest_returncode"] = manifest_data.get("returncode")
                command = manifest_data.get("command")
                checks["manifest_command"] = command
                checks["manifest_mentions_vllm_bench"] = (
                    isinstance(command, str)
                    and "vllm" in command
                    and "bench" in command
                )
                probe_rc = manifest_data.get("async_path_probe_rc")
                checks["manifest_probe_rc"] = probe_rc
                checks["manifest_probe_ok"] = probe_rc == 0
        except json.JSONDecodeError as exc:
            checks["manifest_json_object"] = False
            checks["manifest_error"] = str(exc)
    return checks


def upstream_benchmark_ready(checks: dict[str, object]) -> bool:
    return (
        bool(checks.get("raw_exists"))
        and checks.get("manifest_returncode") == 0
        and bool(checks.get("manifest_probe_ok"))
    )


def normalize_real_or_sample(
    tests_dir: Path, raw_dir: Path, metrics_path: Path
) -> dict[str, object]:
    raw_latency = raw_dir / "vllm_bench_latency.json"
    raw_throughput = raw_dir / "vllm_bench_throughput.json"
    raw_serve = raw_dir / "vllm_bench_serve.json"
    if raw_latency.exists():
        return normalize_engine_metrics(raw_latency, metrics_path, metric_source="vllm")
    if raw_throughput.exists():
        return normalize_engine_metrics(raw_throughput, metrics_path, metric_source="vllm")
    if raw_serve.exists():
        return normalize_engine_metrics(raw_serve, metrics_path, metric_source="vllm")
    raise SystemExit(
        "no engine bench output in "
        f"{raw_dir}: looked for vllm_bench_latency.json, vllm_bench_throughput.json and "
        "vllm_bench_serve.json. The bench did not run to completion -- read "
        "raw/*_runner.log and raw/*.server.log in the artifacts. This task does not fall back to a "
        "canned sample: it declares requires_real_engine, so a sample would be a fabricated score."
    )


def load_baseline_tps(tests_dir: Path, raw_dir: Path | None = None) -> float | None:
    """Load the regressed-state baseline TPS."""
    if raw_dir is not None:
        baseline_raw = raw_dir.parent / "baseline" / "raw"
        for name in ("vllm_bench_throughput.json", "vllm_bench_latency.json", "vllm_bench_serve.json"):
            bench_file = baseline_raw / name
            if bench_file.exists():
                try:
                    tmp_metrics = normalize_engine_metrics(
                        bench_file,
                        baseline_raw / "_baseline_metrics.json",
                        metric_source="vllm",
                    )
                    tps = _nested_number(
                        tmp_metrics.get("summary", {}).get("throughput", {}),
                        "total_tokens_per_sec",
                    )
                    if isinstance(tps, (int, float)) and tps > 0:
                        return float(tps)
                except Exception:
                    pass
    baseline_file = tests_dir / "baseline_tps.json"
    if not baseline_file.exists():
        return None
    try:
        data = json.loads(baseline_file.read_text(encoding="utf-8"))
        val = data.get("total_tokens_per_sec")
        return float(val) if isinstance(val, (int, float)) and val > 0 else None
    except (json.JSONDecodeError, ValueError, OSError):
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
    """The LIVENESS floors: the baked table, refined by task.toml when the tree is shipped whole."""
    thresholds = {name: dict(values) for name, values in DEFAULT_PERFORMANCE_THRESHOLDS.items()}
    task_toml = _task_toml_path()
    if task_toml is None:
        return thresholds
    try:
        config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return thresholds
    metadata = config.get("metadata", {})
    infb = metadata.get("inference_bench", {}) if isinstance(metadata, dict) else {}
    raw = infb.get("real_engine_thresholds", {}) if isinstance(infb, dict) else {}
    if not isinstance(raw, dict):
        return thresholds
    for benchmark_name, values in raw.items():
        if not isinstance(values, dict):
            continue
        target = thresholds.setdefault(str(benchmark_name), {})
        for key, value in values.items():
            if isinstance(value, (int, float)):
                target[str(key)] = float(value)
    return thresholds


MIN_ROUNDS_PER_ARM = 2
MATRIX_KEYS = ("speedup_per_workload", "worst_workload_speedup", "best_workload_speedup",
               "scored_workloads", "guard_workloads", "anchor_workload", "guard_ratios",
               "guards_ok", "guard_unmeasured", "unusable_workloads")


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
    throughput = summary.get("throughput", {}) if isinstance(summary, dict) else {}
    candidate = _nested_number(throughput, SCORED_THROUGHPUT_KEY)
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


def check_performance(
    metrics: dict[str, object],
    *,
    thresholds_by_benchmark: dict[str, dict[str, float]] | None = None,
    baseline_tps: float | None = None,
    scored: dict[str, object] | None = None,
    bench_rounds: dict[str, object] | None = None,
) -> dict[str, object]:
    """`bench_rounds` is accepted for the shared call shape used across the family."""
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
    arms = scored if isinstance(scored, dict) else {}
    baseline_tps = arms.get("baseline_tps")
    agent_tps = arms.get("candidate_tps")
    estimator = arms.get("estimator")
    speedup = arms.get("speedup")
    if speedup is None and isinstance(agent_tps, (int, float)) and baseline_tps:
        speedup = (baseline_tps / agent_tps) if SCORED_LOWER_IS_BETTER else (agent_tps / baseline_tps)
    if speedup is not None and not (isinstance(speedup, (int, float)) and math.isfinite(speedup)):
        speedup = None
    checks["scored_metric"] = SCORED_THROUGHPUT_KEY
    checks["scored_lower_is_better"] = SCORED_LOWER_IS_BETTER
    checks["scored_estimator"] = estimator
    checks["speedup_estimator"] = estimator
    checks["paired_round_ratios"] = arms.get("paired_round_ratios")
    checks["scored_arms_reason"] = arms.get("reason")
    checks["pristine_arm_ok"] = arms.get("pristine_arm_ok")
    for _key in MATRIX_KEYS:
        if _key in arms:
            checks[_key] = arms[_key]
    checks["candidate_arm_ok"] = arms.get("candidate_arm_ok")
    checks["speedup_vs_baseline"] = speedup
    checks["baseline_tps"] = baseline_tps
    checks["candidate_tps"] = agent_tps
    checks["noise_floor"] = NOISE_FLOOR
    if isinstance(bench_rounds, dict):
        checks["rounds"] = bench_rounds
    if estimator is None or baseline_tps is None or agent_tps is None:
        checks["baseline_tps_missing"] = baseline_tps is None
        if arms.get("pristine_arm_ok") is True and arms.get("candidate_arm_ok") is False:
            failures.append("candidate_tps_unavailable")
        else:
            failures.append("baseline_tps_unavailable")
    elif speedup is None:
        failures.append("ratio_unavailable")
    elif speedup < NOISE_FLOOR:
        failures.append("speedup_vs_baseline")
    checks["above_noise_floor"] = speedup is not None and speedup >= NOISE_FLOOR
    if arms.get("guards_ok") is False:
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
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    return value if isinstance(value, (int, float)) else None


def load_output_equivalence(raw_dir: Path | None) -> dict[str, object]:
    """Did the candidate still answer what the target model answers?"""
    record = load_bench_rounds(raw_dir)
    block = (record or {}).get("correctness")
    if not isinstance(block, dict):
        return {"verdict": "unavailable", "passed": False,
                "reason": "bench_rounds.json carries no correctness phase"}
    out = dict(block)
    out["passed"] = block.get("verdict") == "match"
    return out


def _equivalence_sides(equivalence: dict[str, object]) -> tuple[bool, bool]:
    """(reference_ok, candidate_measured) off the correctness phase's record."""
    verdict = equivalence.get("verdict") if isinstance(equivalence, dict) else None
    if verdict in ("match", "mismatch"):
        return True, True
    if verdict != "unavailable":
        return False, False
    ref = equivalence.get("reference_digests")
    ref2 = equivalence.get("reference_again_digests")
    cand = equivalence.get("candidate_digests")
    reference_ok = (isinstance(ref, list) and isinstance(ref2, list) and bool(ref)
                    and len(ref) == len(ref2) and ref == ref2)
    candidate_measured = isinstance(cand, list) and bool(cand) and (
        not reference_ok or len(cand) == len(ref))
    return reference_ok, candidate_measured


def _acceptance(raw_dir: Path) -> float | None:
    """`mean_acceptance_length` off one arm's raw bench record, or None."""
    path = raw_dir / "vllm_bench_throughput.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("mean_acceptance_length")
    except (OSError, json.JSONDecodeError):
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


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
    static_reward, engine_tree_edited = score(candidate_checks)
    repo = locate_candidate_repo(workspace)
    real_checks = check_real_vllm_repo(repo)
    real_ready = all_real_checks_pass(real_checks)
    raw_dir = Path(args.raw_dir)
    upstream_checks = check_upstream_benchmark(raw_dir)
    upstream_ready = upstream_benchmark_ready(upstream_checks)

    metrics = normalize_real_or_sample(Path(args.tests_dir), raw_dir, Path(args.metrics))
    scored = resolve_scored_arms(Path(args.tests_dir), raw_dir, metrics)
    bench_rounds = load_bench_rounds(raw_dir)
    performance_checks = check_performance(
        metrics,
        thresholds_by_benchmark=load_performance_thresholds(),
        scored=scored,
        bench_rounds=bench_rounds,
    )
    performance_ready = bool(performance_checks.get("passed"))
    equivalence = load_output_equivalence(raw_dir)
    equivalence_ready = bool(equivalence.get("passed"))
    reference_ok, candidate_measured = _equivalence_sides(equivalence)
    ratio = performance_checks.get("speedup_vs_baseline")
    edited = candidate_checks.get("file_edited")

    stands = (
        real_ready and upstream_ready and performance_ready and equivalence_ready
        and reference_ok and candidate_measured and edited is True
    )
    reason = _reward_reason(
        reward=1.0 if stands else 0.0,
        static_reward=static_reward,
        upstream_ready=upstream_ready,
        performance_checks=performance_checks,
        outputs_ok=equivalence_ready,
        outputs_measured_ok=candidate_measured,
        reference_ok=reference_ok,
        edited_ok=edited is True,
        diffable_ok=edited is not None,
    )
    verdict = compute_regression_reward(
        matrix={key: performance_checks[key] for key in MATRIX_KEYS if key in performance_checks},
        reason=reason,
        ratio=ratio if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) else None,
        noise_floor=NOISE_FLOOR,
        correctness_ok=equivalence_ready and reference_ok and candidate_measured,
        oracle_issue={
            "note": "diagnostic only, never a vote: did the gain come from the planted cause",
            "candidate_mean_acceptance_length": _acceptance(raw_dir),
            "pristine_mean_acceptance_length": _acceptance(raw_dir.parent / "baseline" / "raw"),
            "anchor_file_edited": candidate_checks.get("anchor_file_edited"),
            "required_terms": candidate_checks.get("required_terms"),
        },
        scored_metric="throughput." + SCORED_THROUGHPUT_KEY,
        higher_is_better=not SCORED_LOWER_IS_BETTER,
    )
    metrics["candidate_checks"] = candidate_checks
    metrics["engine_tree_edited"] = engine_tree_edited
    metrics["real_engine_checks"] = real_checks
    metrics["upstream_benchmark_checks"] = upstream_checks
    metrics["upstream_benchmark_ready"] = upstream_ready
    metrics["performance_checks"] = performance_checks
    metrics["scored_arms"] = scored
    metrics["performance_ready"] = performance_ready
    metrics["output_equivalence"] = equivalence
    metrics["output_equivalence_ready"] = equivalence_ready
    metrics["score_profile"] = SCORING_CONTRACT
    metrics["real_engine_ready"] = equivalence_ready and performance_ready
    metrics.update(verdict["metrics"])
    metrics["reason"] = reason
    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rewards = dict(verdict["rewards"])
    rewards["output_equivalence"] = 1.0 if equivalence_ready else 0.0
    Path(args.reward).write_text(
        json.dumps(rewards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
