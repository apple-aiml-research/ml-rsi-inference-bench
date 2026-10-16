#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for regression_mtp_decode_throughput."""

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
from _verifier_lib.bench_stats import usable_round_arms  # noqa: E402
from _verifier_lib.metrics import normalize_engine_metrics  # noqa: E402
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
    "vllm/entrypoints/cli/benchmark/serve.py",
    "vllm/model_executor/models/registry.py",
    "vllm/model_executor/layers/fla/ops/utils.py",
)

DEFAULT_PERFORMANCE_THRESHOLDS = {
    "latency": {
        "max_e2e_mean_ms": 120_000.0,
    },
    "throughput": {
        "min_request_per_sec": 0.1,
        "min_total_tokens_per_sec": 1.0,
    },
    "serve": {
        "min_request_per_sec": 0.1,
        "min_total_tokens_per_sec": 1.0,
        "max_ttft_mean_ms": 60_000.0,
    },
}

NOISE_FLOOR = 1.03
SCORED_THROUGHPUT_KEY = "total_tokens_per_sec"
SCORED_LOWER_IS_BETTER = False

MIN_OUTPUT_TOKEN_AGREEMENT = 1.0


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
    required = checks.get("required_paths", {})
    return (
        bool(checks.get("repo_exists"))
        and bool(checks.get("pyproject_declares_vllm"))
        and isinstance(required, dict)
        and all(bool(value) for value in required.values())
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
        except json.JSONDecodeError as exc:
            checks["manifest_json_object"] = False
            checks["manifest_error"] = str(exc)
    return checks


def check_output_equivalence(raw_dir: Path) -> dict[str, object]:
    """THE CORRECTNESS GATE. Did the candidate change what the engine says?"""
    out: dict[str, object] = {}
    artifacts = raw_dir.parent
    ref_ids, ref_err = _arm_generations(artifacts / "baseline" / "raw" / "equivalence.json")
    again_ids, again_err = _arm_generations(artifacts / REFERENCE_AGAIN_REL)
    cand_ids, cand_err = _arm_generations(raw_dir / "equivalence.json")
    out["pristine_measured"] = ref_ids is not None
    out["reference_again_measured"] = again_ids is not None
    out["candidate_measured"] = cand_ids is not None
    out["outputs_compared"] = False
    for side, err in (("pristine", ref_err), ("reference_again", again_err), ("candidate", cand_err)):
        if err:
            out[f"{side}_equivalence_error"] = err

    reproduced = False
    if ref_ids is not None and again_ids is not None:
        same = sum(1 for a, b in zip(ref_ids, again_ids) if a == b)
        total = max(len(ref_ids), len(again_ids))
        reproduced = len(ref_ids) == len(again_ids) and same == total
        out["reference_self_agreement"] = f"{same}/{total}"
    out["reference_reproduced"] = reproduced

    agreement: float | None = None
    probe = raw_dir / "equivalence_compare.json"
    payload: object = None
    if probe.is_file():
        try:
            payload = json.loads(probe.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            out["output_equivalence_error"] = f"{type(exc).__name__}: {exc}"
    out["output_equivalence_recorded"] = isinstance(payload, dict)
    if isinstance(payload, dict):
        out["output_probe_rc"] = payload.get("rc")
        for key in ("sequences_compared", "exact_sequences", "exact_fraction",
                    "token_agreement", "prefix_agreement", "verdict"):
            if key in payload:
                out[f"output_{key}"] = payload[key]
        value = payload.get("token_agreement")
        if payload.get("rc") == 0 and isinstance(value, (int, float)) and not isinstance(value, bool):
            agreement = float(value)
            out["outputs_compared"] = True
            out["candidate_measured"] = True
            out["pristine_measured"] = True
            out["min_output_token_agreement"] = MIN_OUTPUT_TOKEN_AGREEMENT

    if reproduced and agreement is not None:
        unchanged = agreement >= MIN_OUTPUT_TOKEN_AGREEMENT
        out["outputs_unchanged"] = unchanged
        out["equivalence_verdict"] = "match" if unchanged else "mismatch"
        return out
    out["outputs_unchanged"] = None
    out["equivalence_verdict"] = "unavailable"
    if ref_ids is None or again_ids is None:
        out["equivalence_reason"] = (
            "the pristine reference was not decoded twice, so whether it reproduces itself -- and "
            "with it anything about the candidate -- is unknown; the harness's zero"
        )
    elif not reproduced:
        out["equivalence_reason"] = (
            f"the reference did not reproduce ITSELF ({out['reference_self_agreement']} sequences) "
            f"across two decodes of the same pristine tree in this trial; nothing can be concluded "
            f"about the candidate from a reference that moves -- an environment fault, a re-fly"
        )
    elif not out["candidate_measured"]:
        out["equivalence_reason"] = "the candidate produced no generations to compare"
    else:
        out["equivalence_reason"] = "both sides answered and the comparison did not run"
    return out


REFERENCE_AGAIN_REL = Path("equivalence") / "reference_again" / "equivalence.json"


def _arm_generations(path: Path) -> tuple[list | None, str | None]:
    """(this decode's token ids, None) if `_equivalence_probe.py` recorded any, else (None, why)."""
    if not path.is_file():
        return None, f"no {path.name}"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(record, dict):
        return None, "record is not a JSON object"
    token_ids = record.get("token_ids")
    if record.get("ok") is True and isinstance(token_ids, list) and token_ids:
        return token_ids, None
    return None, str(record.get("error") or "no generations recorded")


def _equivalence_sides(checks: dict[str, object]) -> tuple[bool, bool]:
    """(reference_ok, candidate_measured) off check_output_equivalence's record."""
    verdict = checks.get("equivalence_verdict")
    if verdict in ("match", "mismatch"):
        return True, True
    if verdict != "unavailable":
        return False, False
    candidate_measured = checks.get("candidate_measured") is True
    reference_ok = checks.get("reference_reproduced") is True and not candidate_measured
    return reference_ok, candidate_measured


def check_defect_path_exercised(raw_dir: Path) -> dict[str, object]:
    """Did the BASELINE arm actually recompile the GDN kernel on this workload?"""
    probe = raw_dir / "gdn_jit_probe.json"
    checks: dict[str, object] = {"probe_exists": probe.is_file()}
    if not checks["probe_exists"]:
        return checks
    try:
        payload = json.loads(probe.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks["probe_json_object"] = False
        checks["probe_error"] = str(exc)
        return checks
    checks["probe_json_object"] = True
    for key in ("rc", "verdict", "baseline_gdn_compiles", "candidate_gdn_compiles",
                "gdn_compile_reduction"):
        checks[key] = payload.get(key)
    rc = payload.get("rc")
    checks["defect_path_exercised"] = rc in (0, 5)
    checks["defect_actually_fixed"] = rc == 0
    return checks


def upstream_benchmark_ready(checks: dict[str, object]) -> bool:
    """WORKLOAD INTEGRITY: the candidate's bench ran to completion and the workload was the declared one."""
    return (
        bool(checks.get("raw_exists"))
        and bool(checks.get("manifest_exists"))
        and bool(checks.get("raw_json_object"))
        and bool(checks.get("manifest_json_object"))
        and checks.get("manifest_returncode") == 0
        and bool(checks.get("manifest_mentions_vllm_bench"))
        and checks.get("defect_path_exercised") is not False
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
    return normalize_sample(tests_dir, raw_dir, metrics_path)


def load_baseline_tps(tests_dir: Path, raw_dir: Path | None = None) -> float | None:
    """The regressed-state baseline on the scored key, for the LEGACY single-bench estimator only."""
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
                        SCORED_THROUGHPUT_KEY,
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
    """The LIVENESS floors: the baked table, refined by task.toml when the tree is shipped whole.

    No relative gate lives here any more -- under the continuous contract the only relative number
    the verifier holds is NOISE_FLOOR, a module constant, and it is not a gate. (The operator
    override INFB_REGRESSION_MIN_SPEEDUP went with the gate it overrode.)
    """
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

SAMPLE_RAW = "sample_vllm_bench.json"


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


def _attribute_the_short_arm(resolved: dict[str, object], rounds: dict[str, object]) -> None:
    """When the aggregate cannot be scored, name the arm that came up short by its own round count."""
    arms = rounds.get("arms")
    if not isinstance(arms, dict):
        return

    def n_ok(arm: str) -> int:
        entry = arms.get(arm)
        count = entry.get("n_ok") if isinstance(entry, dict) else None
        return count if isinstance(count, int) and not isinstance(count, bool) else 0

    pristine, candidate = n_ok("pristine"), n_ok("candidate")
    resolved["pristine_rounds_ok"] = pristine
    resolved["candidate_rounds_ok"] = candidate
    if candidate < pristine:
        resolved["candidate_arm_ok"] = False
    elif pristine < candidate:
        resolved["pristine_arm_ok"] = False


def resolve_scored_arms(
    tests_dir: Path,
    raw_dir: Path | None,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Both arms of the scored ratio, from ONE estimator or from NEITHER."""
    rounds = load_bench_rounds(raw_dir)
    if rounds is not None:
        resolved = resolve_rounds_ratio(
            rounds, usable=usable_round_arms(rounds, MIN_ROUNDS_PER_ARM),
            lower_is_better=SCORED_LOWER_IS_BETTER,
        )
        if resolved.get("estimator") is None:
            _attribute_the_short_arm(resolved, rounds)
        return resolved

    baseline = load_baseline_tps(tests_dir, raw_dir)
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    throughput = summary.get("throughput", {}) if isinstance(summary, dict) else {}
    candidate = _nested_number(throughput, SCORED_THROUGHPUT_KEY)
    source = str(metrics.get("source_path") or "") if isinstance(metrics, dict) else ""
    if source.endswith(SAMPLE_RAW):
        candidate = None
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
    """The liveness floors of the guessed benchmark kind, then the scored ratio and the noise floor."""
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
    checks["speedup_paired_geomean"] = arms.get("speedup_paired_geomean")
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
    real_checks = check_real_vllm_repo(repo)
    real_ready = all_real_checks_pass(real_checks)
    raw_dir = Path(args.raw_dir)
    upstream_checks = check_upstream_benchmark(raw_dir)
    upstream_checks.update(check_defect_path_exercised(raw_dir))
    upstream_checks.update(check_output_equivalence(raw_dir))
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
    performance_ready = (
        upstream_ready
        and bool(performance_checks.get("passed"))
        and upstream_checks.get("outputs_unchanged") is True
    )
    ratio = performance_checks.get("speedup_vs_baseline")
    edited = candidate_checks.get("file_edited")
    outputs_ok = upstream_checks.get("outputs_unchanged") is True
    reference_ok, candidate_measured = _equivalence_sides(upstream_checks)

    stands = (
        real_ready and upstream_ready and performance_ready and outputs_ok
        and candidate_measured and reference_ok and edited is True
    )

    reason = _reward_reason(
        reward=1.0 if stands else 0.0,
        static_reward=static_reward,
        upstream_ready=upstream_ready,
        performance_checks=performance_checks,
        outputs_ok=outputs_ok,
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
        correctness_ok=outputs_ok and candidate_measured and reference_ok,
        oracle_issue={
            "note": "diagnostic only, never a vote: did the gain come from the planted cause",
            "anchor_file_edited": candidate_checks.get("anchor_file_edited"),
            "defect_actually_fixed": upstream_checks.get("defect_actually_fixed"),
            "gdn_probe_rc": upstream_checks.get("rc"),
            "baseline_gdn_compiles": upstream_checks.get("baseline_gdn_compiles"),
            "candidate_gdn_compiles": upstream_checks.get("candidate_gdn_compiles"),
            "gdn_compile_reduction": upstream_checks.get("gdn_compile_reduction"),
            "gdn_probe_verdict": upstream_checks.get("verdict"),
            "required_terms": candidate_checks.get("required_terms"),
        },
        scored_metric="throughput." + SCORED_THROUGHPUT_KEY,
        higher_is_better=not SCORED_LOWER_IS_BETTER,
    )
    metrics["candidate_checks"] = candidate_checks
    metrics["static_check_fraction"] = static_fraction
    metrics["real_engine_checks"] = real_checks
    metrics["upstream_benchmark_checks"] = upstream_checks
    metrics["upstream_benchmark_ready"] = upstream_ready
    metrics["bench_rounds"] = bench_rounds
    metrics["scored_arms"] = scored
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
    Path(args.reward).write_text(
        json.dumps(rewards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
