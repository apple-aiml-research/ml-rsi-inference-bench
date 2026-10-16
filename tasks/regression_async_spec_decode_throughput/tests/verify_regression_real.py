#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for regression_async_spec_decode_throughput."""

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
    "vllm/v1/core/sched/scheduler.py",
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
        "max_ttft_mean_ms": 5_000.0,
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
                probe_rc = manifest_data.get("async_path_probe_rc")
                if probe_rc is None:
                    probe_rc = manifest_data.get("defect_path_probe_rc")
                checks["manifest_probe_rc"] = probe_rc
                checks["manifest_probe_ok"] = probe_rc == 0
        except json.JSONDecodeError as exc:
            checks["manifest_json_object"] = False
            checks["manifest_error"] = str(exc)
    return checks


MECHANISM_MAX_CANDIDATE_RATIO = 0.25


def _probe_non_uniform(rounds_dir: Path, arm: str) -> tuple[int | None, int | None]:
    """Worst-case ragged-step count for one arm, over every round that recorded a probe."""
    worst_nu: int | None = None
    steps: int | None = None
    for f in sorted(rounds_dir.glob(f"*/{arm}/raw/async_path_probe.json")):
        try:
            pr = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        nu = pr.get("non_uniform_decode_steps")
        if not isinstance(nu, int):
            continue
        if worst_nu is None or nu > worst_nu:
            worst_nu = nu
            ds = pr.get("decode_steps")
            steps = ds if isinstance(ds, int) else None
    return worst_nu, steps


def check_mechanism(raw_dir: Path) -> dict[str, object]:
    """Was the rectangular-decode invariant restored? RECORDED, NEVER SCORED."""
    rounds_dir = raw_dir.parent / "rounds"
    cand_nu, cand_steps = _probe_non_uniform(rounds_dir, "candidate")
    pris_nu, pris_steps = _probe_non_uniform(rounds_dir, "pristine")
    checks: dict[str, object] = {
        "rounds_dir": str(rounds_dir),
        "candidate_non_uniform": cand_nu,
        "candidate_decode_steps": cand_steps,
        "pristine_non_uniform": pris_nu,
        "pristine_decode_steps": pris_steps,
        "max_candidate_ratio": MECHANISM_MAX_CANDIDATE_RATIO,
    }
    if cand_nu is None or pris_nu is None:
        checks["passed"] = False
        checks["reason"] = (
            "no per-arm probe found under rounds/*/{candidate,pristine}/raw/async_path_probe.json; "
            "the mechanism was not observed on this run"
        )
        return checks
    if pris_nu <= 0:
        checks["passed"] = False
        checks["reason"] = (
            f"the pristine arm shows the defect firing on {pris_nu} decode steps, so there is "
            "nothing for the candidate to have repaired; the run did not exercise the defect"
        )
        return checks
    ratio = cand_nu / pris_nu
    checks["ratio"] = ratio
    checks["passed"] = ratio <= MECHANISM_MAX_CANDIDATE_RATIO
    if not checks["passed"]:
        checks["reason"] = (
            f"the defect still fires on the candidate arm: {cand_nu}/{cand_steps} decode steps "
            f"non-uniform against the pristine arm's {pris_nu}/{pris_steps} (ratio {ratio:.2f}, "
            f"limit {MECHANISM_MAX_CANDIDATE_RATIO}). The throughput improved, but not by repairing "
            "this regression"
        )
    return checks


def check_tree_stability(raw_dir: Path, repo: Path) -> dict[str, object]:
    """Was the tree that got benched the tree that was captured? THIS GATES."""
    checks: dict[str, object] = {"passed": None}
    stamp = raw_dir / "tree_head_at_verify_start.txt"
    if not stamp.is_file():
        checks["reason"] = "no tree stamp was written before benching; nothing is asserted"
        return checks
    before = stamp.read_text(errors="replace").strip()
    checks["head_at_verify_start"] = before
    if not before or before == "unavailable":
        checks["reason"] = "the tree stamp is empty or unavailable; nothing is asserted"
        return checks
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=60,
        )
        after = proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        after = ""
    checks["head_at_scoring"] = after or None
    if not after:
        checks["reason"] = "could not read HEAD at scoring time; nothing is asserted"
        return checks
    checks["passed"] = before == after
    if not checks["passed"]:
        checks["reason"] = (
            f"the tree changed during verification: HEAD was {before[:12]} when it started and "
            f"{after[:12]} when it finished, so the engine that was scored is not the engine that "
            f"was captured. The ratio may still be internally consistent -- that cannot be told "
            f"apart from the case where the change landed between scored rounds and the mean is of "
            f"two different engines. Void on that ground; this says nothing about the candidate"
        )
    return checks


def check_configuration_conformance(raw_dir: Path) -> dict[str, object]:
    """Is the candidate running the engine configuration the harness pinned? THIS GATES."""
    checks: dict[str, object] = {"passed": None, "fields": {}}
    rounds = raw_dir.parent / "rounds"

    def read(arm: str) -> dict:
        for f in sorted(rounds.glob(f"*/{arm}/raw/async_path_probe.json")):
            try:
                d = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(d, dict):
                return d
        return {}

    cand, pris = read("candidate"), read("pristine")
    FIELDS = ("mxfp4_backend", "attention_backend")

    def usable(v: object) -> bool:
        return isinstance(v, str) and v and "unavailable" not in v

    mismatches = []
    observed = 0
    for key in FIELDS:
        cv, pv = cand.get(key), pris.get(key)
        checks["fields"][key] = {"candidate": cv, "pristine": pv}
        if not usable(pv):
            continue
        observed += 1
        if cv != pv:
            mismatches.append(f"{key}: candidate {cv!r} against the pinned {pv!r}")
    if observed == 0:
        checks["reason"] = (
            "the pristine arm recorded no usable backend identity, so the recorder is not working "
            "in this image; nothing is asserted about the candidate"
        )
        return checks
    checks["fields_compared"] = observed
    checks["passed"] = not mismatches
    if mismatches:
        checks["reason"] = (
            "the candidate is not running the engine configuration the harness pinned: "
            + "; ".join(mismatches)
        )
    return checks


NUMERIC_EQUIVALENCE_MAX_ABS_DELTA = 1e-3


def check_numeric_equivalence(raw_dir: Path) -> dict[str, object]:
    """Do the two arms compute the same numbers? THIS GATES -- it is the "and it is correct" half."""
    checks: dict[str, object] = {"passed": None, "max_abs_delta_limit": NUMERIC_EQUIVALENCE_MAX_ABS_DELTA}
    arms: dict[str, dict] = {}
    for arm in ("candidate", "pristine"):
        for f in sorted((raw_dir.parent / "rounds").glob(f"*/{arm}/raw/async_path_probe.json")):
            try:
                fp = json.loads(f.read_text()).get("numeric_fingerprint")
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(fp, dict) and fp.get("available"):
                arms[arm] = fp
                break
    checks["arms_available"] = sorted(arms)
    checks["pristine_measured"] = "pristine" in arms
    checks["candidate_measured"] = "candidate" in arms

    if "pristine" not in arms:
        checks["reason"] = (
            "the pristine arm recorded no numeric fingerprint, so the recorder is not working in "
            "this image; nothing is asserted about the candidate"
        )
        return checks
    if "candidate" not in arms:
        checks["passed"] = False
        checks["reason"] = (
            "the pristine arm recorded a numeric fingerprint and the candidate did not, so the "
            "candidate's engine could not be compared against the reference"
        )
        return checks

    c, p_ = arms["candidate"], arms["pristine"]
    checks["candidate_positions"] = c.get("n_positions")
    checks["pristine_positions"] = p_.get("n_positions")
    checks["prompt_sha256_match"] = c.get("prompt_sha256") == p_.get("prompt_sha256")
    if not checks["prompt_sha256_match"]:
        checks["passed"] = False
        checks["reason"] = (
            f"the two arms scored different token sequences "
            f"({c.get('prompt_sha256')} vs {p_.get('prompt_sha256')}), so the comparison is void"
        )
        return checks
    cv, pv = c.get("logprobs") or [], p_.get("logprobs") or []
    if len(cv) != len(pv) or not cv:
        checks["passed"] = False
        checks["reason"] = f"logprob vectors do not align ({len(cv)} vs {len(pv)})"
        return checks

    deltas = [abs(a - b) for a, b in zip(cv, pv)]
    checks["n_positions"] = len(deltas)
    checks["max_abs_delta"] = max(deltas)
    checks["mean_abs_delta"] = sum(deltas) / len(deltas)
    checks["identical"] = max(deltas) == 0.0
    checks["sum_logprob_candidate"] = c.get("sum_logprob")
    checks["sum_logprob_pristine"] = p_.get("sum_logprob")
    checks["passed"] = max(deltas) <= NUMERIC_EQUIVALENCE_MAX_ABS_DELTA
    if not checks["passed"]:
        checks["reason"] = (
            f"the candidate engine computes different numbers from the reference: the largest "
            f"per-position logprob difference is {max(deltas):.6g} over {len(deltas)} positions, "
            f"against a limit of {NUMERIC_EQUIVALENCE_MAX_ABS_DELTA}. A repair does not change what "
            f"the model computes"
        )
    return checks


def _declared_workload() -> dict[str, object] | None:
    """The scored shape, from the SEALED tests/workloads.json -- the file _candidate_runner.py benches."""
    try:
        doc = json.loads((Path(__file__).resolve().parent / "workloads.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    workloads = doc.get("workloads") if isinstance(doc, dict) else None
    if not isinstance(workloads, list) or not workloads:
        return None
    primary = doc.get("primary_workload")
    for workload in workloads:
        if isinstance(workload, dict) and primary and workload.get("name") == primary:
            return workload
    return workloads[0] if isinstance(workloads[0], dict) else None


def check_workload_delivered(metrics: dict[str, object]) -> dict[str, object]:
    """Did the candidate's scored bench deliver the declared workload? THIS GATES (integrity)."""
    workload = _declared_workload() or {}
    try:
        expected_requests = int(workload["num_prompts"])
        expected_tokens = expected_requests * int(workload["output_len"])
    except (KeyError, TypeError, ValueError):
        expected_requests = expected_tokens = None
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    completed = _nested_number(summary.get("requests"), "completed_requests")
    generated = _nested_number(summary.get("tokens"), "output")
    checks: dict[str, object] = {
        "expected_requests": expected_requests,
        "completed_requests": completed,
        "expected_output_tokens": expected_tokens,
        "output_tokens": generated,
    }
    checks["passed"] = (
        expected_requests is not None and completed == expected_requests
        and expected_tokens is not None and generated == expected_tokens
    )
    if not checks["passed"]:
        checks["reason"] = (
            f"the candidate's bench did not deliver the declared workload: {completed} of "
            f"{expected_requests} requests and {generated} of {expected_tokens} generated tokens"
        )
    return checks


def upstream_benchmark_ready(checks: dict[str, object]) -> bool:
    return (
        bool(checks.get("raw_exists"))
        and bool(checks.get("manifest_exists"))
        and bool(checks.get("raw_json_object"))
        and bool(checks.get("manifest_json_object"))
        and checks.get("manifest_returncode") == 0
        and bool(checks.get("manifest_mentions_vllm_bench"))
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
    return normalize_sample(tests_dir, raw_dir, metrics_path)


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
    """`bench_rounds` is accepted for the shared call shape used across the family, and recorded."""
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

    metrics = normalize_real_or_sample(Path(args.tests_dir), raw_dir, Path(args.metrics))
    delivery_checks = check_workload_delivered(metrics)
    upstream_ready = (
        upstream_benchmark_ready(upstream_checks) and delivery_checks.get("passed") is True
    )
    scored = resolve_scored_arms(Path(args.tests_dir), raw_dir, metrics)
    bench_rounds = load_bench_rounds(raw_dir)
    performance_checks = check_performance(
        metrics,
        thresholds_by_benchmark=load_performance_thresholds(),
        scored=scored,
        bench_rounds=bench_rounds,
    )
    performance_ready = upstream_ready and bool(performance_checks.get("passed"))
    mechanism_checks = check_mechanism(raw_dir)
    equivalence_checks = check_numeric_equivalence(raw_dir)
    equivalence_ok = equivalence_checks.get("passed") is True
    reference_ok = equivalence_checks.get("pristine_measured") is True
    candidate_measured = equivalence_checks.get("candidate_measured") is True
    conformance_checks = check_configuration_conformance(raw_dir)
    conformance_ok = conformance_checks.get("passed") is not False
    stability_checks = check_tree_stability(raw_dir, repo)
    tree_stable = stability_checks.get("passed") is not False
    integrity_ok = upstream_ready and conformance_ok and tree_stable
    ratio = performance_checks.get("speedup_vs_baseline")
    edited = candidate_checks.get("file_edited")

    stands = (
        real_ready and upstream_ready and conformance_ok and tree_stable and performance_ready
        and equivalence_ok and reference_ok and candidate_measured and edited is True
    )
    reason = _reward_reason(
        reward=1.0 if stands else 0.0,
        static_reward=static_reward,
        upstream_ready=integrity_ok,
        performance_checks=performance_checks,
        outputs_ok=equivalence_ok,
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
        correctness_ok=equivalence_ok and reference_ok and candidate_measured,
        oracle_issue={
            "note": "diagnostic only, never a vote: did the gain come from the planted cause",
            "candidate_non_uniform_decode_steps": mechanism_checks.get("candidate_non_uniform"),
            "candidate_decode_steps": mechanism_checks.get("candidate_decode_steps"),
            "pristine_non_uniform_decode_steps": mechanism_checks.get("pristine_non_uniform"),
            "pristine_decode_steps": mechanism_checks.get("pristine_decode_steps"),
            "non_uniform_ratio": mechanism_checks.get("ratio"),
            "non_uniform_ratio_at_most_for_repair": MECHANISM_MAX_CANDIDATE_RATIO,
            "invariant_restored": mechanism_checks.get("passed"),
            "target_file_edited": candidate_checks.get("target_edited"),
            "changed_paths": candidate_checks.get("changed_paths"),
        },
        scored_metric="throughput." + SCORED_THROUGHPUT_KEY,
        higher_is_better=not SCORED_LOWER_IS_BETTER,
    )
    metrics["mechanism_checks"] = mechanism_checks
    metrics["numeric_equivalence_checks"] = equivalence_checks
    metrics["numeric_equivalence_ok"] = equivalence_ok
    metrics["tree_stability"] = stability_checks
    metrics["tree_stable"] = tree_stable
    metrics["configuration_conformance"] = conformance_checks
    metrics["configuration_conformance_ok"] = conformance_ok
    metrics["workload_delivery_checks"] = delivery_checks
    metrics["workload_integrity_ok"] = integrity_ok
    metrics["candidate_checks"] = candidate_checks
    metrics["static_check_fraction"] = static_fraction
    metrics["real_engine_checks"] = real_checks
    metrics["upstream_benchmark_checks"] = upstream_checks
    metrics["upstream_benchmark_ready"] = upstream_ready
    metrics["performance_checks"] = performance_checks
    metrics["scored_arms"] = scored
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
