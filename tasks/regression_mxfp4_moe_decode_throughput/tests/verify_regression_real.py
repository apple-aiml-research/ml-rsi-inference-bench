#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for regression_mxfp4_moe_decode_throughput."""

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



TARGET_REL = "repo/python/sglang/srt/layers/quantization/mxfp4.py"

BASELINE_REPO_REL = "baseline_repo"
CANDIDATE_REPO_REL = "repo"

REQUIRED_TERMS = (
    "_is_sm90_supported",
)



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
        text = candidate.read_text(encoding="utf-8")
        checks["required_terms"] = {term: term in text for term in REQUIRED_TERMS}
        checks["candidate_sha256"] = _candidate_sha(candidate)
    else:
        checks["target_compiles"] = False
        checks["required_terms"] = {term: False for term in REQUIRED_TERMS}

    return checks


NUMERIC_PROBE_FILE = "moe_numeric_probe.json"

NUMERIC_TOP1_AGREEMENT_MIN: float | None = None


def check_numeric_equivalence(raw_dir: Path) -> dict[str, object]:
    """Do the two arms compute the same thing? RECORDED, NOT SCORED -- yet."""
    checks: dict[str, object] = {
        "passed": None,
        "top1_agreement_min": NUMERIC_TOP1_AGREEMENT_MIN,
        "probe_file": NUMERIC_PROBE_FILE,
    }
    arms: dict[str, dict] = {}
    for arm in ("candidate", "pristine"):
        for f in sorted((raw_dir.parent / "rounds").glob(f"*/{arm}/raw/{NUMERIC_PROBE_FILE}")):
            try:
                fp = json.loads(f.read_text()).get("numeric_fingerprint")
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(fp, dict) and fp.get("available"):
                arms[arm] = fp
                break
    checks["arms_available"] = sorted(arms)

    if "pristine" not in arms:
        checks["reason"] = (
            "the pristine arm recorded no numeric fingerprint, so the probe is not working in this "
            "image; nothing is asserted about the candidate"
        )
        return checks
    if "candidate" not in arms:
        checks["reason"] = (
            "the pristine arm recorded a fingerprint and the candidate did not, so the candidate's "
            "engine could not be compared against the reference"
        )
        return checks

    pr, cd = arms["pristine"], arms["candidate"]
    if pr.get("prompt_sha256") != cd.get("prompt_sha256"):
        checks["reason"] = (
            f"the two arms were probed with different inputs "
            f"({pr.get('prompt_sha256')} vs {cd.get('prompt_sha256')}); the comparison is void"
        )
        return checks
    checks["prompt_sha256"] = pr.get("prompt_sha256")
    checks["prompt_kind"] = pr.get("prompt_kind")

    a, b = pr.get("logprobs") or [], cd.get("logprobs") or []
    n = min(len(a), len(b))
    checks["n_positions_compared"] = n
    if n:
        deltas = [abs(float(a[i]) - float(b[i])) for i in range(n)]
        checks["max_abs_delta"] = round(max(deltas), 6)
        checks["mean_abs_delta"] = round(sum(deltas) / n, 6)
        checks["sum_logprob"] = {"pristine": pr.get("sum_logprob"), "candidate": cd.get("sum_logprob")}

    ta, tb = pr.get("top1_ids") or [], cd.get("top1_ids") or []
    m = min(len(ta), len(tb))
    checks["n_top1_compared"] = m
    if m:
        agree = sum(1 for i in range(m) if int(ta[i]) == int(tb[i]))
        checks["top1_agreement"] = round(agree / m, 6)
        checks["top1_disagreements"] = m - agree
        if NUMERIC_TOP1_AGREEMENT_MIN is not None:
            checks["passed"] = checks["top1_agreement"] >= NUMERIC_TOP1_AGREEMENT_MIN
            if checks["passed"] is False:
                checks["reason"] = (
                    f"the candidate's argmax differs from the pristine arm's at "
                    f"{m - agree} of {m} teacher-forced positions "
                    f"(agreement {checks['top1_agreement']}, floor {NUMERIC_TOP1_AGREEMENT_MIN}). "
                    f"A repair does not change what the model computes."
                )
    elif n:
        checks["reason"] = (
            "logprobs were compared but no top-1 ids were recorded, so the quantity a gate would "
            "use is missing; the server may not support top_logprobs_num on this path"
        )
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


REQUIRED_VLLM_PATHS = (
    "python/pyproject.toml",
    "python/sglang",
    "python/sglang/launch_server.py",
    "python/sglang/bench_serving.py",
    "python/sglang/srt/server_args.py",
    "python/sglang/srt/layers/quantization/mxfp4.py",
)

DEFAULT_PERFORMANCE_THRESHOLDS = {
    "latency": {
        "max_e2e_mean_ms": 300_000.0,
    },
    "throughput": {
        "min_request_per_sec": 0.1,
        "min_total_tokens_per_sec": 1.0,
    },
    "serve": {
        "min_request_per_sec": 0.5,
        "min_total_tokens_per_sec": 1.0,
        "max_ttft_mean_ms": 60_000.0,
    },
}

ENGINE_PACKAGE = "sglang"
ENGINE_PYPROJECT_REL = "python/pyproject.toml"
RAW_RECORD_NAMES = (
    "sglang_bench_serving.json",
    "vllm_bench_latency.json",
    "vllm_bench_throughput.json",
    "vllm_bench_serve.json",
)


def _first_existing_raw(raw_dir: Path) -> Path:
    """The raw bench record this task actually wrote, or the first candidate name for the message."""
    for name in RAW_RECORD_NAMES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    return raw_dir / RAW_RECORD_NAMES[0]


SCORED_AXIS = "throughput"

SCORED_THROUGHPUT_KEY = "output_tokens_per_sec"
SCORED_METRIC_KEY = "throughput.output_tokens_per_sec"
assert SCORED_METRIC_KEY == "throughput." + SCORED_THROUGHPUT_KEY

SCORED_LOWER_IS_BETTER = False

NOISE_FLOOR = 1.03


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
    pyproject = repo / ENGINE_PYPROJECT_REL
    declares = pyproject.exists() and f'name = "{ENGINE_PACKAGE}"' in pyproject.read_text(
        encoding="utf-8",
        errors="replace",
    )
    checks["pyproject_declares_vllm"] = declares
    checks["pyproject_declares_engine"] = {"engine": ENGINE_PACKAGE, "declared": declares,
                                           "path": str(pyproject)}
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
    manifest = raw_dir / "vllm_bench_manifest.json"
    selected = _first_existing_raw(raw_dir)
    checks: dict[str, object] = {
        "raw_record_names": list(RAW_RECORD_NAMES),
        "raw_record_path": str(selected),
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


def upstream_benchmark_ready(checks: dict[str, object]) -> bool:
    return (
        bool(checks.get("raw_exists"))
        and bool(checks.get("manifest_exists"))
        and bool(checks.get("raw_json_object"))
        and bool(checks.get("manifest_json_object"))
        and checks.get("manifest_returncode") == 0
        and bool(checks.get("manifest_mentions_vllm_bench"))
    )


def _unmeasured_metrics(metrics_path: Path) -> dict[str, object]:
    """Metrics for a run that produced NO bench output."""
    metrics: dict[str, object] = {
        "schema_version": 1,
        "metric_source": "vllm",
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
    for name in RAW_RECORD_NAMES:
        candidate = raw_dir / name
        if candidate.exists():
            return normalize_engine_metrics(
                candidate, metrics_path, metric_source=ENGINE_PACKAGE
            )
    return _unmeasured_metrics(metrics_path)


def load_baseline_tps(tests_dir: Path, raw_dir: Path | None = None) -> float | None:
    """Load the regressed-state baseline TPS."""
    if raw_dir is not None:
        rounds_mean = _round_arm_mean(load_bench_rounds(raw_dir), "pristine")
        if rounds_mean is not None:
            return rounds_mean
        baseline_raw = raw_dir.parent / "baseline" / "raw"
        bench_file = baseline_raw / "sglang_bench_serving.json"
        if bench_file.exists():
            try:
                tmp_metrics = normalize_engine_metrics(
                    bench_file,
                    baseline_raw / "_baseline_metrics.json",
                    metric_source="sglang",
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
        val = data.get(SCORED_THROUGHPUT_KEY)
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
    checks["scored_axis"] = SCORED_AXIS
    checks["scored_metric"] = SCORED_METRIC_KEY
    checks["scored_metric_key"] = SCORED_METRIC_KEY
    checks["scored_lower_is_better"] = SCORED_LOWER_IS_BETTER
    checks["candidate_tps"] = agent_tps
    checks["speedup_vs_baseline"] = speedup
    checks["paired_round_ratios"] = scored.get("paired_round_ratios")
    checks["speedup_paired_geomean"] = scored.get("speedup_paired_geomean")
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
    """A finite number at `values[key]`, else None."""
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None



CORRECTNESS_PROBE_FILE = "correctness_probe.json"
CORRECTNESS_ATOL = 0.05
CORRECTNESS_RTOL = 0.05
CORRECTNESS_MIN_PASS_RATE = 0.95
CORRECTNESS_MIN_PROMPTS = 8


def _finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _within_band(candidate: float, reference: float) -> bool:
    return abs(candidate - reference) <= CORRECTNESS_ATOL + CORRECTNESS_RTOL * abs(reference)


def _round_order(source: str) -> tuple[int, int | str]:
    return (0, int(source)) if source.isdigit() else (1, source)


def _load_probe(path: Path) -> tuple[dict | None, str | None]:
    """(usable probe, None), or (None, why it is not usable). Shape-strict on purpose: a schema the
    reader does not understand must read as "could not look", never as "the arms agree"."""
    if not path.is_file():
        return None, "absent"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({type(exc).__name__})"
    if not isinstance(doc, dict):
        return None, "not a JSON object"
    if doc.get("error"):
        return None, f"probe error: {str(doc['error'])[:300]}"
    prompts, declared = doc.get("prompts"), doc.get("n_prompts")
    if not isinstance(prompts, list) or isinstance(declared, bool) or not isinstance(declared, int):
        return None, "malformed probe record"
    if len(prompts) != declared:
        return None, f"answered {len(prompts)} of {declared} prompts"
    if declared < CORRECTNESS_MIN_PROMPTS:
        return None, f"only {declared} prompts (fewer than {CORRECTNESS_MIN_PROMPTS})"
    for entry in prompts:
        if (not isinstance(entry, dict) or not isinstance(entry.get("prompt_logprobs"), list)
                or not isinstance(entry.get("prompt_ids_sha256"), str)):
            return None, "malformed prompt entry"
    if not any(_finite(v) for entry in prompts for v in entry["prompt_logprobs"]):
        return None, "no prompt log-probability in any prompt"
    return doc, None


def _arm_probes(artifacts: Path, arm: str) -> tuple[dict[str, dict], list[str]]:
    """({round: usable probe}, [why each benched round has none]) for one arm."""
    probes: dict[str, dict] = {}
    errors: list[str] = []
    rounds = artifacts / "rounds"
    round_dirs = (sorted((d for d in rounds.iterdir() if d.is_dir()), key=lambda d: _round_order(d.name))
                  if rounds.is_dir() else [])
    if round_dirs:
        for directory in round_dirs:
            raw = directory / arm / "raw"
            if not raw.is_dir():
                continue
            probe, why = _load_probe(raw / CORRECTNESS_PROBE_FILE)
            if probe is not None:
                probes[directory.name] = probe
            elif (raw / RAW_RECORD_NAMES[0]).is_file():
                errors.append(f"round {directory.name}: {why}")
        return probes, errors
    published = artifacts / "raw" if arm == "candidate" else artifacts / "baseline" / "raw"
    probe, why = _load_probe(published / CORRECTNESS_PROBE_FILE)
    if probe is not None:
        probes["published"] = probe
    elif (published / RAW_RECORD_NAMES[0]).is_file():
        errors.append(f"published: {why}")
    return probes, errors


def _compare_prompt_logprobs(cand: dict, ref: dict) -> dict[str, object]:
    """THE VOTE: the same fixed text in, the same log-probabilities out? Candidate against reference."""
    out: dict[str, object] = {"compared": 0, "agreed": 0, "pass_rate": None, "max_abs_delta": None,
                              "prompts_tokenised_differently": [], "passed": False}
    if (cand.get("prompt_set_sha256") != ref.get("prompt_set_sha256")
            or cand.get("n_prompts") != ref.get("n_prompts")):
        out["reason"] = "the two arms were probed with different prompt sets"
        return out
    compared = agreed = top1_compared = top1_agreed = 0
    worst = 0.0
    for index, (c, r) in enumerate(zip(cand["prompts"], ref["prompts"])):
        if c.get("prompt_ids_sha256") != r.get("prompt_ids_sha256"):
            out["prompts_tokenised_differently"].append(index)
            continue
        c_lp, r_lp = c["prompt_logprobs"], r["prompt_logprobs"]
        for position, y in enumerate(r_lp):
            if not _finite(y):
                continue
            compared += 1
            x = c_lp[position] if position < len(c_lp) else None
            if not _finite(x):
                continue
            delta = abs(float(x) - float(y))
            worst = max(worst, delta)
            if _within_band(float(x), float(y)):
                agreed += 1
        for a, b in zip(c.get("prompt_top1") or [], r.get("prompt_top1") or []):
            if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
                top1_compared += 1
                top1_agreed += int(a == b)
    out.update(compared=compared, agreed=agreed,
               max_abs_delta=(worst if compared else None),
               pass_rate=(agreed / compared if compared else None),
               top1_compared=top1_compared,
               top1_agreement=(top1_agreed / top1_compared if top1_compared else None))
    if out["prompts_tokenised_differently"]:
        out["reason"] = (f"{len(out['prompts_tokenised_differently'])} of {cand.get('n_prompts')} "
                         f"prompts reached the two engines as different token ids")
    elif not compared:
        out["reason"] = "no position was comparable"
    elif agreed / compared < CORRECTNESS_MIN_PASS_RATE:
        out["reason"] = (f"{agreed} of {compared} positions within the band "
                         f"({agreed / compared:.4f} < {CORRECTNESS_MIN_PASS_RATE})")
    else:
        out["passed"] = True
    return out


def _token_ids(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    return [int(v) for v in values if isinstance(v, int) and not isinstance(v, bool)]


def _logprob_of(top: object, position: int, token: int) -> float | None:
    """The candidate's log-probability for `token` at `position`, from its top-k list there."""
    if not isinstance(top, list) or position >= len(top) or not isinstance(top[position], list):
        return None
    for pair in top[position]:
        if (isinstance(pair, list) and len(pair) == 2 and pair[0] == token
                and _finite(pair[1])):
            return float(pair[1])
    return None


def _compare_greedy(cand: dict, ref: dict) -> dict[str, object]:
    """DIAGNOSTIC ONLY, never a vote: the two arms' greedy continuations of the same prompts."""
    requests = identical = prefix = total = aligned = aligned_ok = 0
    first_divergence: list[int | None] = []
    for c, r in zip(cand.get("prompts") or [], ref.get("prompts") or []):
        if not isinstance(c, dict) or not isinstance(r, dict):
            continue
        c_ids, r_ids = _token_ids(c.get("output_ids")), _token_ids(r.get("output_ids"))
        requests += 1
        n = min(len(c_ids), len(r_ids))
        diverge = next((k for k in range(n) if c_ids[k] != r_ids[k]), n)
        same = bool(r_ids) and c_ids == r_ids
        identical += int(same)
        first_divergence.append(None if same else diverge)
        prefix += diverge
        total += len(r_ids)
        c_lp = c.get("output_logprobs") if isinstance(c.get("output_logprobs"), list) else []
        r_lp = r.get("output_logprobs") if isinstance(r.get("output_logprobs"), list) else []
        for k in range(min(diverge + 1, len(r_ids), len(r_lp))):
            y = r_lp[k]
            if not _finite(y):
                continue
            aligned += 1
            x = (c_lp[k] if k < len(c_lp) else None) if k < diverge else _logprob_of(
                c.get("output_top"), k, r_ids[k])
            if _finite(x) and _within_band(float(x), float(y)):
                aligned_ok += 1
    c_digest, r_digest = cand.get("output_digest"), ref.get("output_digest")
    return {
        "requests": requests,
        "requests_identical": identical,
        "digest_equal": (c_digest == r_digest) if isinstance(c_digest, str) and isinstance(r_digest, str) else None,
        "first_divergence": first_divergence,
        "prefix_agreement": (prefix / total) if total else None,
        "aligned_positions": aligned,
        "aligned_pass_rate": (aligned_ok / aligned) if aligned else None,
    }


def check_outputs_agree(raw_dir: Path) -> dict[str, object]:
    """Does the candidate assign the same log-probability to the same token in the same context?"""
    artifacts = raw_dir.parent
    cand, cand_errors = _arm_probes(artifacts, "candidate")
    prist, prist_errors = _arm_probes(artifacts, "pristine")
    checks: dict[str, object] = {
        "comparison": "teacher_forced_prompt_logprobs",
        "probe_file": CORRECTNESS_PROBE_FILE,
        "atol": CORRECTNESS_ATOL, "rtol": CORRECTNESS_RTOL,
        "min_pass_rate": CORRECTNESS_MIN_PASS_RATE,
        "candidate_rounds": sorted(cand, key=_round_order),
        "pristine_rounds": sorted(prist, key=_round_order),
        "candidate_errors": cand_errors,
        "pristine_errors": prist_errors,
    }
    reference_round = min(prist, key=_round_order) if prist else None
    reproducibility: dict[str, object] = {}
    for source in sorted(prist, key=_round_order):
        if source == reference_round:
            continue
        again = _compare_prompt_logprobs(prist[source], prist[reference_round])
        reproducibility[source] = {k: again.get(k) for k in ("pass_rate", "max_abs_delta", "passed",
                                                             "top1_agreement", "reason")}
    reproducible = all(entry["passed"] for entry in reproducibility.values())
    checks["reference_round"] = reference_round
    checks["reference_reproducibility"] = reproducibility
    checks["reference_reproducible"] = reproducible if prist else None
    checks["pristine_measured"] = bool(prist) and reproducible
    checks["candidate_measured"] = bool(cand)

    per_round: list[dict[str, object]] = []
    if prist:
        for source in sorted(cand, key=_round_order):
            against = source if source in prist else reference_round
            verdict = _compare_prompt_logprobs(cand[source], prist[against])
            verdict["round"] = source
            verdict["against"] = against
            verdict["greedy_decode"] = _compare_greedy(cand[source], prist[against])
            per_round.append(verdict)
    checks["rounds"] = per_round
    rates = [entry["pass_rate"] for entry in per_round if entry.get("pass_rate") is not None]
    deltas = [entry["max_abs_delta"] for entry in per_round if entry.get("max_abs_delta") is not None]
    checks["compared"] = sum(int(entry["compared"]) for entry in per_round)
    checks["agreed"] = sum(int(entry["agreed"]) for entry in per_round)
    checks["pass_rate"] = min(rates) if rates else None
    checks["max_abs_delta"] = max(deltas) if deltas else None
    checks["greedy_digests"] = {
        "note": "recorded, never a vote: see _compare_greedy",
        "candidate": {s: cand[s].get("output_digest") for s in sorted(cand, key=_round_order)},
        "pristine": {s: prist[s].get("output_digest") for s in sorted(prist, key=_round_order)},
        "candidate_stable_across_rounds": len({cand[s].get("output_digest") for s in cand}) <= 1,
        "pristine_stable_across_rounds": len({prist[s].get("output_digest") for s in prist}) <= 1,
    }
    agree = bool(checks["candidate_measured"] and checks["pristine_measured"] and per_round
                 and all(entry["passed"] for entry in per_round))
    checks["outputs_agree"] = agree
    if not prist:
        checks["reason"] = "the pristine arm left no usable probe: nothing was compared"
    elif not reproducible:
        checks["reason"] = "the pristine arm's probes do not reproduce each other"
    elif not cand:
        checks["reason"] = "the candidate arm left no usable probe beside a working reference"
    elif not agree:
        failing = [entry for entry in per_round if not entry["passed"]]
        checks["reason"] = f"round {failing[0]['round']}: {failing[0].get('reason')}"
    return checks


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
    upstream_ready = upstream_benchmark_ready(upstream_checks)
    output_checks = check_outputs_agree(raw_dir)
    outputs_ok = output_checks.get("outputs_agree") is True

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
    edited_paths = candidate_checks.get("edited_paths")

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
            "required_terms": candidate_checks.get("required_terms"),
            "target_file_edited": (
                TARGET_REL.removeprefix(f"{CANDIDATE_REPO_REL}/") in edited_paths
                if isinstance(edited_paths, list) else None
            ),
        },
        scored_metric=SCORED_METRIC_KEY,
        higher_is_better=not SCORED_LOWER_IS_BETTER,
    )
    metrics["candidate_checks"] = candidate_checks
    metrics["static_check_fraction"] = static_fraction
    metrics["real_engine_checks"] = real_checks
    metrics["upstream_benchmark_checks"] = upstream_checks
    metrics["upstream_benchmark_ready"] = upstream_ready
    metrics["correctness_checks"] = output_checks
    metrics["outputs_agree"] = outputs_ok
    metrics["bench_rounds"] = bench_rounds
    metrics["performance_checks"] = performance_checks
    metrics["performance_ready"] = performance_ready
    metrics["numeric_equivalence"] = check_numeric_equivalence(raw_dir)
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
    rewards["outputs_agree"] = 1.0 if outputs_ok else 0.0
    Path(args.reward).write_text(
        json.dumps(rewards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
