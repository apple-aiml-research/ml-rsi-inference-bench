#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for regression_fp8_decode_throughput."""

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
from _verifier_lib.metrics import (  # noqa: E402
    load_records,
    normalize_engine_metrics,
    standardize_record,
)
from _verifier_lib.regression_scoring import (  # noqa: E402
    CONTRACT as SCORING_CONTRACT,
    compute_regression_reward,
    resolve_rounds_ratio,
)

from verify_regression import (  # noqa: E402
    check_candidate,
    normalize_sample,
    score,
)


REQUIRED_VLLM_PATHS = (
    "pyproject.toml",
    "vllm",
    "vllm/model_executor/models/registry.py",
)

DEFAULT_PERFORMANCE_THRESHOLDS = {
    "throughput": {},
    "serve": {},
}

#: The ratio under which a run scores 0 as `no_speedup`, because an untouched tree reads there.
#: KEEP IN SYNC with task.toml [metadata.inference_bench.scoring].noise_floor.
NOISE_FLOOR = 1.03
#: The dotted path under metrics["summary"] both arms are compared on. tests/test.sh passes the same
#: string to _bench_rounds.py as --metric-key, and task.toml declares it as `measured_key`.
SCORED_METRIC_KEY = "latency_ms.tpot.mean"
#: Mean TPOT is a cost: the faster tree has the SMALLER number, so every ratio here is
#: pristine / candidate, and above 1.0 means the candidate is faster.
SCORED_LOWER_IS_BETTER = True

#: The benchmark kinds this task is willing to score. `_verifier_lib/metrics.py::_detect_benchmark_kind`
#: GUESSES the kind from JSON key names, and `vllm bench serve` -- the only bench this task's runner
#: runs -- normalises to `serve`. Anything else is a record this task did not produce, and it fails
#: closed as workload integrity rather than being scored against a bucket nobody declared for it.
SCORED_BENCHMARK_KINDS = ("serve",)

#: Scored rounds required per arm. 1 would be the n=1 estimator the rounds rig replaced.
MIN_ROUNDS_PER_ARM = 2
MATRIX_KEYS = ("speedup_per_workload", "worst_workload_speedup", "best_workload_speedup",
               "scored_workloads", "guard_workloads", "anchor_workload", "guard_ratios",
               "guards_ok", "guard_unmeasured", "unusable_workloads")

#: The record `_candidate_runner.py` writes for every arm, and the one every scored reader looks for.
RAW_SERVE_NAME = "vllm_bench_serve.json"

#: tests/test.sh's per-arm verdict of tests/assert_defect_path_is_exercised.py.
PATH_PROBE_NAME = "exercised_probe.json"

#: Failed checks that say "this run did not deliver the declared workload" and vote as workload
#: integrity (`candidate_failed`). `incomplete_benchmark` is deliberately NOT here: it has its own
#: token, which the board classifies as a non-result -- see _reward_reason.
_INTEGRITY_FAILURES = (
    "failed_requests",
    "candidate_path_not_exercised",
    "benchmark_kind_not_scored",
)


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
    # vLLM <= 0.7.x predates PEP 621 packaging: the `[project] name = "vllm"`
    # only appears in pyproject.toml from ~0.8.x. At the pinned 0.7.0 commit the
    # package name is declared in setup.py instead, so accept that too.
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


def _engine_tree_edited(workspace: Path) -> bool | None:
    """Does `repo/` differ from its own `inference_base` tag anywhere under vllm/?

    THE `no_patch` QUESTION, asked of the whole engine tree. The static half's `file_edited`
    (verify_regression.py) answers a narrower one -- "did custom_op.py's sealed hash move, or does a
    specialising term appear in custom_op.py or mla_attention.py" -- which is the PLANTED SITE, not the
    tree. Keyed on that, an unrelated repair would read as "nothing changed" and land on `no_patch`, a
    non-result. That check is kept as a diagnostic under `oracle_issue_checks`; this one decides.

    None when the question cannot be answered -- no repo, no tag, no git -- which the verdict reads as
    an undiffable tree (`static_checks`): prepare_target_env.sh tags every staged tree, so only the
    workspace itself can have lost the tag.

    Compiled artifacts and the overlays test.sh and preparation link in from the built wheel are
    excluded; they show as changed on every trial whatever the candidate did, and counting them would
    make this check always-true, which is the same as not having it.
    """
    repo = workspace / "repo"
    if not (repo / ".git").exists():
        return None
    lines: list[str] = []
    # MODIFIED tracked files, and then UNTRACKED new ones. `git diff <tag>` shows only the first, so a
    # candidate that adds a module and imports it would have read "nothing changed".
    for cmd in (
        ["git", "-C", str(repo), "diff", "--name-only", "inference_base", "--", "vllm"],
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "--", "vllm"],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        lines.extend(proc.stdout.splitlines())
    changed = [
        line for line in lines
        if line.strip()
        and not line.endswith((".so", ".pyc"))
        and not line.endswith("_version.py")
        and "__pycache__/" not in line
        and "third_party/triton_kernels" not in line
        and "third_party/flashmla" not in line
        and "vllm_flash_attn/" not in line
    ]
    return bool(changed)


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


def upstream_benchmark_ready(checks: dict[str, object]) -> bool:
    return (
        bool(checks.get("raw_exists"))
        and bool(checks.get("manifest_exists"))
        and bool(checks.get("raw_json_object"))
        and bool(checks.get("manifest_json_object"))
        and checks.get("manifest_returncode") == 0
        and bool(checks.get("manifest_mentions_vllm_bench"))
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


CORRECTNESS_DISTINCT_K_FLOOR: float | None = None
CORRECTNESS_LOGPROB_BAND: float | None = 0.07

_UNMEASURED_KINDS = frozenset({
    "no_records", "request_errors", "empty_generations", "nan_logprobs", "too_few_decode_positions",
})


def _read_recording(path: Path) -> tuple[dict | None, str | None]:
    """(document, error) for one arm's correctness_recording.json."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"recording unavailable: {type(exc).__name__}: {exc}"
    if not isinstance(doc, dict):
        return None, "recording is not a JSON object"
    if doc.get("error"):
        return None, str(doc.get("error"))
    records = doc.get("records")
    if not isinstance(records, list) or not records:
        return None, "recording carries no records"
    return doc, None


def check_output_correctness(raw_dir: Path | None = None) -> dict[str, object]:
    """Compare the candidate arm's greedy continuations against the baseline arm's."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _correctness  # noqa: PLC0415

    raw_dir = raw_dir or Path(os.environ.get("INFB_RAW_DIR", "artifacts/raw"))
    cand_path = raw_dir / "correctness_recording.json"
    base_path = raw_dir.parent / "baseline" / "raw" / "correctness_recording.json"

    checks: dict[str, object] = {
        "candidate_recording": str(cand_path),
        "baseline_recording": str(base_path),
    }
    cand_doc, cand_err = _read_recording(cand_path)
    base_doc, base_err = _read_recording(base_path)
    checks["candidate_error"] = cand_err
    checks["pristine_error"] = base_err
    if cand_doc is None or base_doc is None:
        checks["passed"] = False
        checks["reason"] = base_err or cand_err
        checks["candidate_measured"] = cand_doc is not None
        checks["pristine_measured"] = base_doc is not None
        return checks

    summary = _correctness.compare(
        base_doc.get("records", []), cand_doc.get("records", []),
        distinct_k_floor=CORRECTNESS_DISTINCT_K_FLOOR,
        logprob_band=CORRECTNESS_LOGPROB_BAND,
    )
    checks.update(summary)
    failures = [f for f in summary.get("failures", []) if isinstance(f, dict)]
    checks["candidate_measured"] = not any(
        f.get("arm") == "candidate" and f.get("kind") in _UNMEASURED_KINDS for f in failures
    )
    checks["pristine_measured"] = not any(
        f.get("arm") == "baseline" and f.get("kind") in _UNMEASURED_KINDS for f in failures
    )
    return checks


def _expected_request_count() -> int | None:
    """How many requests the scored workload declares, read from the workload file itself."""
    here = Path(__file__).resolve().parent
    candidates = [here / "workloads.json"]
    for base in (here.parent, Path("/app")):
        candidates.append(base / "environment" / "public" / "workloads.json")
        candidates.append(base / "public" / "workloads.json")
    for wl in candidates:
        if not wl.is_file():
            continue
        try:
            doc = json.loads(wl.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        primary = doc.get("primary_workload")
        for entry in doc.get("workloads", []):
            if entry.get("name") == primary:
                value = entry.get("num_prompts")
                return int(value) if isinstance(value, (int, float)) else None
    return None


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


def load_baseline_tpot(raw_dir: Path | None = None) -> float | None:
    """Mean TPOT of the published pristine bench, in ms -- the LEGACY single-bench denominator."""
    if raw_dir is None:
        return None
    baseline_raw = raw_dir.parent / "baseline" / "raw"
    for name in ("vllm_bench_serve.json", "vllm_bench_throughput.json", "vllm_bench_latency.json"):
        bench_file = baseline_raw / name
        if not bench_file.exists():
            continue
        try:
            tmp = normalize_engine_metrics(
                bench_file, baseline_raw / "_baseline_metrics.json", metric_source="vllm"
            )
        except Exception:  # noqa: BLE001 -- a malformed baseline is "no baseline", not a crash
            continue
        value = _scored_value(tmp.get("summary", {}))
        if value is not None:
            return value
    return None


def load_baseline_tps(tests_dir: Path, raw_dir: Path | None = None) -> float | None:
    """The published pristine bench's total tokens/sec -- for `throughput_ratio_not_scored` only."""
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
    """The ABSOLUTE floors (req/s, tok/s, latency caps): the baked table, refined by task.toml when
    the task tree is shipped whole (Pier ships it, tests/ one level below task.toml)."""
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


def _arm_samples(rounds: dict[str, object] | None, arm: str) -> int:
    """How many scored rounds one arm of the aggregate produced (its `n_ok`), 0 when unknown."""
    arms = rounds.get("arms") if isinstance(rounds, dict) else None
    entry = arms.get(arm) if isinstance(arms, dict) else None
    count = entry.get("n_ok") if isinstance(entry, dict) else None
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def _count(value: object) -> float | None:
    """A finite, non-boolean number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _bench_delivery(bench: Path, arm: str, source: str) -> dict[str, object]:
    """One scored bench's request counts, read through the shared normaliser's field mapping."""
    try:
        records = load_records(bench)
    except (OSError, ValueError):
        records = []
    standard = standardize_record(records[-1], metric_source="vllm") if records else {}
    requests = standard.get("requests", {}) if isinstance(standard, dict) else {}
    requests = requests if isinstance(requests, dict) else {}
    return {
        "arm": arm, "source": source,
        "completed": _count(requests.get("completed_requests")),
        "failed": _count(requests.get("failed_requests")),
    }


def _round_deliveries(raw_dir: Path | None) -> list[dict[str, object]]:
    """The request counts of EVERY scored bench the rounds rig ran, one entry per arm per round."""
    if raw_dir is None:
        return []
    return [
        _bench_delivery(bench, bench.parent.parent.name, f"rounds/{bench.parent.parent.parent.name}")
        for bench in sorted((raw_dir.parent / "rounds").glob(f"*/*/raw/{RAW_SERVE_NAME}"))
        if bench.parent.parent.name in ("pristine", "candidate")
    ]


def _is_canned_sample(metrics: dict[str, object]) -> bool:
    """Did normalize_real_or_sample() fall back to tests/sample_vllm_bench.json?"""
    source = metrics.get("source_path") if isinstance(metrics, dict) else None
    return isinstance(source, str) and Path(source).name == "sample_vllm_bench.json"


def _single_bench_arms(baseline: float | None, candidate: float | None) -> dict[str, object]:
    """Both arms from ONE single-bench pair, or neither: the legacy estimator."""
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


def resolve_scored_arms(
    tests_dir: Path,
    raw_dir: Path | None,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Both arms of the scored ratio, from ONE estimator or from NEITHER."""
    rounds = load_bench_rounds(raw_dir)
    if rounds is not None:
        resolved = dict(resolve_rounds_ratio(
            rounds, usable=usable_round_arms(rounds, MIN_ROUNDS_PER_ARM),
            lower_is_better=SCORED_LOWER_IS_BETTER,
        ))
        resolved["deliveries"] = _round_deliveries(raw_dir)
        return resolved
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    canned = _is_canned_sample(metrics)
    resolved = _single_bench_arms(load_baseline_tpot(raw_dir), None if canned else _scored_value(summary))
    deliveries: list[dict[str, object]] = []
    if raw_dir is not None:
        for arm, bench in (("pristine", raw_dir.parent / "baseline" / "raw" / RAW_SERVE_NAME),
                           ("candidate", raw_dir / RAW_SERVE_NAME)):
            if bench.is_file():
                deliveries.append(_bench_delivery(bench, arm, "single_bench"))
    resolved["deliveries"] = deliveries
    return resolved


def load_defect_path_probe(raw_dir: Path | None) -> dict[str, object]:
    """Per-arm verdict of tests/assert_defect_path_is_exercised.py, as tests/test.sh recorded it."""
    out: dict[str, object] = {
        "recorded": False, "pristine_exercised": None, "candidate_exercised": None,
    }
    if raw_dir is None:
        return out
    path = raw_dir / PATH_PROBE_NAME
    out["path"] = str(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(doc, dict):
        return out
    out["recorded"] = True
    for arm in ("pristine", "candidate"):
        entry = doc.get(arm)
        entry = entry if isinstance(entry, dict) else {}
        verdict = entry.get("verdict")
        verdict = verdict if isinstance(verdict, dict) else {}
        rc = entry.get("rc")
        out[arm] = {
            "rc": rc,
            "exercised": verdict.get("exercised"),
            "reasons": verdict.get("reasons"),
            "kv_cache_dtype": verdict.get("kv_cache_dtype"),
            "custom_ops": verdict.get("custom_ops"),
            "heads_per_rank": verdict.get("heads_per_rank"),
            "rows_x_heads": verdict.get("rows_x_heads"),
        }
        out[f"{arm}_exercised"] = (
            isinstance(rc, int) and not isinstance(rc, bool) and rc == 0
            and verdict.get("exercised") is True
        )
    return out




def check_performance(
    metrics: dict[str, object],
    *,
    thresholds_by_benchmark: dict[str, dict[str, float]] | None = None,
    scored: dict[str, object] | None = None,
    bench_rounds: dict[str, object] | None = None,
    path_probe: dict[str, object] | None = None,
    baseline_tpot: float | None = None,
    baseline_tps: float | None = None,
) -> dict[str, object]:
    """Workload integrity, the liveness floors, and the scored ratio against NOISE_FLOOR."""
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    benchmark = summary.get("benchmark", {})
    benchmark_name = benchmark.get("name") if isinstance(benchmark, dict) else None
    thresholds_by_benchmark = thresholds_by_benchmark or load_performance_thresholds()
    thresholds = thresholds_by_benchmark.get(str(benchmark_name)) or {}
    checks: dict[str, object] = {
        "benchmark": benchmark_name,
        "thresholds": thresholds,
        "passed": False,
    }
    throughput = summary.get("throughput", {})
    latency_ms = summary.get("latency_ms", {})
    failures: list[str] = []
    if scored is None:
        scored = _single_bench_arms(baseline_tpot, _scored_value(summary))

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

    expected = _expected_request_count()
    requests = summary.get("requests", {})
    checks["expected_requests"] = expected
    checks["completed_requests"] = _nested_number(requests, "completed_requests")
    checks["failed_requests"] = _nested_number(requests, "failed_requests")
    deliveries = scored.get("deliveries") if isinstance(scored, dict) else None
    if not isinstance(deliveries, list):
        deliveries = [{
            "arm": "candidate", "source": "summary",
            "completed": checks["completed_requests"], "failed": checks["failed_requests"],
        }]
    checks["deliveries"] = deliveries
    short = [
        d for d in deliveries
        if expected is None or _count(d.get("completed")) is None or d["completed"] < expected
    ]
    errored = [d for d in deliveries if (_count(d.get("failed")) or 0.0) > 0]
    if short:
        checks["incomplete_runs"] = short
        failures.append("incomplete_benchmark")
    if errored:
        checks["runs_with_failed_requests"] = errored
        failures.append("failed_requests")

    if isinstance(path_probe, dict):
        checks["defect_path"] = path_probe
        if path_probe.get("pristine_exercised") is not True:
            failures.append("pristine_path_not_exercised")
        if path_probe.get("candidate_exercised") is not True:
            failures.append("candidate_path_not_exercised")

    pristine_value = scored.get("baseline_tps")
    candidate_value = scored.get("candidate_tps")
    estimator = scored.get("estimator")
    speedup = scored.get("speedup")
    if (speedup is None and _count(pristine_value) and _count(candidate_value)
            and pristine_value > 0 and candidate_value > 0):
        speedup = ((pristine_value / candidate_value) if SCORED_LOWER_IS_BETTER
                   else (candidate_value / pristine_value))
    if speedup is not None and _count(speedup) is None:
        speedup = None
    checks["scored_metric"] = SCORED_METRIC_KEY
    checks["scored_lower_is_better"] = SCORED_LOWER_IS_BETTER
    checks["speedup_metric"] = "mean_tpot_ms (pristine/candidate; lower TPOT is faster)"
    checks["baseline_tpot_ms"] = pristine_value
    checks["candidate_tpot_ms"] = candidate_value
    checks["speedup_vs_baseline"] = speedup
    checks["paired_round_ratios"] = scored.get("paired_round_ratios")
    checks["speedup_paired_geomean"] = scored.get("speedup_paired_geomean")
    checks["scored_arms_reason"] = scored.get("reason")
    checks["speedup_estimator"] = estimator
    checks["scored_estimator"] = estimator
    checks["pristine_arm_ok"] = scored.get("pristine_arm_ok")
    for _key in MATRIX_KEYS:
        if _key in scored:
            checks[_key] = scored[_key]
    checks["candidate_arm_ok"] = scored.get("candidate_arm_ok")
    checks["noise_floor"] = NOISE_FLOOR
    candidate_throughput = _nested_number(throughput, "total_tokens_per_sec")
    checks["throughput_ratio_not_scored"] = (
        (candidate_throughput / baseline_tps) if candidate_throughput and baseline_tps else None
    )
    if isinstance(bench_rounds, dict):
        checks["rounds"] = {
            key: bench_rounds.get(key)
            for key in (
                "ratio", "ratio_stdev", "ratio_direction", "arm_order", "warmup_runs",
                "warmup_plateau", "first_scored_run_vs_warmup_plateau", "elapsed_sec", "notes",
            )
        }
        arms_summary = bench_rounds.get("arms")
        if isinstance(arms_summary, dict):
            checks["rounds"]["arms"] = arms_summary
    if estimator is None or pristine_value is None or candidate_value is None:
        checks["baseline_tpot_missing"] = pristine_value is None
        pristine_ok = scored.get("pristine_arm_ok") is True
        candidate_ok = scored.get("candidate_arm_ok") is True
        if pristine_ok and (
            not candidate_ok
            or _arm_samples(bench_rounds, "candidate") < _arm_samples(bench_rounds, "pristine")
        ):
            failures.append("candidate_tps_unavailable")
        else:
            failures.append("baseline_tps_unavailable")
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
    if "pristine_path_not_exercised" in failed:
        return "pristine_failed"
    if not reference_ok:
        return "pristine_failed"
    if not diffable_ok:
        return "static_checks"
    if not edited_ok:
        return "no_patch"
    if "incomplete_benchmark" in failed:
        return "incomplete_benchmark"
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
    tree_edited = _engine_tree_edited(workspace)
    repo = locate_candidate_repo(workspace)
    real_checks = check_real_vllm_repo(repo)
    real_ready = all_real_checks_pass(real_checks)
    raw_dir = Path(args.raw_dir)
    upstream_checks = check_upstream_benchmark(raw_dir)
    upstream_ready = upstream_benchmark_ready(upstream_checks)

    metrics = normalize_real_or_sample(Path(args.tests_dir), raw_dir, Path(args.metrics))
    bench_rounds = load_bench_rounds(raw_dir)
    path_probe = load_defect_path_probe(raw_dir)
    performance_checks = check_performance(
        metrics,
        thresholds_by_benchmark=load_performance_thresholds(),
        scored=resolve_scored_arms(Path(args.tests_dir), raw_dir, metrics),
        bench_rounds=bench_rounds,
        path_probe=path_probe,
        baseline_tps=load_baseline_tps(Path(args.tests_dir), raw_dir),
    )
    failed_checks = performance_checks.get("failed_checks") or []
    workload_ok = upstream_ready and not any(name in failed_checks for name in _INTEGRITY_FAILURES)
    performance_ready = real_ready and workload_ok and bool(performance_checks.get("passed"))
    output_checks = check_output_correctness(raw_dir)
    output_ok = bool(output_checks.get("passed"))
    candidate_measured = output_checks.get("candidate_measured") is True
    pristine_measured = output_checks.get("pristine_measured") is True
    ratio = performance_checks.get("speedup_vs_baseline")
    if tree_edited is not None:
        edited = tree_edited
    else:
        edited = True if candidate_checks.get("file_edited") is True else None

    reward = 1.0 if performance_ready and output_ok and edited is True else 0.0
    reason = _reward_reason(
        reward=reward,
        static_reward=static_reward,
        upstream_ready=workload_ok,
        performance_checks=performance_checks,
        outputs_ok=output_ok,
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
        correctness_ok=output_ok and candidate_measured and pristine_measured,
        oracle_issue={
            "note": "diagnostic only, never a vote: did the gain come from the planted cause",
            "planted_site_edited": candidate_checks.get("file_edited"),
            "custom_op_sha_changed": candidate_checks.get("sha_changed"),
            "mechanism_terms_present": candidate_checks.get("mechanism_terms_present"),
            "required_terms": candidate_checks.get("required_terms"),
        },
        scored_metric=SCORED_METRIC_KEY,
        higher_is_better=not SCORED_LOWER_IS_BETTER,
    )
    metrics["output_correctness_checks"] = output_checks
    metrics["output_correctness_passed"] = output_ok
    metrics["candidate_checks"] = candidate_checks
    metrics["static_check_fraction"] = static_fraction
    metrics["engine_tree_edited"] = tree_edited
    metrics["real_engine_checks"] = real_checks
    metrics["upstream_benchmark_checks"] = upstream_checks
    metrics["upstream_benchmark_ready"] = upstream_ready
    metrics["workload_delivered"] = workload_ok
    metrics["defect_path_checks"] = path_probe
    metrics["bench_rounds"] = bench_rounds
    metrics["performance_checks"] = performance_checks
    metrics["performance_ready"] = performance_ready
    metrics["score_profile"] = SCORING_CONTRACT
    metrics["real_engine_ready"] = performance_ready
    metrics.update(verdict["metrics"])
    metrics["reason"] = reason
    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rewards = dict(verdict["rewards"])
    rewards["real_engine_ready"] = 1.0 if performance_ready else 0.0
    rewards["upstream_benchmark_ready"] = 1.0 if upstream_ready else 0.0
    Path(args.reward).write_text(
        json.dumps(rewards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
