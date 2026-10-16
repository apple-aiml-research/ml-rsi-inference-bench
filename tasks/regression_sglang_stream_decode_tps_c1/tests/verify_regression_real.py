#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real-engine verifier for regression_sglang_stream_decode_tps_c1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import py_compile
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path


def _lib(module: str):
    """Load `_verifier_lib/<module>.py` from THIS file's own directory, by path."""
    import importlib.util

    path = Path(__file__).resolve().parent / "_verifier_lib" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"{__name__}__lib_{module}", path)
    assert spec is not None and spec.loader is not None, path
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


usable_round_arms = _lib("bench_stats").usable_round_arms
normalize_engine_metrics = _lib("metrics").normalize_engine_metrics
_scoring = _lib("regression_scoring")
SCORING_CONTRACT = _scoring.CONTRACT
compute_regression_reward = _scoring.compute_regression_reward
resolve_rounds_ratio = _scoring.resolve_rounds_ratio




BASELINE_REPO_REL = "baseline_repo"
CANDIDATE_REPO_REL = "repo"

REQUIRED_TERMS = (
    "needs_output_ids",
)

SCORED_THROUGHPUT_KEY = "output_tokens_per_sec"

MAX_SYNC_ITL_RATIO = 1.30

CROSSCHECK_NAME = "crosscheck_stream_client.json"


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
    """Static checks, and none of them may name a file."""
    candidate_root = workspace / CANDIDATE_REPO_REL
    checks: dict[str, object] = {"target_path": f"{CANDIDATE_REPO_REL}/<edited *.py>"}

    edited, unreadable = _tree_diff(workspace)
    checks["edited_paths"] = edited
    checks["unreadable_paths"] = unreadable
    checks["target_exists"] = candidate_root.is_dir()
    checks["file_edited"] = None if (unreadable or edited is None) else bool(edited)

    if not candidate_root.is_dir() or edited is None:
        checks["target_compiles"] = False
        checks["required_terms"] = {term: False for term in REQUIRED_TERMS}
        return checks

    failures: dict[str, str] = {}
    corpus: list[str] = []
    for rel in edited:
        if not rel.endswith(".py"):
            continue
        path = candidate_root / rel
        if not path.is_file():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures[rel] = str(exc)
        try:
            corpus.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    checks["target_compiles"] = not failures
    if failures:
        checks["compile_error"] = json.dumps(failures)[:2000]
    checks["compiled_paths"] = sorted(r for r in edited if r.endswith(".py"))
    blob = "\n".join(corpus)
    checks["required_terms"] = {term: term in blob for term in REQUIRED_TERMS}
    checks["candidate_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    return checks


def check_output_equivalence(artifacts_dir: Path) -> dict[str, object]:
    """Did the candidate still emit the SAME TOKENS as the pristine tree?"""
    PROBES = ("crosscheck_stream_client.json",)

    def _read_probe(path: Path) -> tuple[str | None, int | None, int | None]:
        """(digest, generated, requested) -- (None, None, None) if the arm produced nothing."""
        if not path.is_file():
            return None, None, None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, None, None
        digest = doc.get("output_digest")
        want = doc.get("output_len_requested")
        got = doc.get("min_completion_tokens")
        return (digest if isinstance(digest, str) else None,
                got if isinstance(got, int) else None,
                want if isinstance(want, int) else None)

    def _non_degenerate(cand: tuple, prist: tuple) -> bool:
        """Equality is only evidence when there is an output to be equal about."""
        _, c_got, c_want = cand
        _, p_got, p_want = prist
        if None in (c_got, p_got, c_want, p_want) or c_want != p_want or c_want <= 0:
            return False
        return c_got == p_got and c_got >= 0.9 * c_want

    rounds = artifacts_dir / "rounds"
    checks: dict[str, object] = {"rounds_dir": str(rounds), "pairs": [],
                                 "candidate_measured": False, "pristine_measured": False}
    if not rounds.is_dir():
        checks["output_equivalent"] = None
        checks["reason"] = "no rounds/ directory -- nothing to compare"
        return checks

    pairs: list[dict[str, object]] = []
    for round_dir in sorted(rounds.iterdir()):
        if not round_dir.is_dir():
            continue
        for probe in PROBES:
            arm_data: dict[str, tuple] = {}
            for arm in ("candidate", "pristine"):
                arm_data[arm] = _read_probe(round_dir / arm / "raw" / probe)
            c_dig, c_got, _ = arm_data["candidate"]
            p_dig, p_got, _ = arm_data["pristine"]
            if c_dig and p_dig:
                match = bool(c_dig == p_dig
                             and _non_degenerate(arm_data["candidate"], arm_data["pristine"]))
            else:
                match = None
            pairs.append({
                "round": round_dir.name,
                "probe": probe,
                "candidate": c_dig,
                "pristine": p_dig,
                "candidate_generated": c_got,
                "pristine_generated": p_got,
                "match": match,
            })
    checks["pairs"] = pairs
    checks["pristine_measured"] = any(bool(p["pristine"]) for p in pairs)
    checks["candidate_measured"] = any(bool(p["candidate"]) and bool(p["pristine"]) for p in pairs)
    checks["min_new_tokens_floor"] = _read_floor_probes(rounds)
    decided = [p["match"] for p in pairs if p["match"] is not None]
    undecided_probes = sorted({
        str(p["probe"]) for p in pairs if p["probe"] not in
        {q["probe"] for q in pairs if q["match"] is not None}
    })
    if not decided:
        checks["output_equivalent"] = None
        checks["reason"] = "no round produced a digest on both arms"
    elif undecided_probes:
        checks["output_equivalent"] = None
        checks["reason"] = f"no round decided these probes on both arms: {undecided_probes}"
    else:
        checks["output_equivalent"] = all(decided)
        if not all(decided):
            bad = sorted({str(p["probe"]) for p in pairs if p["match"] is False})
            checks["reason"] = (
                "the candidate did not reproduce the pristine tree's tokens, or an arm did not "
                f"generate in full: {bad}"
            )
    return checks


def _json_number(path: Path, key: str) -> float | None:
    """A finite, positive number at `key` in the JSON object at `path`, else None. Never raises."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = doc.get(key) if isinstance(doc, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None



def _read_floor_probes(rounds: Path) -> dict:
    """The one check that looks at the path the repair is actually on. RECORDED, NOT GATED YET."""
    out: dict = {"arms": [], "note": "scored: two classifications fail, the rest are inconclusive"}
    if not rounds.is_dir():
        return out
    for round_dir in sorted(rounds.iterdir()):
        if not round_dir.is_dir():
            continue
        for arm in ("candidate", "pristine"):
            path = round_dir / arm / "raw" / "min_new_tokens_floor.json"
            if not path.is_file():
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                doc = {"error": f"unreadable: {exc}"}
            doc["round"], doc["arm"] = round_dir.name, arm
            doc["would_pass"] = doc.get("classification") == "floor_enforced_and_released"
            out["arms"].append(doc)
    if out["arms"]:
        out["all_arms_would_pass"] = all(a.get("would_pass") for a in out["arms"])
        broken = sorted({a["classification"] for a in out["arms"]
                         if a.get("classification") in ("never_released", "floor_not_enforced")})
        out["floor_broken"] = bool(broken)
        out["broken_classifications"] = broken
    return out


def check_sync_itl_ratio(artifacts_dir: Path) -> dict[str, object]:
    """DIAGNOSTIC, never a vote: did the gain come from removing the per-step host sync?"""
    rounds = artifacts_dir / "rounds"
    per_arm: dict[str, list[float]] = {"candidate": [], "pristine": []}
    if rounds.is_dir():
        for round_dir in sorted(rounds.iterdir()):
            if not round_dir.is_dir():
                continue
            for arm, series in per_arm.items():
                raw = round_dir / arm / "raw"
                scored_itl = _json_number(raw / RAW_THROUGHPUT_NAME, "inter_token_latency_ms")
                control_itl = _json_number(raw / CROSSCHECK_NAME, "inter_token_latency_ms")
                if scored_itl is not None and control_itl is not None:
                    series.append(scored_itl / control_itl)
    candidate = statistics.median(per_arm["candidate"]) if per_arm["candidate"] else None
    pristine = statistics.median(per_arm["pristine"]) if per_arm["pristine"] else None
    return {
        "candidate_sync_itl_ratios": per_arm["candidate"],
        "pristine_sync_itl_ratios": per_arm["pristine"],
        "candidate_sync_itl_ratio": candidate,
        "pristine_sync_itl_ratio": pristine,
        "sync_itl_ratio_at_most_for_repair": MAX_SYNC_ITL_RATIO,
        "sync_removed": (candidate <= MAX_SYNC_ITL_RATIO) if candidate is not None else None,
    }


def score(checks: dict[str, object]) -> tuple[float, float]:
    """THE CORRECTNESS VERDICT, as (verdict, verdict): 1.0 exactly when the outputs held."""
    correctness = 1.0 if checks.get("output_equivalent") is True else 0.0
    floor = checks.get("min_new_tokens_floor") or {}
    if floor.get("floor_broken"):
        correctness = 0.0
    reward = correctness
    return reward, correctness



REQUIRED_ENGINE_PATHS = (
    "pyproject.toml",
    "sglang",
    "sglang/srt/server_args.py",
    "sglang/srt/managers/scheduler.py",
    "sglang/srt/model_executor/model_runner.py",
    "sglang/srt/model_executor/cuda_graph_runner.py",
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


ENGINE_DISTRIBUTION = "sglang"
ENGINE_PYPROJECT_PATHS = ("pyproject.toml", "python/pyproject.toml")


def check_real_engine_repo(repo: Path) -> dict[str, object]:
    checks: dict[str, object] = {
        "repo_path": str(repo),
        "repo_exists": repo.exists(),
        "git_commit": repo_commit(repo),
        "required_paths": {},
    }
    checks["required_paths"] = {rel: (repo / rel).exists() for rel in REQUIRED_ENGINE_PATHS}
    declared = False
    for rel in ENGINE_PYPROJECT_PATHS:
        pyproject = repo / rel
        if not pyproject.exists():
            continue
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        if f'name = "{ENGINE_DISTRIBUTION}"' in text:
            declared = True
            checks["pyproject_path"] = rel
            break
    checks["pyproject_declares_engine"] = declared
    return checks


def all_real_checks_pass(checks: dict[str, object]) -> bool:
    required = checks.get("required_paths", {})
    return (
        bool(checks.get("repo_exists"))
        and bool(checks.get("pyproject_declares_engine"))
        and isinstance(required, dict)
        and all(bool(value) for value in required.values())
    )


RAW_THROUGHPUT_NAME = "sglang_bench_throughput.json"
RAW_MANIFEST_NAME = "sglang_bench_manifest.json"


def check_upstream_benchmark(raw_dir: Path) -> dict[str, object]:
    raw_throughput = raw_dir / RAW_THROUGHPUT_NAME
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
                checks["engine_kwargs"] = raw.get("engine_kwargs")
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
                server_command = manifest_data.get("server_command")
                client_command = manifest_data.get("scored_client_command")
                checks["manifest_command"] = command or server_command
                checks["manifest_client_command"] = client_command
                offline_shape = (
                    isinstance(command, str) and "sglang" in command and "bench" in command
                )
                server_shape = (
                    isinstance(server_command, str)
                    and "sglang.launch_server" in server_command
                    and isinstance(client_command, str)
                    and bool(client_command.strip())
                )
                checks["manifest_mentions_engine_bench"] = offline_shape or server_shape
                checks["decode_batches_reported"] = manifest_data.get("decode_batches_reported")
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
        and bool(checks.get("manifest_mentions_engine_bench"))
    )


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
    raw_throughput = raw_dir / RAW_THROUGHPUT_NAME
    if raw_throughput.exists():
        return normalize_engine_metrics(raw_throughput, metrics_path, metric_source="sglang")
    return _unmeasured_metrics(metrics_path)


def load_baseline_tps(tests_dir: Path, raw_dir: Path | None = None) -> float | None:
    """The regressed-state baseline throughput, on the scored key."""
    if raw_dir is None:
        return None
    rounds_mean = _round_arm_mean(load_bench_rounds(raw_dir), "pristine")
    if rounds_mean is not None:
        return rounds_mean
    bench_file = raw_dir.parent / "baseline" / "raw" / RAW_THROUGHPUT_NAME
    if not bench_file.exists():
        return None
    try:
        tmp_metrics = normalize_engine_metrics(
            bench_file,
            bench_file.parent / "_baseline_metrics.json",
            metric_source="sglang",
        )
        tps = _nested_number(
            tmp_metrics.get("summary", {}).get("throughput", {}), SCORED_THROUGHPUT_KEY
        )
        if isinstance(tps, (int, float)) and tps > 0:
            return float(tps)
    except Exception:  # noqa: BLE001 -- a malformed baseline is "no baseline", not a crash
        return None
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


SCORED_BENCHMARK_KINDS = ("throughput",)

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
    checks["scored_metric"] = SCORED_THROUGHPUT_KEY
    checks["scored_lower_is_better"] = SCORED_LOWER_IS_BETTER
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
        elif scored.get("candidate_arm_ok") is True:
            failures.append("insufficient_data")
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
    if "insufficient_data" in failed:
        return "insufficient_data"
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
    raw_dir = Path(args.raw_dir)
    candidate_checks = check_candidate(workspace)
    equivalence = check_output_equivalence(raw_dir.parent)
    candidate_checks.update(equivalence)
    static_reward, correctness = score(candidate_checks)
    repo = locate_candidate_repo(workspace)
    real_checks = check_real_engine_repo(repo)
    real_ready = all_real_checks_pass(real_checks)
    upstream_checks = check_upstream_benchmark(raw_dir)
    upstream_ready = upstream_benchmark_ready(upstream_checks)
    sync_checks = check_sync_itl_ratio(raw_dir.parent)

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
    floor_broken = bool((candidate_checks.get("min_new_tokens_floor") or {}).get("floor_broken"))
    outputs_ok = candidate_checks.get("output_equivalent") is True and not floor_broken
    candidate_measured = equivalence.get("candidate_measured") is True
    pristine_measured = equivalence.get("pristine_measured") is True

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
            **sync_checks,
            "required_terms": candidate_checks.get("required_terms"),
        },
        scored_metric="throughput." + SCORED_THROUGHPUT_KEY,
        higher_is_better=not SCORED_LOWER_IS_BETTER,
    )
    metrics["candidate_checks"] = candidate_checks
    metrics["correctness"] = correctness
    metrics["real_engine_checks"] = real_checks
    metrics["upstream_benchmark_checks"] = upstream_checks
    metrics["upstream_benchmark_ready"] = upstream_ready
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
    Path(args.reward).write_text(
        json.dumps(rewards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
