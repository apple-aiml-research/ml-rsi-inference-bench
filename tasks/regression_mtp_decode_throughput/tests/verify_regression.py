#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Static checks for regression_mtp_decode_throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib.metrics import normalize_engine_metrics  # noqa: E402


TARGET_REL = "repo/vllm/model_executor/layers/fla/ops/fused_recurrent.py"

REQUIRED_TERMS = (
    "N: tl.int64",
    "T: tl.int64",
)


def add_repo_to_path() -> None:
    source_task = os.environ.get("INFB_SOURCE_TASK_DIR")
    if source_task:
        repo_root = Path(source_task).resolve().parents[1]
        sys.path.insert(0, str(repo_root))


def _tests_dir() -> Path:
    """Resolve the sealed tests dir."""
    tests_dir_env = os.environ.get("INFB_TESTS_DIR")
    if tests_dir_env and Path(tests_dir_env).is_dir():
        return Path(tests_dir_env)
    return Path(__file__).resolve().parent


def _baseline_sha() -> str | None:
    """SHA-256 of the target file at the pinned baseline commit."""
    sha_path = _tests_dir() / "baseline_fused_recurrent.py.sha256"
    if not sha_path.exists():
        return None
    raw = sha_path.read_text(encoding="utf-8").strip()
    return raw.split()[0] if raw else None


def _candidate_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _engine_tree_edited(workspace: Path) -> bool | None:
    """Does `repo/` differ from the sealed `inference_base` tag anywhere under vllm/?"""
    repo = workspace / "repo"
    if not (repo / ".git").exists():
        return None
    lines: list[str] = []
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
        and not line.endswith(".so")
        and not line.endswith("_version.py")
        and "third_party/triton_kernels" not in line
        and "vllm_flash_attn/" not in line
        and "__pycache__/" not in line
        and not line.endswith(".pyc")
    ]
    return bool(changed)


def check_candidate(workspace: Path) -> dict[str, object]:
    candidate = workspace / TARGET_REL
    checks: dict[str, object] = {
        "target_path": TARGET_REL,
        "target_exists": candidate.exists(),
    }
    checks["file_edited"] = _engine_tree_edited(workspace)

    if candidate.exists():
        try:
            py_compile.compile(str(candidate), doraise=True)
            checks["target_compiles"] = True
        except py_compile.PyCompileError as exc:
            checks["target_compiles"] = False
            checks["compile_error"] = str(exc)
        text = candidate.read_text(encoding="utf-8")
        checks["required_terms"] = {term: term in text for term in REQUIRED_TERMS}
        baseline_sha = _baseline_sha()
        candidate_sha = _candidate_sha(candidate)
        checks["candidate_sha256"] = candidate_sha
        checks["baseline_sha256"] = baseline_sha
        checks["anchor_file_edited"] = (
            candidate_sha != baseline_sha if baseline_sha else None
        )
    else:
        checks["target_compiles"] = False
        checks["required_terms"] = {term: False for term in REQUIRED_TERMS}
        checks["anchor_file_edited"] = True if (workspace / "repo").is_dir() else None

    return checks


def score(checks: dict[str, object]) -> tuple[float, float]:
    """The static checks as (all-passed, fraction). A DIAGNOSTIC under the continuous contract."""
    booleans: list[bool] = [
        bool(checks.get("target_exists")),
        bool(checks.get("target_compiles")),
    ]
    booleans.append(checks.get("file_edited") is True)

    correctness = sum(1 for b in booleans if b) / len(booleans)
    reward = 1.0 if correctness == 1.0 else 0.0
    return reward, correctness


def normalize_sample(
    tests_dir: Path, raw_dir: Path, metrics_path: Path
) -> dict[str, object]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "sample_vllm_bench.json"
    shutil.copy2(tests_dir / "sample_vllm_bench.json", raw_path)
    return normalize_engine_metrics(raw_path, metrics_path, metric_source="vllm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--tests-dir", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    checks = check_candidate(workspace)
    reward, correctness = score(checks)
    metrics = normalize_sample(Path(args.tests_dir), Path(args.raw_dir), Path(args.metrics))
    metrics["candidate_checks"] = checks
    metrics["correctness"] = correctness
    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(args.reward).write_text(
        json.dumps(
            {
                "reward": reward,
                "correctness": correctness,
                "performance": 1.0 if reward == 1.0 else 0.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
