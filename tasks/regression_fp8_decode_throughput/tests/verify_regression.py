#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Static verifier for regression_fp8_decode_throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib.metrics import normalize_engine_metrics  # noqa: E402


# BOTH files on the defect's path, because the repair is route-independent and the gate must be too.
# This was one file, `custom_op.py`, and that cost a correct answer: one agent diagnosed the defect
# from source with the network closed and fixed it by overriding `maybe_compile` for the op inside
# `mla_attention.py` -- compiling `dynamic=False` and marking only dim 0 via
# `torch._dynamo.maybe_mark_dynamic`, output bit-identical by `torch.equal` at B = 32/64/256/1024/2048.
# The harness measured it at 1.2454x against a 1.20 gate. It scored 0, because `file_edited` and
# `required_terms` were both evaluated against a file it had no reason to touch. Upstream split the
# change across two files; it did the same thing in one. Keying on the patch's SHAPE is
# patch-matching, not measurement.
TARGET_RELS = (
    "repo/vllm/model_executor/custom_op.py",
    "repo/vllm/model_executor/layers/attention/mla_attention.py",
)
# Kept for the sha comparison and for the reported target_path: it is the file with a sealed
# baseline hash, and the one an upstream-shaped repair edits.
TARGET_REL = TARGET_RELS[0]

# Post-fix only, and BELONGING TO THIS TASK. `dynamic=False` rather than `dynamic_arg_dims`,
# deliberately: a literal term is a LITERAL gate, and a correct fix that upstream did not happen to
# write would score as incorrect. This defect has more than one legitimate repair -- upstream made
# shape specialisation opt-in per op to stay backward compatible, but simply not compiling this op
# with every dimension symbolic also restores the throughput without changing a token of output.
# `dynamic=False` is the narrowest thing any specialising repair must say.
#
# WIDENED TO THE MECHANISM VOCABULARY: one literal is one spelling. Any of these four is a
# specialising edit; none of them is something an unrelated optimisation says. Checked against the
# pristine files at the pin rather than assumed -- all four appear ZERO times in both
# `custom_op.py` (12,459 B) and `mla_attention.py` (100,285 B), and all four are added-only in the
# reference patch, so presence cannot be satisfied by an untouched tree.
REQUIRED_TERMS = (
    "dynamic=False",
    "dynamic_arg_dims",
    "maybe_mark_dynamic",
    "mark_dynamic",
)


def add_repo_to_path() -> None:
    source_task = os.environ.get("INFB_SOURCE_TASK_DIR")
    if source_task:
        repo_root = Path(source_task).resolve().parents[1]
        sys.path.insert(0, str(repo_root))


def _tests_dir() -> Path:
    """Resolve the sealed tests dir.

    Prefers INFB_TESTS_DIR (set by the harness), but falls back to this
    script's own directory: the baseline sha256 ships alongside the verifier,
    so the fallback is correct even if the env var is missing.
    """
    tests_dir_env = os.environ.get("INFB_TESTS_DIR")
    if tests_dir_env and Path(tests_dir_env).is_dir():
        return Path(tests_dir_env)
    return Path(__file__).resolve().parent


def _baseline_sha() -> str | None:
    """SHA-256 of the target file at the pinned baseline commit.

    Returns None only when the sha256 file is genuinely absent, which score()
    treats as a FAILED check rather than a silent skip.
    """
    sha_path = _tests_dir() / "baseline_custom_op.py.sha256"
    if not sha_path.exists():
        return None
    raw = sha_path.read_text(encoding="utf-8").strip()
    return raw.split()[0] if raw else None


def _candidate_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_candidate(workspace: Path) -> dict[str, object]:
    candidate = workspace / TARGET_REL
    checks: dict[str, object] = {
        "target_path": TARGET_REL,
        "target_exists": candidate.exists(),
    }

    if candidate.exists():
        try:
            py_compile.compile(str(candidate), doraise=True)
            checks["target_compiles"] = True
        except py_compile.PyCompileError as exc:
            checks["target_compiles"] = False
            checks["compile_error"] = str(exc)
        text = candidate.read_text(encoding="utf-8")
        for rel in TARGET_RELS[1:]:
            other = workspace / rel
            if other.exists():
                text += "\n" + other.read_text(encoding="utf-8")
        checks["scanned_paths"] = [rel for rel in TARGET_RELS if (workspace / rel).exists()]
        checks["required_terms"] = {term: term in text for term in REQUIRED_TERMS}
        baseline_sha = _baseline_sha()
        candidate_sha = _candidate_sha(candidate)
        checks["candidate_sha256"] = candidate_sha
        checks["baseline_sha256"] = baseline_sha
        sha_changed = candidate_sha != baseline_sha if baseline_sha else None
        terms_present = any(checks["required_terms"].values())
        checks["sha_changed"] = sha_changed
        checks["mechanism_terms_present"] = terms_present
        # ROUTE-INDEPENDENT. The gate is "did a specialising repair happen on this path", not "did
        # you edit the file upstream happened to edit". Two signals, OR'd, because each covers the
        # other's blind spot:
        #
        #   sha_changed     the sealed hash of custom_op.py moved. Catches an upstream-shaped repair
        #                   however it is worded.
        #   terms_present   a mechanism term appears anywhere on the defect's path. Catches a repair
        #                   that lives entirely in mla_attention.py, which the sha check alone
        #                   false-FAILED at 1.2454x.
        #
        # RESIDUAL GAP, stated rather than hidden: a correct repair living outside custom_op.py that
        # specialises without naming any of the four torch APIs would still fail. That is narrower
        # than what it replaces -- the old check failed EVERY repair outside one file.
        checks["file_edited"] = (
            True if terms_present else sha_changed
        )
    else:
        checks["target_compiles"] = False
        checks["required_terms"] = {term: False for term in REQUIRED_TERMS}
        checks["file_edited"] = False

    return checks


def score(checks: dict[str, object]) -> tuple[float, float]:
    # required_terms are recorded in checks["required_terms"] as a metrics-only diagnostic (did the
    # candidate reproduce the upstream patch verbatim?) but they do NOT gate reward: a
    # functionally-correct fix written with different naming must not be false-FAILed.
    # file_edited is None when the sealed baseline sha256 is missing; an unresolvable baseline is a
    # FAILED check, because the sha256 ships in tests/, so absence means the task is mis-packaged.
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
