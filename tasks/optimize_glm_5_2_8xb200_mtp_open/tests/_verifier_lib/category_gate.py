#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Patch shape classification for the throughput-optimization category gate."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable


INVOCATION_FILE_PATTERNS = (
    re.compile(r"\.sh$"),
    re.compile(r"\.bash$"),
    re.compile(r"^scripts/.*"),
    re.compile(r"^environment/scripts/.*"),
)
CONFIG_FILE_PATTERNS = (
    re.compile(r"vllm/config\.py$"),
    re.compile(r"vllm/config/[a-z_]+\.py$"),
    re.compile(r"^[^/]*\.yaml$"),
    re.compile(r"^[^/]*\.yml$"),
    re.compile(r"^[^/]*\.toml$"),
)
DOC_FILE_PATTERNS = (
    re.compile(r"\.md$"),
    re.compile(r"\.rst$"),
    re.compile(r"\.txt$"),
    re.compile(r"^docs/.*"),
)
ARTIFACT_FILE_PATTERNS = (
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"\.pyc$"),
    re.compile(r"\.pyo$"),
    re.compile(r"\.so$"),
    re.compile(r"\.o$"),
    re.compile(r"(^|/)\.pytest_cache/"),
    re.compile(r"(^|/)\.ninja_(deps|log)$"),
)

TEST_FILE_PATTERNS = (
    re.compile(r"^tests/.*"),
    re.compile(r"_test\.py$"),
    re.compile(r"test_.*\.py$"),
)


def classify_diff(diff_text: str) -> dict:
    """Classify a unified diff."""
    if not diff_text or not diff_text.strip():
        return {
            "kind": "empty",
            "files_touched": [],
            "source_files_touched": [],
            "config_files_touched": [],
            "invocation_files_touched": [],
            "doc_files_touched": [],
            "test_files_touched": [],
            "reason": "diff is empty",
        }

    files = sorted(_files_from_diff(diff_text))
    if not files:
        return {
            "kind": "unknown",
            "files_touched": [],
            "source_files_touched": [],
            "config_files_touched": [],
            "invocation_files_touched": [],
            "doc_files_touched": [],
            "test_files_touched": [],
            "reason": "could not parse any files from diff",
        }

    invocation, config, doc, test, source, artifact = [], [], [], [], [], []
    for f in files:
        if any(p.search(f) for p in ARTIFACT_FILE_PATTERNS):
            artifact.append(f)
        elif any(p.search(f) for p in TEST_FILE_PATTERNS):
            test.append(f)
        elif any(p.search(f) for p in INVOCATION_FILE_PATTERNS):
            invocation.append(f)
        elif any(p.search(f) for p in CONFIG_FILE_PATTERNS):
            config.append(f)
        elif any(p.search(f) for p in DOC_FILE_PATTERNS):
            doc.append(f)
        else:
            source.append(f)

    if artifact and not (source or config or invocation or doc or test):
        return {
            "kind": "empty",
            "files_touched": files,
            "source_files_touched": [],
            "config_files_touched": [],
            "invocation_files_touched": [],
            "doc_files_touched": [],
            "test_files_touched": [],
            "artifact_files_touched": artifact,
            "reason": (
                f"the whole diff is {len(artifact)} build artefact(s) "
                f"({', '.join(artifact[:3])}{'…' if len(artifact) > 3 else ''}) "
                "and no source, config, invocation, doc or test file"
            ),
        }
    has_source = bool(source)
    only_invocation = bool(invocation) and not (source or config)
    only_config = bool(config) and not (source or invocation)

    is_config_default_only = False
    if only_config:
        is_config_default_only = _is_literal_value_flip_only(diff_text)

    if has_source:
        kind = "mixed" if (invocation or config) else "source_edit"
        reason = (
            f"substantive source edits in {len(source)} file(s); "
            f"also touches {len(invocation)} invocation, {len(config)} config, "
            f"{len(doc)} doc files"
        )
    elif only_invocation:
        kind = "pure_flag_flip"
        reason = f"only invocation/script edits ({len(invocation)} file(s))"
    elif is_config_default_only:
        kind = "config_default_flip"
        reason = f"only config-module literal-default flips ({len(config)} file(s)): {', '.join(config)}"
    elif only_config:
        kind = "config_default_flip"
        reason = f"config edits only ({len(config)} file(s)); not a pure literal flip"
    elif doc and not (invocation or config):
        kind = "pure_flag_flip"
        reason = f"docs-only edits ({len(doc)} file(s))"
    else:
        kind = "mixed"
        reason = (
            f"no source edits; mix of {len(invocation)} invocation, "
            f"{len(config)} config, {len(doc)} doc, {len(test)} test files"
        )

    return {
        "kind": kind,
        "files_touched": files,
        "source_files_touched": source,
        "artifact_files_touched": artifact,
        "config_files_touched": config,
        "invocation_files_touched": invocation,
        "doc_files_touched": doc,
        "test_files_touched": test,
        "reason": reason,
    }


_DIFF_CONFIG_OVERRIDES = (
    "-c", "diff.external=",
    "-c", "diff.noprefix=false",
    "-c", "diff.srcPrefix=a/",
    "-c", "diff.dstPrefix=b/",
)
_DIFF_HARDENING_FLAGS = ("--no-ext-diff", "--no-textconv")


def classify_repo_diff(repo_dir: str | Path, base_ref: str = "inference_base") -> dict:
    """Classify the diff between a repo's HEAD/working tree and base_ref."""
    repo_dir = Path(repo_dir)
    unknown = {
        "kind": "unknown",
        "files_touched": [],
        "source_files_touched": [],
        "config_files_touched": [],
        "invocation_files_touched": [],
        "doc_files_touched": [],
        "test_files_touched": [],
    }
    if not (repo_dir / ".git").exists():
        return {**unknown, "reason": f"no .git in {repo_dir}"}

    def _diff(*extra: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), *_DIFF_CONFIG_OVERRIDES, "diff",
             *_DIFF_HARDENING_FLAGS, "--unified=0", *extra],
            capture_output=True, check=True,
        )
        return completed.stdout.decode("utf-8", errors="replace")

    try:
        try:
            diff = _diff(base_ref)
        except subprocess.CalledProcessError:
            diff = _diff()
        return classify_diff(diff)
    except Exception as exc:  # noqa: BLE001 -- see the docstring: this must never raise
        return {**unknown, "reason": f"classification failed: {type(exc).__name__}: {exc}"}




def _files_from_diff(diff_text: str) -> set[str]:
    """Extract the b-side (post-image) file path from each diff hunk header."""
    files: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                files.add(parts[3][2:])
        elif line.startswith("+++ b/"):
            files.add(line[6:])
        elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            stripped = line[4:].strip()
            if stripped and stripped != "/dev/null":
                files.add(stripped)
    return files


def _is_literal_value_flip_only(diff_text: str) -> bool:
    """True if every hunk is a pure literal value flip (single line replacement)."""
    flip_re = re.compile(
        r"^[-+]\s*(\w+\s*[:=]\s*|.+default[:=]\s*|.+default_factory.*lambda.*:\s*)"
        r"(True|False|None|-?\d+\.?\d*|\".*\"|'.*')\s*[,)]?\s*$"
    )
    saw_replacement = False
    for line in diff_text.splitlines():
        if not (line.startswith("+") or line.startswith("-")):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if not flip_re.match(line):
            return False
        saw_replacement = True
    return saw_replacement


def is_rejected_by_narrow(classification: dict) -> bool:
    """Whether the narrow-variant verifier should zero this patch's reward."""
    return classification.get("kind") in {"pure_flag_flip", "config_default_flip"}


def render_summary(classification: dict) -> str:
    """One-line human-readable summary, for verifier metrics."""
    kind = classification.get("kind", "unknown")
    files = len(classification.get("files_touched", []))
    src = len(classification.get("source_files_touched", []))
    return f"kind={kind} files={files} source={src} reason={classification.get('reason','')}"
