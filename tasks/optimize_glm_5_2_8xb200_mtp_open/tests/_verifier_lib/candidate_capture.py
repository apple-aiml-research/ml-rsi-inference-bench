#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Capture untracked candidate files before verifier scoring."""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


_ROOT_BOOKKEEPING_NAMES = frozenset({".infb_autoresearch", "iterations", "result.jsonl"})
_CACHE_DIRECTORY_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_COMPILED_SUFFIXES = (".pyc", ".pyo")
_PRIVATE_DIRECTORY_NAMES = frozenset(
    {
        ".aws",
        ".claude",
        ".codex",
        ".gemini",
        ".ssh",
    }
)
_PRIVATE_CREDENTIAL_FILENAMES = (
    ".credentials.json",
    ".env",
    "auth.json",
    "cert.pem",
    "codex-auth.json",
    "credentials",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "oauth_creds.json",
    "openai-api-key",
    "private.key",
    "private.pem",
    "project-token",
)
_PRIVATE_FILENAMES = frozenset(
    {
        *_PRIVATE_CREDENTIAL_FILENAMES,
        ".env",
        "credentials",
        "id_ed25519",
        "id_rsa",
        "private.key",
    }
)
_PRIVATE_FILE_SUFFIXES = (".p12", ".pfx")

_ARTIFACT_FILENAMES = frozenset(
    {
        "bench_progress.log",
        "result.json",
        "result.jsonl",
        "bench.json",
        "dmon.txt",
        "torch_compile.log",
        "profile.nsys-rep",
        "profile.qdstrm",
        "profile.sqlite",
        "nsys_partial_stdout.log",
        "nsys_partial_stderr.log",
        "serve_profile.log",
        "sc_gate_candidate.log",
        "sc_gate_pristine.log",
        "prompts.json",
        "correctness_prompts.json",
        "baseline.npz",
    }
)
_ARTIFACT_NAME_SHAPES = (
    ("bench_", ".json"),
    ("vllm_serve_", ".log"),
    ("nsys_stats_", ".txt"),
)
_ARTIFACT_SUFFIXES = (
    ".progress.log",
    ".nsys-rep",
    ".qdstrm",
    ".pt.trace.json",
    ".pt.trace.json.gz",
)
_ARTIFACT_DIRECTORY_NAMES = frozenset({"_paired_pristine_worktree"})

CAPTURE_VOID_REASON = "candidate_capture_failed"
_STDERR_TAIL_CHARS = 800
CAPTURE_RECORD_MAX_PATHS = 20

_COMMIT_MESSAGE = "infb: capture untracked candidate files for scoring"
_COMMIT_IDENTITY_ARGS = (
    "-c",
    "user.name=inference-bench",
    "-c",
    "user.email=inference-bench@localhost",
    "-c",
    "commit.gpgsign=false",
)

_CAPTURE_CONFIG_OVERRIDES = (
    "-c",
    "core.excludesFile=",
    "-c",
    "core.quotePath=false",
    "-c",
    "diff.external=",
)


class CandidateCaptureError(RuntimeError):
    """A git operation needed to capture the candidate tree failed."""

    def __init__(self, message: str, *, operation: str, stderr: str = "") -> None:
        super().__init__(message)
        self.operation = operation
        self.stderr_tail = (stderr or "").strip()[-_STDERR_TAIL_CHARS:]


def capture_void_reason(exc: CandidateCaptureError) -> dict:
    """Build the ``infra_void`` reason dict for a failed capture."""
    return {
        "reason": CAPTURE_VOID_REASON,
        "note": (
            "the candidate's untracked files could not be made visible to scoring "
            "-> RE-RUN; scoring an incomplete diff would under-score the model"
        ),
        "operation": getattr(exc, "operation", "unknown"),
        "error": str(exc),
        "git_stderr_tail": getattr(exc, "stderr_tail", ""),
        "remedy": (
            "usually a stale <repo>/.git/index.lock left by a killed agent process, "
            "a 'detected dubious ownership' refusal (repo owned by another uid), or "
            "a git too old for --pathspec-from-file"
        ),
    }


def _is_bookkeeping_path(value: str) -> bool:
    """Harness-bookkeeping / build output — safe to keep out of the scored tree."""
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return False
    parts = tuple(part.lower() for part in path.parts)
    if any(part in _CACHE_DIRECTORY_NAMES for part in parts):
        return True
    if parts[-1].endswith(_COMPILED_SUFFIXES):
        return True
    root = parts[0]
    return root in _ROOT_BOOKKEEPING_NAMES or root.startswith("run_benchmark_")


def _is_credential_path(value: str) -> bool:
    """Credential / private paths that must never be committed."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return True
    parts = tuple(part.lower() for part in path.parts)
    name = parts[-1]
    root = parts[0]
    if root in _PRIVATE_DIRECTORY_NAMES:
        return True
    if any(
        parts[index : index + 2] in (("agent", "sessions"), (".config", "opencode"))
        for index in range(max(0, len(parts) - 1))
    ):
        return True
    if name in _PRIVATE_FILENAMES or name.startswith(".env."):
        return True
    return name.endswith(_PRIVATE_FILE_SUFFIXES)


def _is_excluded_path(value: str) -> bool:
    """Kept out of the captured candidate tree (credential OR bookkeeping)."""
    return _is_credential_path(value) or _is_bookkeeping_path(value)


def _is_benchmark_artifact_path(value: str) -> bool:
    """Output of the task's own bench / correctness / profile scripts."""
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return False
    parts = tuple(part.lower() for part in path.parts)
    if any(part in _ARTIFACT_DIRECTORY_NAMES for part in parts):
        return True
    name = parts[-1]
    if name in _ARTIFACT_FILENAMES:
        return True
    if any(name.startswith(prefix) and name.endswith(suffix)
           for prefix, suffix in _ARTIFACT_NAME_SHAPES):
        return True
    return name.endswith(_ARTIFACT_SUFFIXES)


def _is_capture_excluded_path(value: str) -> bool:
    """Everything capture refuses to stage: credential, bookkeeping, or artifact."""
    return _is_excluded_path(value) or _is_benchmark_artifact_path(value)


def _git(
    repo: Path, *args: str, operation: str, stdin: str | None = None
) -> subprocess.CompletedProcess:
    """Run git, converting an ``OSError`` into :class:`CandidateCaptureError`."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *_CAPTURE_CONFIG_OVERRIDES, *args],
            capture_output=True,
            text=True,
            input=stdin,
        )
    except OSError as exc:
        raise CandidateCaptureError(
            f"could not run git to {operation}: {exc}",
            operation=operation,
            stderr=str(exc),
        ) from exc


def _untracked_paths(repo: Path) -> list[str]:
    """Untracked, non-gitignored files, repo-relative (one path per file)."""
    operation = "list untracked candidate files"
    result = _git(repo, "ls-files", "--others", "--exclude-standard", "-z",
                  operation=operation)
    if result.returncode != 0:
        raise CandidateCaptureError(
            f"could not list untracked candidate files: {result.stderr.strip()}",
            operation=operation,
            stderr=result.stderr,
        )
    return [p for p in result.stdout.split("\0") if p]


def _record(status: str, *, captured: list[str], excluded_count: int) -> dict:
    """The forensic record stamped into ``metrics["candidate_capture"]``."""
    ordered = sorted(captured)
    return {
        "status": status,
        "captured_file_count": len(ordered),
        "captured_paths": ordered[:CAPTURE_RECORD_MAX_PATHS],
        "captured_paths_omitted": max(0, len(ordered) - CAPTURE_RECORD_MAX_PATHS),
        "excluded_file_count": excluded_count,
    }


def capture_untracked_candidate(repo: str | Path) -> dict:
    """Commit the agent's untracked files so they are scored on the public path."""
    repo = Path(repo)
    if not (repo / ".git").exists():
        return _record("no_git", captured=[], excluded_count=0)

    untracked = _untracked_paths(repo)
    to_add = [p for p in untracked if not _is_capture_excluded_path(p)]
    excluded_count = len(untracked) - len(to_add)
    if not to_add:
        return _record("nothing_to_capture", captured=[], excluded_count=excluded_count)

    nul = "\0".join(to_add)
    operation = "stage untracked candidate files"
    added = _git(repo, "add", "--pathspec-from-file=-", "--pathspec-file-nul",
                 stdin=nul, operation=operation)
    if added.returncode != 0:
        raise CandidateCaptureError(
            f"could not stage untracked candidate files: {added.stderr.strip()}",
            operation=operation,
            stderr=added.stderr,
        )

    operation = "inspect staged candidate files"
    staged = _git(repo, "diff", "--cached", "--name-only", operation=operation)
    if staged.returncode != 0:
        raise CandidateCaptureError(
            f"could not inspect staged candidate files: {staged.stderr.strip()}",
            operation=operation,
            stderr=staged.stderr,
        )
    if not staged.stdout.strip():
        return _record("nothing_staged", captured=[], excluded_count=excluded_count)

    operation = "commit captured candidate files"
    committed = _git(
        repo,
        *_COMMIT_IDENTITY_ARGS,
        "commit",
        "--no-verify",
        "--no-gpg-sign",
        "--pathspec-from-file=-",
        "--pathspec-file-nul",
        "-m",
        _COMMIT_MESSAGE,
        stdin=nul,
        operation=operation,
    )
    if committed.returncode != 0:
        raise CandidateCaptureError(
            f"could not commit captured candidate files: {committed.stderr.strip()}",
            operation=operation,
            stderr=committed.stderr,
        )
    return _record("captured", captured=to_add, excluded_count=excluded_count)
