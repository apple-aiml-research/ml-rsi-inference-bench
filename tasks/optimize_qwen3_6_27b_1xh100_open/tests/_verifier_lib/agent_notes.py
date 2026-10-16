#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Copy the agent's own markdown notes into the artifact tree before scoring."""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

_NOTE_SUFFIXES = (".md", ".markdown", ".rst")

_PRIVATE_ROOT_DIRECTORY_NAMES = frozenset(
    {
        ".aws",
        ".claude",
        ".codex",
        ".gemini",
        ".ssh",
    }
)
_EXCLUDED_DIRECTORY_NAMES = frozenset({"_paired_pristine_worktree"})

MAX_FILE_BYTES = 1 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_FILES = 50
RECORD_MAX_PATHS = 20

_CONFIG_OVERRIDES = ("-c", "core.excludesFile=", "-c", "core.quotePath=false")

_BASE_REF = "inference_base"
_DEFAULT_DEST_NAME = "agent_notes"
_STDERR_TAIL_CHARS = 400


def _git(repo: Path, *args: str) -> str:
    """Run git and return stdout. Raises ``_GitError`` — never propagates."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *_CONFIG_OVERRIDES, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise _GitError(f"git {args[0]}: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-_STDERR_TAIL_CHARS:]
        raise _GitError(f"git {' '.join(args)} exited {proc.returncode}: {tail}")
    return proc.stdout


class _GitError(RuntimeError):
    """Internal only. :func:`collect_agent_notes` converts it into a record."""


def _is_note_path(value: str) -> bool:
    """A repo-relative path this module is willing to copy."""
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return False
    if not path.name.lower().endswith(_NOTE_SUFFIXES):
        return False
    parts = tuple(part.lower() for part in path.parts)
    if parts[0] in _PRIVATE_ROOT_DIRECTORY_NAMES:
        return False
    return not any(part in _EXCLUDED_DIRECTORY_NAMES for part in parts)


def _changed_paths(repo: Path, base_ref: str) -> list[str]:
    """Notes the candidate added or edited relative to ``base_ref``."""
    out = _git(
        repo, "diff", "--name-only", "--diff-filter=ACMR", "-z", base_ref, "--"
    )
    return [value for value in out.split("\0") if value]


def _untracked_paths(repo: Path) -> list[str]:
    """Notes still untracked when this runs."""
    out = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    return [value for value in out.split("\0") if value]


def _copy_truncated(source: Path, destination: Path, limit: int) -> tuple[int, bool]:
    """Copy at most ``limit`` bytes. Returns (bytes_written, truncated)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    truncated = False
    with source.open("rb") as src, destination.open("wb") as dst:
        while written < limit:
            chunk = src.read(min(64 * 1024, limit - written))
            if not chunk:
                break
            dst.write(chunk)
            written += len(chunk)
        truncated = bool(src.read(1))
    return written, truncated


def collect_agent_notes(
    repo: Path,
    dest_dir: Path,
    *,
    base_ref: str = _BASE_REF,
) -> dict:
    """Copy the candidate's markdown notes from ``repo`` into ``dest_dir``."""
    repo = Path(repo)
    dest_dir = Path(dest_dir)
    record: dict = {
        "dest": dest_dir.name,
        "copied": [],
        "copied_count": 0,
        "skipped": [],
        "skipped_count": 0,
        "total_bytes": 0,
    }

    if not (repo / ".git").exists():
        record["error"] = f"not a git repo: {repo}"
        return record

    try:
        candidates = sorted(
            {
                value
                for value in (*_changed_paths(repo, base_ref), *_untracked_paths(repo))
                if _is_note_path(value)
            }
        )
    except _GitError as exc:
        record["error"] = str(exc)
        return record

    def skip(path: str, reason: str) -> None:
        record["skipped_count"] += 1
        if len(record["skipped"]) < RECORD_MAX_PATHS:
            record["skipped"].append({"path": path, "reason": reason})

    for rel in candidates:
        source = repo / rel
        if record["copied_count"] >= MAX_FILES:
            skip(rel, "max_files")
            continue
        if record["total_bytes"] >= MAX_TOTAL_BYTES:
            skip(rel, "max_total_bytes")
            continue
        try:
            if source.is_symlink() or not source.is_file():
                skip(rel, "not_a_regular_file")
                continue
            limit = min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - record["total_bytes"])
            written, truncated = _copy_truncated(source, dest_dir / rel, limit)
        except OSError as exc:
            skip(rel, f"io_error: {exc}")
            continue
        record["copied_count"] += 1
        record["total_bytes"] += written
        if len(record["copied"]) < RECORD_MAX_PATHS:
            entry: dict = {"path": rel, "bytes": written}
            if truncated:
                entry["truncated"] = True
            record["copied"].append(entry)

    return record
