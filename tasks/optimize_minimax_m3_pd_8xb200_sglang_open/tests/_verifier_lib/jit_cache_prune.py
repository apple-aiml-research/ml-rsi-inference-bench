#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Count the JIT caches under the collected directories into one record, then remove them."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

SCHEMA = "infb-jit-cache-summary-v1"
# _set_jit_cache_env names every cache it hands out jit_cache_<side>.
CACHE_DIR_PREFIX = "jit_cache_"
MAX_ENTRIES = 200


def find_cache_dirs(root: Path) -> list[Path]:
    """Real directories named jit_cache_* under root. A cache is not searched for further caches."""
    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=False):
        kept = []
        for name in sorted(dirnames):
            path = Path(dirpath) / name
            if not name.startswith(CACHE_DIR_PREFIX):
                kept.append(name)
            elif not path.is_symlink():
                found.append(path)
        dirnames[:] = kept
    return found


def _count(tree: Path) -> tuple[int, int, int]:
    files = size = cubins = 0
    for dirpath, _, filenames in os.walk(tree, followlinks=False):
        for name in filenames:
            try:
                size += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
            files += 1
            cubins += name.endswith(".cubin")
    return files, size, cubins


def measure(cache: Path) -> dict:
    """What one cache holds: totals, Triton's share, and the file count of each top-level entry."""
    files, size, _ = _count(cache)
    triton = cache / ".triton" / "cache"
    try:
        triton_entries = sum(1 for p in triton.iterdir() if p.is_dir())
    except OSError:
        triton_entries = 0
    entries: dict[str, int] = {}
    try:
        names = sorted(p.name for p in cache.iterdir())
    except OSError:
        names = []
    for name in names[:MAX_ENTRIES]:
        child = cache / name
        entries[name] = _count(child)[0] if child.is_dir() and not child.is_symlink() else 1
    return {
        "path": str(cache),
        "files": files,
        "bytes": size,
        "triton_cache_entries": triton_entries,
        "triton_cubins": _count(cache / ".triton")[2],
        "entries": entries,
        "entries_omitted": max(0, len(names) - MAX_ENTRIES),
        "removed": False,
    }


def _write(record_path: Path, record: dict) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = record_path.with_name(record_path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, record_path)


def prune(roots: list[Path], record_path: Path) -> dict:
    """Count first, write the record, then remove. Never raises: a failure is a field of the record."""
    started = time.monotonic()
    record: dict = {"schema": SCHEMA, "roots": [str(r) for r in roots], "caches": [], "errors": []}
    try:
        for root in roots:
            if root.is_dir():
                record["caches"] += [measure(c) for c in find_cache_dirs(root)]
        record["files"] = sum(c["files"] for c in record["caches"])
        record["bytes"] = sum(c["bytes"] for c in record["caches"])
        _write(record_path, record)
        for cache in record["caches"]:
            try:
                shutil.rmtree(cache["path"])
                cache["removed"] = True
            except OSError as exc:
                cache["error"] = f"{type(exc).__name__}: {exc}"
        record["files_left"] = sum(
            _count(Path(c["path"]))[0] for c in record["caches"] if not c["removed"])
    except Exception as exc:  # noqa: BLE001
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    record["seconds"] = round(time.monotonic() - started, 3)
    try:
        _write(record_path, record)
    except OSError as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args(argv)
    record = prune(args.roots, args.record)
    removed = sum(1 for c in record["caches"] if c["removed"])
    print(f"[jit-cache] {record.get('files', 0)} files, {record.get('bytes', 0)} bytes in "
          f"{len(record['caches'])} cache(s); removed {removed}; record: {args.record}", flush=True)
    for error in record["errors"] + [c["error"] for c in record["caches"] if "error" in c]:
        print(f"[jit-cache] {error}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
