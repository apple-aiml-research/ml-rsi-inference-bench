#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prepare and verify the task's immutable model inputs before offline execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "model-assets.lock.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text())
    if lock.get("schema") != 1 or set(lock.get("models", {})) != {"sd3", "clip"}:
        raise ValueError("expected an SD3 and CLIP asset lock")
    for name, spec in lock["models"].items():
        if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
            raise ValueError(f"{name}: revision must be a full commit SHA")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", spec.get("repo_id", "")):
            raise ValueError(f"{name}: invalid public repository id")
        if not isinstance(spec.get("gated"), bool) or not spec.get("files"):
            raise ValueError(f"{name}: invalid model declaration")
        for filename, item in spec["files"].items():
            path = PurePosixPath(filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in filename
                    or str(path) != filename or not path.parts):
                raise ValueError(f"{name}: invalid file path")
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
                raise ValueError(f"{name}/{filename}: invalid SHA-256")
            if type(item.get("bytes")) is not int or item["bytes"] <= 0:
                raise ValueError(f"{name}/{filename}: invalid byte count")
    return lock


def check_file(path: Path, item: dict, cache: dict | None = None) -> None:
    if not path.is_file():
        raise ValueError(f"missing model file: {path}")
    stat = path.stat()
    if stat.st_size != item["bytes"]:
        raise ValueError(f"size mismatch: {path}")
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    digest = cache.get(key) if cache is not None else None
    if digest is None:
        digest = sha256(path)
        if cache is not None:
            cache[key] = digest
    if digest != item["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify(root: Path, lock: dict, lock_sha: str) -> dict:
    cache = {}
    result = {"ok": True, "lock_sha256": lock_sha, "models": {}}
    for name, spec in lock["models"].items():
        model_root = root / name
        for filename, item in spec["files"].items():
            check_file(model_root / filename, item, cache)
        # Extra weights can change loader preference even if every required file exists.
        for path in model_root.rglob("*"):
            if (path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
                    and path.relative_to(model_root).as_posix() not in spec["files"]):
                raise ValueError(f"unlisted model weights: {path}")
        result["models"][name] = {
            "repo_id": spec["repo_id"], "revision": spec["revision"],
            "files": len(spec["files"]),
            "bytes": sum(item["bytes"] for item in spec["files"].values()),
        }
    return result


def copy_verified(source: Path, destination: Path, item: dict, verified: dict) -> None:
    if destination.exists():
        check_file(destination, item)
        verified[item["sha256"]] = destination
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".asset-", dir=destination.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        previous = verified.get(item["sha256"])
        if previous is not None:
            temporary.unlink()
            os.link(previous, temporary)
        else:
            shutil.copyfile(source, temporary)
        check_file(temporary, item)
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        verified[item["sha256"]] = destination
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root: Path, lock: dict, sources: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    verified = {}
    if sources is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ValueError("fetch requires huggingface-hub==1.28.0; see the task README") from error
    for name, spec in lock["models"].items():
        for filename, item in spec["files"].items():
            destination = root / name / filename
            if destination.exists():
                check_file(destination, item)
                verified[item["sha256"]] = destination
                continue
            if item["sha256"] in verified:
                source = verified[item["sha256"]]
            elif sources is not None:
                source = sources[name] / filename
            else:
                # Public CLIP never consumes an ambient token. SD3 requires the user's
                # normal HF login and upstream model access, outside the offline trial.
                source = Path(hf_hub_download(
                    repo_id=spec["repo_id"], revision=spec["revision"],
                    filename=filename, token=True if spec["gated"] else False,
                ))
            copy_verified(source, destination, item, verified)


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "stage", "verify", "mounts"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--sd3-source", type=Path)
    parser.add_argument("--clip-source", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        root = args.root.expanduser().resolve()
        lock = load_lock(args.lock)
        if args.command == "fetch":
            prepare(root, lock)
        elif args.command == "stage":
            if args.sd3_source is None or args.clip_source is None:
                raise ValueError("stage requires both --sd3-source and --clip-source")
            prepare(root, lock, {"sd3": args.sd3_source, "clip": args.clip_source})
        receipt = verify(root, lock, sha256(args.lock))
        if args.receipt:
            write_receipt(args.receipt, receipt)
        if args.command == "mounts":
            output = [{"type": "bind", "source": str(root),
                       "target": "/opt/infb/models", "read_only": True}]
        else:
            output = receipt
        print(json.dumps(output, indent=2))
        return 0
    except Exception as error:
        # Hub exception text may contain signed URLs. Report its type, never credentials.
        message = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
        print(f"model assets not ready: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
