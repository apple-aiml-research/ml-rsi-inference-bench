#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The trusted restoration of the paths the verifier reads from the candidate's tree, before any import from them."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP = Path(os.environ.get("APP_DIR", "/app"))
GIT_ENV = {"GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}


class RestoreFault(Exception):
    """A restored path is not the pinned tree: a verifier fault, never a candidate verdict."""


def git(*args: str, stdin: bytes | None = None) -> bytes:
    r = subprocess.run(["git", "-c", "core.useReplaceRefs=false", "-C", str(APP), *args],
                       env={**os.environ, **GIT_ENV}, input=stdin, capture_output=True)
    if r.returncode != 0:
        raise RestoreFault(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()[:500]}")
    return r.stdout


def remove_as_path(rel: str) -> None:
    """Unlink a symlink, or a non-directory where a directory belongs, on the way down to `rel`; then remove `rel`."""
    cur = APP
    parts = Path(rel).parts
    for i, part in enumerate(parts):
        cur = cur / part
        if cur.is_symlink() or (i < len(parts) - 1 and cur.exists() and not cur.is_dir()):
            cur.unlink()
            return
        if not cur.exists():
            return
    if cur.is_dir():
        shutil.rmtree(cur)
    else:
        cur.unlink()


def tree_entries(base: str, rel: str) -> list[tuple[str, str, str, str]]:
    """(mode, type, object id, path) of every entry under `rel` in `base`'s tree, recursively."""
    out = []
    for record in git("ls-tree", "-r", "-z", "--full-tree", base, "--", rel).split(b"\0"):
        if record:
            meta, path = record.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            out.append((mode, kind, oid, os.fsdecode(path)))
    return out


def blobs(oids: list[str]) -> dict[str, bytes]:
    """The raw bytes of each blob, as the object store holds them (`cat-file --batch`: no filters)."""
    raw = git("cat-file", "--batch", stdin="".join(f"{oid}\n" for oid in oids).encode())
    out, at = {}, 0
    for oid in oids:
        end = raw.find(b"\n", at)
        header = raw[at:end].decode(errors="replace").split() if end >= 0 else []
        if len(header) != 3 or header[0] != oid or header[1] != "blob":
            raise RestoreFault(f"object {oid} read back as {' '.join(header) or 'nothing'}")
        size = header[2]
        start = end + 1
        out[oid] = raw[start:start + int(size)]
        at = start + int(size) + 1
    return out


def blob_id(data: bytes, algo: str) -> str:
    h = hashlib.new(algo)
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def restore(base: str, rel: str, pinned: str, algo: str) -> int:
    """Restore `rel` from the pinned tree; return the number of files written and verified."""
    tree = git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}:{rel}").decode().strip()
    if tree != pinned:
        raise RestoreFault(f"{rel} is tree {tree or 'missing'} in {base[:10]}, not the pinned {pinned}")
    entries = tree_entries(base, rel)
    odd = [(mode, path) for mode, kind, _, path in entries if kind != "blob"]
    if odd:
        raise RestoreFault(f"{rel} holds entries that are not files: {odd[:3]}")
    data = blobs(sorted({oid for _, _, oid, _ in entries}))
    remove_as_path(rel)
    for mode, _, oid, path in entries:
        dst = APP / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            os.symlink(data[oid], dst)
        else:
            dst.write_bytes(data[oid])
            os.chmod(dst, 0o755 if mode == "100755" else 0o644)
    want = {path: (mode, oid) for mode, _, oid, path in entries}
    seen = {}
    for root, dirs, files in os.walk(APP / rel, followlinks=False):
        for name in files + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            p = Path(root) / name
            path = str(p.relative_to(APP))
            if p.is_symlink():
                seen[path] = ("120000", blob_id(os.fsencode(os.readlink(p)), algo))
            else:
                seen[path] = ("100755" if os.access(p, os.X_OK) else "100644", blob_id(p.read_bytes(), algo))
    if seen != want:
        extra = sorted(set(seen) - set(want))[:3]
        differ = sorted(k for k in set(seen) & set(want) if seen[k] != want[k])[:3]
        missing = sorted(set(want) - set(seen))[:3]
        raise RestoreFault(f"{rel} on disk differs from the pinned tree: extra {extra}, differing {differ}, "
                           f"missing {missing}")
    return len(entries)


def main(argv: list[str]) -> int:
    cfg = json.loads(Path(argv[0]).read_text())
    base = cfg["base_commit"]
    algo = git("rev-parse", "--show-object-format").decode().strip() or "sha1"
    try:
        for rel, pinned in cfg["infb"]["restore"]["trees"].items():
            n = restore(base, rel, pinned, algo)
            print(f"[restore] {rel}: removed as a path, {n} files written from {base[:10]}'s tree {pinned[:12]} "
                  f"and read back equal", flush=True)
    except RestoreFault as exc:
        print(f"[restore] FATAL: {exc}", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
