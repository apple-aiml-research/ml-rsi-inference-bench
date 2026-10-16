#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The withheld-material audit of the sparse-indexer preparation, run by the image build at
the end of the preparation step.

    python3 -B audit_image.py [--root DIR ...] [--git-root DIR ...] [--layers DIR ...] [--out FILE]

It looks for the four withheld implementation files and their restoration, by content, in:

  * every git object store under the git roots (default /app, /opt, /usr/local, /root) -- the engine's repository,
    every dependency clone under .deps, every submodule store under a .git/modules directory, the build tools' own
    stores (cargo's git database and checkouts) -- reading EVERY object (`git cat-file --batch-all-objects`),
    reachable or not: a blob equal to a withheld file, or any object whose content carries one of the withheld
    implementations' distinctive symbols; and the required form of each store: no remote-tracking ref,
    no alternates, and, for every dependency store (every store but the engine's own repository, /app/.git), no
    commit beyond the one it has checked out;
  * every file under the scanned roots (source trees, installed packages, caches, the sparse runtime's staged include
    tree), by the four files' SHA256 and by those symbols;
  * optionally, image layer directories (`--layers`), the same file scan per layer.

The symbols are identifiers that occur only in the withheld implementations and in their callers in the unstubbed
attention.hpp (launch functions, kernel and scheduler names); the stubbed declarations, their registrations and the
API names are public and are not searched for. A hit is printed with its store/path and the process exits 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Identifiers that occur only in the withheld implementations (the four removed files and the four stubbed bodies
#: of attention.hpp) -- none in the sanitized tree or the engine's base: a hit is a finding.
SYMBOLS = [
    "launch_sm100_sparse_mqa_logits",
    "sm100_sparse_mqa_logits",
    "get_sparse_mqa_logits_workspace",
    "get_sparse_mqa_split_kv",
]
#: Names the withheld layout and scheduler also use, generic enough to occur in unrelated packages: reported for
#: review, never a build failure.
REVIEW = ["get_num_metadata_bytes", "QBlockInfo", "WorkspaceState"]
SYMBOL_RE = re.compile("|".join(re.escape(s) for s in sorted(SYMBOLS, key=len, reverse=True)).encode())
REVIEW_RE = re.compile("|".join(re.escape(s) for s in REVIEW).encode())
#: Files that may name the symbols because they ARE the audit or the record of what was withheld.
SELF = {"audit_image.py", "removal-manifest.json", "sources.json"}
SKIP_DIRS = {"/proc", "/sys", "/dev", "/run", "/tmp/infb-sparse-prep"}
MAX_FILE = 256 * 1024 * 1024


def withheld_hashes() -> tuple[set[str], set[str]]:
    """(sha256 of the four withheld files, their git blob ids), from the removal manifest's inputs."""
    manifest = json.loads((HERE / "removal-manifest.json").read_text())
    sources = json.loads((HERE / "sources.json").read_text())
    sha = {manifest["inputs"][r] for r in sources["removed"]}
    return sha, set(manifest["blob_ids"].values())


def git_stores(git_root: Path) -> list[Path]:
    stores = []
    for dirpath, dirnames, filenames in os.walk(git_root):
        p = Path(dirpath)
        if p.name == "objects" and (p.parent / "HEAD").is_file() and (p.parent / "refs").is_dir():
            stores.append(p.parent)
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not (p / d).is_symlink()]
    return sorted(set(stores))


def scan_store(store: Path, blob_ids: set[str], review: list[str]) -> list[str]:
    hits = []
    listing = subprocess.run(["git", "--git-dir", str(store), "cat-file", "--batch-all-objects", "--batch-check"],
                             capture_output=True, text=True)
    if listing.returncode != 0:
        return [f"{store}: unreadable store ({listing.stderr.strip()[:200]})"]
    ids = [ln.split()[0] for ln in listing.stdout.splitlines() if ln]
    for oid in ids:
        if oid in blob_ids:
            hits.append(f"{store}: object {oid} is a withheld file")
    proc = subprocess.Popen(["git", "--git-dir", str(store), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    for oid in ids:
        proc.stdin.write(oid.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        m = SYMBOL_RE.search(data)
        if m:
            hits.append(f"{store}: object {oid} ({header[1].decode()}) carries {m.group(0).decode()}")
        elif (r := REVIEW_RE.search(data)):
            review.append(f"{store}: object {oid} ({header[1].decode()}) names {r.group(0).decode()}")
    proc.stdin.close()
    proc.wait()
    return hits


def scan_files(root: Path, sha: set[str], review: list[str]) -> list[str]:
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(dirpath == s or dirpath.startswith(s + "/") for s in SKIP_DIRS):
            dirnames[:] = []
            continue
        for name in filenames:
            path = Path(dirpath) / name
            if name in SELF and path.parent.name in ("sparse-indexer", "infb-sparse-indexer"):
                continue
            try:
                if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if hashlib.sha256(data).hexdigest() in sha:
                hits.append(f"{path}: a withheld file")
            elif "/objects/" not in str(path):  # git objects are compressed; the store scan reads them
                m = SYMBOL_RE.search(data)
                if m:
                    hits.append(f"{path}: carries {m.group(0).decode()}")
                elif (r := REVIEW_RE.search(data)):
                    review.append(f"{path}: names {r.group(0).decode()}")
    return hits


#: The engine's own repository: its history is the upstream base's, which the task gives the agent; the
#: one-commit rule is for dependency stores.
ENGINE_STORE = Path("/app/.git")


def store_form(store: Path) -> list[str]:
    """The required form of one store: remote-tracking refs, alternates, and (a dependency store) more than one commit."""
    problems = []
    refs = subprocess.run(["git", "--git-dir", str(store), "for-each-ref", "--format=%(refname)"],
                          capture_output=True, text=True)
    remotes = [r for r in refs.stdout.split() if r.startswith("refs/remotes/")]
    if remotes:
        problems.append(f"{store}: {len(remotes)} remote-tracking ref(s), e.g. {remotes[0]}")
    alt = store / "objects" / "info" / "alternates"
    if alt.is_file() and alt.read_text().strip():
        problems.append(f"{store}: alternates ({alt.read_text().strip()[:120]})")
    if store.resolve() != ENGINE_STORE.resolve():
        n = subprocess.run(["git", "--git-dir", str(store), "rev-list", "--all", "--count"],
                           capture_output=True, text=True)
        if n.returncode == 0 and n.stdout.strip() and int(n.stdout.strip()) > 1:
            problems.append(f"{store}: a dependency store with {n.stdout.strip()} commits, not its one checked-out commit")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", action="append", type=Path, help="file-scan roots (default /app /opt /usr/local /root)")
    ap.add_argument("--git-root", action="append", type=Path,
                    help="store-scan roots (default /app /opt /usr/local /root)")
    ap.add_argument("--layers", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    sha, blob_ids = withheld_hashes()
    roots = a.root or [Path("/app"), Path("/opt"), Path("/usr/local"), Path("/root")]
    record = {"stores": [], "roots": [str(r) for r in roots], "layers": [str(x) for x in a.layers], "hits": [],
              "review": []}
    git_roots = a.git_root or [Path("/app"), Path("/opt"), Path("/usr/local"), Path("/root")]
    record["git_roots"] = [str(r) for r in git_roots]
    stores = sorted({s for r in git_roots if r.exists() for s in git_stores(r)})
    for store in stores:
        h = scan_store(store, blob_ids, record["review"]) + store_form(store)
        record["stores"].append({"store": str(store), "hits": len(h)})
        record["hits"] += h
    for root in roots:
        if root.exists():
            record["hits"] += scan_files(root, sha, record["review"])
    for layer in a.layers:
        record["hits"] += scan_files(layer, sha, record["review"])
    if a.out:
        a.out.write_text(json.dumps(record, indent=1) + "\n")
    for h in record["hits"]:
        print(f"[audit] HIT {h}")
    print(f"[audit] {len(record['stores'])} git stores read object by object; {len(record['hits'])} hit(s); "
          f"{len(record['review'])} name(s) for review")
    return 1 if record["hits"] else 0


if __name__ == "__main__":
    sys.exit(main())
