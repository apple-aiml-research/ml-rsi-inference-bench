#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Reduce every dependency Git store under the given roots to its checked-out commit.

A store is a Git directory: a clone's `.git`, a submodule store under `.git/modules` (nested ones included) or a
bare store such as Cargo's git database. Each store keeps its tip and what the tip reaches, nothing else:

  * the tip is HEAD's commit; a bare Cargo database, whose HEAD is unborn, has instead the locked revisions it
    names under refs/commit/ (Cargo's offline build looks the locked revision up there);
  * every remote-tracking ref, and every other ref not at the tip, is deleted. Tags and branches AT the tip stay:
    FetchContent's update step resolves a GIT_TAG against the local tag and needs no fetch when that tag is
    already checked out, which is what keeps a rebuild offline;
  * the store becomes shallow at the tip, its reflogs are expired, FETCH_HEAD, ORIG_HEAD and any alternates go,
    and `git gc --prune=now` drops every object the tip does not reach.

It runs in the SAME RUN as the fetch (the engine stage's externals step, which clones the four CMake externals and
their submodules into /opt/infb/externals), so no layer of the image ever holds more than the tips. /app's own
repository is not under these roots.

    reduce_git_stores.py <root> [<root> ...]
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


def git(store: Path, *args: str, stdin: str | None = None, check: bool = True) -> str:
    return subprocess.run(['git', f'--git-dir={store}', *args], input=stdin, capture_output=True, text=True,
                          check=check).stdout


def stores(root: Path):
    """Every Git directory under root: HEAD, objects/ and refs (loose or packed), found without entering objects/."""
    for dirpath, dirnames, filenames in os.walk(root):
        if 'HEAD' in filenames and 'objects' in dirnames and ('refs' in dirnames or 'packed-refs' in filenames):
            yield Path(dirpath)
        dirnames[:] = [d for d in dirnames if d != 'objects' and not os.path.islink(os.path.join(dirpath, d))]


def count(store: Path) -> int:
    return sum(1 for _ in git(store, 'cat-file', '--batch-all-objects', '--batch-check=%(objectname)').splitlines())


def reduce(store: Path) -> str:
    before = count(store)
    head = git(store, 'rev-parse', '--verify', '-q', 'HEAD^{commit}', check=False).strip()
    refs = [line.split(' ', 2) for line in
            git(store, 'for-each-ref', '--format=%(refname) %(objectname) %(*objectname)').splitlines()]
    tips = {head} if head else {oid for name, oid, _ in refs if name.startswith('refs/commit/')}
    if not tips:
        raise SystemExit(f'FATAL: {store}: no HEAD commit and no locked revision, so no tip to keep')
    drop = [name for name, oid, peeled in refs if name.startswith('refs/remotes/') or (peeled or oid) not in tips]
    if drop:
        git(store, 'update-ref', '--stdin', stdin=''.join(f'option no-deref\ndelete {name}\n' for name in drop))
    for name in ('FETCH_HEAD', 'ORIG_HEAD', 'objects/info/alternates', 'objects/info/http-alternates'):
        (store / name).unlink(missing_ok=True)
    (store / 'shallow').write_text(''.join(f'{tip}\n' for tip in sorted(tips)))
    git(store, 'reflog', 'expire', '--expire=now', '--all')
    shutil.rmtree(store / 'logs', ignore_errors=True)
    git(store, '-c', 'gc.auto=0', 'gc', '--prune=now', '--quiet')
    left = git(store, 'for-each-ref', '--format=%(refname)').splitlines()
    return (f'{store}: tip {",".join(t[:12] for t in sorted(tips))}; refs {len(refs)} -> {len(left)} '
            f'(deleted {len(drop)}); objects {before} -> {count(store)}')


def main() -> int:
    roots = [Path(arg) for arg in sys.argv[1:]]
    if not roots:
        raise SystemExit('usage: reduce_git_stores.py <root> [<root> ...]')
    found = [store for root in roots if root.is_dir() for store in stores(root)]
    for store in found:
        print('reduced', reduce(store), flush=True)
    print(f'reduced {len(found)} dependency Git stores under {", ".join(map(str, roots))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
