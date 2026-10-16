#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prepare the vendored FlashMLA and its CUTLASS for the task image.

Runs in the recipe's builder stage only, on git checkouts of the pinned FlashMLA and CUTLASS commits. It implements
exactly the transformations `kernel-materials.json` enumerates and nothing else, and it refuses unless every pinned
preimage matches:

  * FlashMLA at the pinned commit, exported from its Git objects (never its working tree), without the `csrc/cutlass`
    submodule and without `.git`;
  * `--variant supplied` (the agent's starting tree): the four optimized sparse subtrees and the two sparse API headers
    are left out, `csrc/api/api.cpp` registers the two sparse entry points as import-safe stubs that raise
    NotImplementedError, and `setup.py` lists only the remaining sources;
  * both variants: `setup.py` makes no network or git-submodule call and carries the fixed local version
    `1.0.0+9241ae3.day0` (the offline packaging adaptations; the reference keeps them);
  * `csrc/cutlass` becomes a symlink to the generic CUTLASS checkout, exported without history to `--out-cutlass`.

Every output is checked against the digests the manifest records: each transformed file's sha256, the Git tree hash
of the produced FlashMLA directory and of the exported CUTLASS. A mismatch is fatal, so the build either produces the
recorded material or stops. The reference variant exists for authoring the reference patch; the image build runs the
supplied variant. No other file is written, and nothing is fetched here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


def die(msg: str) -> "None":
    raise SystemExit(f"[prepare_kernel_materials] FATAL: {msg}")


def git(repo: Path, *args: str) -> bytes:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if p.returncode != 0:
        die(f"git {' '.join(args)} in {repo}: {p.stderr.decode(errors='replace').strip()[-500:]}")
    return p.stdout


def tree_entries(repo: Path, rev: str) -> list[tuple[str, str, str, str]]:
    """(mode, type, sha, path) for every entry of `rev`'s tree, recursively, gitlinks included."""
    out = []
    for rec in git(repo, "ls-tree", "-r", "-z", "--full-tree", rev).split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha = meta.decode().split()
        out.append((mode, typ, sha, path.decode()))
    return out


def blob(repo: Path, sha: str) -> bytes:
    return git(repo, "cat-file", "blob", sha)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_tree_sha(entries: dict[str, tuple[str, bytes]]) -> str:
    """The Git tree hash of a set of files: path -> (mode, content); mode 100644, 100755 or 120000 (content = target)."""
    root: dict = {}
    for path, (mode, data) in entries.items():
        node = root
        parts = path.split("/")
        for d in parts[:-1]:
            node = node.setdefault(d, {})
        node[parts[-1]] = (mode, data)

    def write(node: dict) -> str:
        items = []
        for name, val in node.items():
            if isinstance(val, dict):
                items.append((name + "/", b"40000", name, bytes.fromhex(write(val))))
            else:
                mode, data = val
                items.append((name, mode.encode(), name, bytes.fromhex(git_blob_sha(data))))
        items.sort(key=lambda t: t[0])
        body = b"".join(m + b" " + n.encode() + b"\0" + sha for _, m, n, sha in items)
        return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()

    return write(root)


def replace_exact(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        die(f"{what}: the anchor occurs {text.count(old)} times in the pinned preimage, expected once")
    return text.replace(old, new)


def transform_api_cpp(text: str, m: dict) -> str:
    t = m["transforms"]["csrc/api/api.cpp"]["supplied"]
    text = replace_exact(text, t["includes_old"], t["includes_new"], "api.cpp includes")
    for old, new in t["registrations"]:
        text = replace_exact(text, old, new, "api.cpp registration")
    return text


def transform_setup_py(text: str, m: dict, variant: str) -> str:
    t = m["transforms"]["setup.py"]
    for old, new in t["packaging"]:
        text = replace_exact(text, old, new, "setup.py packaging")
    if variant == "supplied":
        old, new = t["supplied"]["sources"]
        text = replace_exact(text, old, new, "setup.py source list")
    return text


def withheld_paths(m: dict, entries: list[tuple[str, str, str, str]]) -> dict[str, str]:
    """Every withheld file of FlashMLA with its pinned blob sha, the recorded subtrees checked whole."""
    by_path = {p: sha for _, _, sha, p in entries}
    out: dict[str, str] = {}
    for sub in m["flashmla"]["withheld_subtrees"]:
        files = {p: sha for p, sha in by_path.items() if p.startswith(sub["path"] + "/")}
        if sorted(files.items()) != sorted(sub["files"].items()):
            die(f"the pinned subtree {sub['path']} does not list the recorded files and blobs")
        out.update(files)
    for f in m["flashmla"]["withheld_files"]:
        if by_path.get(f["path"]) != f["blob"]:
            die(f"the pinned file {f['path']} is not the recorded blob {f['blob']}")
        out[f["path"]] = f["blob"]
    return out


def write_tree(dest: Path, files: dict[str, tuple[str, bytes]]) -> None:
    if dest.exists() and any(dest.iterdir()):
        die(f"{dest} exists and is not empty")
    for path, (mode, data) in sorted(files.items()):
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            os.symlink(data.decode(), target)
            continue
        target.write_bytes(data)
        os.chmod(target, 0o755 if mode == "100755" else 0o644)


def prepare(args: argparse.Namespace) -> dict:
    m = json.loads(Path(args.manifest).read_text())
    fm, cu = Path(args.flashmla), Path(args.cutlass)
    fcfg, ccfg = m["flashmla"], m["cutlass"]

    head = git(fm, "rev-parse", "HEAD").decode().strip()
    if head != fcfg["commit"] or git(fm, "rev-parse", "HEAD^{tree}").decode().strip() != fcfg["tree"]:
        die(f"FlashMLA checkout is at {head}, not the pinned {fcfg['commit']} (tree {fcfg['tree']})")
    entries = tree_entries(fm, "HEAD")
    gitlinks = [(p, sha) for mode, typ, sha, p in entries if typ == "commit"]
    if gitlinks != [(ccfg["submodule_path"], ccfg["commit"])]:
        die(f"FlashMLA's submodules are {gitlinks}, expected only {ccfg['submodule_path']} at {ccfg['commit']}")
    withheld = withheld_paths(m, entries) if args.variant == "supplied" else {}

    files: dict[str, tuple[str, bytes]] = {}
    for mode, typ, sha, path in entries:
        if typ != "blob" or path in withheld:
            continue
        data = blob(fm, sha)
        tf = m["transforms"].get(path)
        if tf is not None:
            if sha != tf["preimage_blob"]:
                die(f"{path} is blob {sha}, not the recorded preimage {tf['preimage_blob']}")
            text = data.decode("utf-8")
            if path == "csrc/api/api.cpp":
                text = transform_api_cpp(text, m) if args.variant == "supplied" else text
            elif path == "setup.py":
                text = transform_setup_py(text, m, args.variant)
            else:
                die(f"no transformation is implemented for {path}")
            data = text.encode("utf-8")
            want = tf["output_sha256"][args.variant]
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                die(f"the {args.variant} {path} is {got}, the manifest records {want}")
        files[path] = (mode, data)
    files[ccfg["submodule_path"]] = ("120000", ccfg["install_path"].encode())

    produced = git_tree_sha(files)
    want = fcfg["output_tree"][args.variant]
    if produced != want:
        die(f"the {args.variant} FlashMLA tree hashes to {produced}, the manifest records {want}")
    write_tree(Path(args.out_flashmla), files)

    record = {"schema": "infb-kernel-materials-record/1", "variant": args.variant,
              "flashmla": {"commit": head, "output_tree": produced, "files": len(files),
                           "withheld_files": len(withheld), "transformed": sorted(m["transforms"])}}
    if args.out_cutlass:
        chead = git(cu, "rev-parse", "HEAD").decode().strip()
        if chead != ccfg["commit"] or git(cu, "rev-parse", "HEAD^{tree}").decode().strip() != ccfg["tree"]:
            die(f"CUTLASS checkout is at {chead}, not the pinned {ccfg['commit']} (tree {ccfg['tree']})")
        cfiles: dict[str, tuple[str, bytes]] = {}
        for mode, typ, sha, path in tree_entries(cu, "HEAD"):
            if typ != "blob":
                die(f"CUTLASS entry {path} is a {typ}; the pinned tree has no submodules")
            cfiles[path] = (mode, blob(cu, sha))
        ctree = git_tree_sha(cfiles)
        if ctree != ccfg["tree"]:
            die(f"the exported CUTLASS hashes to {ctree}, not its pinned tree {ccfg['tree']}")
        write_tree(Path(args.out_cutlass), cfiles)
        record["cutlass"] = {"commit": chead, "tree": ctree, "files": len(cfiles), "path": args.out_cutlass}
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--flashmla", required=True, help="git checkout of FlashMLA at the pinned commit")
    ap.add_argument("--cutlass", required=True, help="git checkout of CUTLASS at the pinned commit")
    ap.add_argument("--variant", choices=("supplied", "reference"), required=True)
    ap.add_argument("--out-flashmla", required=True)
    ap.add_argument("--out-cutlass", default=None)
    ap.add_argument("--record", default=None, help="write the record of what was produced here")
    args = ap.parse_args(argv)
    record = prepare(args)
    text = json.dumps(record, indent=1, sort_keys=True)
    if args.record:
        Path(args.record).write_text(text + "\n")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
