#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Build the reference tree into the verifier image: the tests/Dockerfile tail step.

    RUN python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json

At verifier-image build time `/app` is the pristine task image -- no agent has touched it, and no
`model.patch` has been applied -- which is why the reference is snapshotted here and not at grading
time, when a patch could already have force-added files at ignored paths.

  1. `git clone --shared` of `/app` into `<root>/tree` and a detached checkout of the base (the
     task's base, the prepared one where the task has one): `/app/.git` itself is not modified (its
     `git status` runs with `--no-optional-locks`, so not even the index's stat cache is refreshed).
  2. `tests/reference/solution.patch` -- byte-identical to `solution/solution.patch`, a repository
     guard holds that -- checked against the digest in `config.json`, then applied.
  3. **The identity guard, on trees**: the result, staged, must hash to the closure tree that
     `solution/oracle.lock` records. Anything else fails the image build.
  4. The engine's git-ignored build products under the task's declared roots (compiled extensions,
     generated `_version.py`, vendored stubs -- never `__pycache__`) are copied from the pristine
     `/app` at the same paths, each with its sha256, so the reference imports what the candidate's
     install has; a partial copy would bias the ratio.
  5. The tree's Python is byte-compiled once, here, by the interpreter that serves it, so serving
     writes nothing into it (a first serve was measured writing 1,250 `.pyc` files into an
     uncompiled tree). `__pycache__` is git-ignored, so the identity checks do not see it.
  6. `<root>/snapshot.json` records all of it; `reference.py` checks it again at grading time.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str, cwd: Path | None = None, capture: bool = True) -> str:
    p = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=capture, text=True)
    if p.returncode != 0:
        raise SystemExit(f"[snapshot] FATAL: git {' '.join(args)} failed ({p.returncode}): {p.stderr.strip()[-2000:]}")
    return (p.stdout or "").strip()


def ignored_products(app: Path, roots: list[str], exclude: list[str]) -> list[str]:
    """Every git-ignored file under `roots` in the pristine tree, minus bytecode and the excludes."""
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(app), "status", "--porcelain", "--ignored",
                          "--untracked-files=all", "-z", "--", *roots], capture_output=True, text=True, check=True).stdout
    paths = []
    for entry in out.split("\0"):
        if not entry.startswith("!! "):
            continue
        rel = entry[3:]
        full = app / rel
        files = [full] if full.is_file() else sorted(p for p in full.rglob("*") if p.is_file()) if full.is_dir() else []
        for f in files:
            r = str(f.relative_to(app))
            if "__pycache__" in f.parts or r.endswith((".pyc", ".pyo")):
                continue
            if any(fnmatch.fnmatch(r, pat) for pat in exclude):
                continue
            paths.append(r)
    return sorted(set(paths))


def compile_tree(tree: Path, roots: list[str]) -> dict:
    t0 = time.time()
    dirs = [str(tree / r) for r in roots if (tree / r).is_dir()]
    p = subprocess.run([sys.executable, "-m", "compileall", "-q", "-j", "0", *dirs], capture_output=True, text=True)
    # A file that does not compile (a template, a fixture) is recorded, not fatal: it only means that
    # file is compiled, or fails, when imported, exactly as in the candidate's tree.
    return {"roots": roots, "rc": p.returncode, "seconds": round(time.time() - t0, 3),
            "output": (p.stdout + p.stderr)[-2000:] if p.returncode else ""}


def refuse_existing_tree(tree: Path) -> None:
    """The reference tree is built once, from the pristine image. An earlier revision removed an existing tree and
    cloned it again in the same build step; on some container builders a directory removed and re-created inside one
    RUN comes back with its lower layer's files, so a stale tree could show through under a fresh clone. No route
    builds over an existing tree, so an existing one is a fault, said out loud."""
    if tree.exists():
        raise SystemExit(f"[snapshot] FATAL: {tree} already exists; the reference tree is built once, from the pristine task "
                         f"image, and is never rebuilt in place (a directory removed and re-created in one build step keeps its "
                         f"lower layer's files on some builders). Remove it in a step of its own, or build from the pristine image.")


def build(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    ref = cfg["infb"]["efficiency"]["reference"]
    app = Path(ref.get("app", "/app"))
    root = Path(ref["root"])
    tree = root / "tree"
    patch = Path(ref["patch"])
    t0 = time.time()
    got = sha256_file(patch)
    if got != ref["patch_sha256"]:
        raise SystemExit(f"[snapshot] FATAL: {patch} has sha256 {got}, config.json names {ref['patch_sha256']}")
    if git("--no-optional-locks", "-C", str(app), "status", "--porcelain"):
        raise SystemExit(f"[snapshot] FATAL: {app} is not clean; the reference must come from the pristine image")
    refuse_existing_tree(tree)
    root.mkdir(parents=True, exist_ok=True)
    git("clone", "--quiet", "--shared", "--no-checkout", str(app), str(tree))
    git("-C", str(tree), "-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", ref["base"])
    if git("-C", str(tree), "rev-parse", "HEAD") != ref["base"]:
        raise SystemExit(f"[snapshot] FATAL: the reference tree is not at the base {ref['base']}")
    git("-C", str(tree), "apply", "--binary", "--whitespace=nowarn", str(patch))
    changed = git("-C", str(tree), "apply", "--numstat", str(patch))
    paths = [line.split("\t", 2)[2] for line in changed.splitlines()]
    git("-C", str(tree), "add", "--", *paths)
    tree_hash = git("-C", str(tree), "write-tree")
    if tree_hash != ref["target_tree"]:
        raise SystemExit(f"[snapshot] FATAL: base + the reference patch hashes to {tree_hash}, not the closure tree "
                         f"{ref['target_tree']} that oracle.lock records")
    products = []
    for rel in ignored_products(app, ref.get("build_product_roots", []), ref.get("build_product_exclude", [])):
        src, dst = app / rel, tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        products.append({"path": rel, "bytes": dst.stat().st_size, "sha256": sha256_file(dst)})
    if subprocess.run(["git", "-C", str(tree), "diff", "--quiet"]).returncode != 0:
        raise SystemExit("[snapshot] FATAL: copying the build products changed a tracked file")
    compiled = compile_tree(tree, ref.get("pythonpath", ["python"])) if ref.get("compile", True) else None
    snap = {"schema": "infb-reference-snapshot/1", "base": ref["base"], "patch": str(patch),
            "patch_sha256": got, "tree": tree_hash, "app": str(app), "products": products, "compiled": compiled,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 3)}
    (root / "snapshot.json").write_text(json.dumps(snap, indent=1))
    print(f"[snapshot] reference tree {tree_hash} at {tree} (base {ref['base'][:12]} + {patch.name}, "
          f"{len(products)} build product(s)) in {snap['seconds']} s", flush=True)
    for prod in products:
        print(f"[snapshot]   product {prod['path']} {prod['bytes']} B {prod['sha256'][:16]}", flush=True)
    if compiled:
        print(f"[snapshot]   byte-compiled {compiled['roots']} in {compiled['seconds']} s (rc {compiled['rc']})", flush=True)
    return snap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="/tests/config.json")
    args = ap.parse_args(argv)
    build(Path(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
