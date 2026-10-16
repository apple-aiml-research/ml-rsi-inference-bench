#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Build the reference tree into the verifier image: the tests/Dockerfile tail step.

    RUN python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json

At verifier-image build time `/app` is the pristine task image -- no agent has touched it, and no
`model.patch` has been applied -- which is why the reference is snapshotted here and not at grading
time, when a patch could already have force-added files at ignored paths. The form below is used
because this reference changes no native code:

  1. Independent Git copy of `/app` into `<root>/<tree_dir>` and a detached checkout of the prepared
     base: `/app/.git` itself is not modified (its `git status` runs with `--no-optional-locks`).
  2. `tests/reference/solution.patch` -- byte-identical to `solution/solution.patch` -- checked
     against the digest in `config.json`, then applied.
  3. **The identity guard, on trees**: the result, staged, must hash to the closure tree that
     `solution/oracle.lock` records.
  4. **The native-source guard**: the patch changes nothing under `csrc/` or `rust/`, no
     `CMakeLists.txt`, nothing under `cmake/` but the landing's one `GIT_TAG` line, and nothing under
     the vendored fork outside `flash_attn/cute/*.py` and `tests/`. The prepared base's compiled
     products therefore serve the reference unchanged; a violation fails the image build.
  5. The engine's git-ignored build products under the declared roots (compiled extensions, the Rust
     frontend, generated `_version.py` -- never `__pycache__`, never the CuTe-DSL link) are copied
     from the pristine `/app` at the same paths, each with its sha256.
  6. `vllm/vllm_flash_attn/cute` is re-created as a symlink into the copy's own
     `third_party/vllm-flash-attn/flash_attn/cute`. A copied link would point at the candidate's
     `/app` tree, and copied files would be the base's fork, not the reference's.
  7. The tree's Python is byte-compiled once, here, by the interpreter that serves it.
  8. `<root>/snapshot.json` records all of it; `reference.py` checks it again at grading time.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
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
    """Every git-ignored file under `roots` in the pristine tree, minus bytecode, links and the excludes."""
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(app), "status", "--porcelain", "--ignored",
                          "--untracked-files=all", "-z", "--", *roots], capture_output=True, text=True, check=True).stdout
    paths = []
    for entry in out.split("\0"):
        if not entry.startswith("!! "):
            continue
        rel = entry[3:].rstrip("/")
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude):
            continue
        full = app / rel
        if full.is_symlink():
            continue
        files = [full] if full.is_file() else sorted(p for p in full.rglob("*") if p.is_file()) if full.is_dir() else []
        for f in files:
            r = str(f.relative_to(app))
            if "__pycache__" in f.parts or r.endswith((".pyc", ".pyo")) or f.is_symlink():
                continue
            if any(fnmatch.fnmatch(r, pat) for pat in exclude):
                continue
            paths.append(r)
    return sorted(set(paths))


def native_source_violations(tree: Path, base: str, rule: dict) -> list[str]:
    """Paths (and cmake lines) the patch changes that the prepared base's compiled products would not serve."""
    names = git("-C", str(tree), "diff", "--cached", "--no-ext-diff", "--no-renames", "--name-only", "-z", base)
    violations = []
    vendored = rule["vendored_root"].rstrip("/") + "/"
    for path in [p for p in names.split("\0") if p]:
        if any(path.startswith(pre) for pre in rule["forbidden_prefixes"]):
            violations.append(path)
        elif os.path.basename(path) in rule["forbidden_basenames"]:
            violations.append(path)
        elif path.startswith(vendored):
            inner = path[len(vendored):]
            if not any(fnmatch.fnmatch(inner, pat) for pat in rule["vendored_allowed"]):
                violations.append(path)
        elif path.startswith(rule["cmake_prefix"]):
            allowed = rule["cmake_allowed_lines"].get(path)
            if allowed is None:
                violations.append(path)
                continue
            diff = git("-C", str(tree), "diff", "--cached", "--no-ext-diff", "-U0", base, "--", path)
            for line in diff.splitlines():
                if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
                    continue
                if not re.fullmatch(allowed, line[1:]):
                    violations.append(f"{path}: {line[:120]}")
    return violations


def link_cute(tree: Path, links: dict[str, str]) -> list[dict]:
    """Re-create each ignored link inside the copy, pointing into the copy's own tracked source."""
    made = []
    for rel, target_rel in links.items():
        link, target = tree / rel, tree / target_rel
        if not target.is_dir():
            raise SystemExit(f"[snapshot] FATAL: link target {target} is not a directory in the reference tree")
        if link.exists() or link.is_symlink():
            raise SystemExit(f"[snapshot] FATAL: {link} exists before the link is made")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        real = link.resolve()
        if real != target.resolve() or not str(real).startswith(str(tree.resolve()) + os.sep):
            raise SystemExit(f"[snapshot] FATAL: {link} resolves to {real}, outside the reference tree")
        made.append({"link": rel, "target": str(target), "realpath": str(real)})
    return made


def compile_tree(tree: Path, roots: list[str]) -> dict:
    t0 = time.time()
    dirs = [str(tree / r) for r in roots if (tree / r).is_dir()]
    p = subprocess.run([sys.executable, "-m", "compileall", "-q", "-j", "0", *dirs], capture_output=True, text=True)
    # A file that does not compile (a template, a fixture) is recorded, not fatal: it only means that
    # file is compiled, or fails, when imported, exactly as in the candidate's tree.
    return {"roots": roots, "rc": p.returncode, "seconds": round(time.time() - t0, 3),
            "output": (p.stdout + p.stderr)[-2000:] if p.returncode else ""}


def refuse_existing_tree(tree: Path) -> None:
    """The reference tree is built once, from the pristine image: a directory removed and re-created
    inside one build step can come back with its lower layer's files on some builders."""
    if tree.exists():
        raise SystemExit(f"[snapshot] FATAL: {tree} already exists; the reference tree is built once, from the pristine task "
                         f"image, and is never rebuilt in place.")


def build(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    ref = cfg["infb"]["efficiency"]["reference"]
    app = Path(ref.get("app", "/app"))
    root = Path(ref["root"])
    tree = root / ref.get("tree_dir", "tree")
    patch = Path(ref["patch"])
    t0 = time.time()
    got = sha256_file(patch)
    if got != ref["patch_sha256"]:
        raise SystemExit(f"[snapshot] FATAL: {patch} has sha256 {got}, config.json names {ref['patch_sha256']}")
    if git("--no-optional-locks", "-C", str(app), "status", "--porcelain"):
        raise SystemExit(f"[snapshot] FATAL: {app} is not clean; the reference must come from the pristine image")
    refuse_existing_tree(tree)
    root.mkdir(parents=True, exist_ok=True)
    git("clone", "--quiet", "--no-hardlinks", "--no-checkout", str(app), str(tree))
    if (tree / ".git/objects/info/alternates").exists():
        raise SystemExit("[snapshot] FATAL: reference objects must not depend on the candidate repository")
    git("-C", str(tree), "-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", ref["base"])
    if git("-C", str(tree), "rev-parse", "HEAD") != ref["base"]:
        raise SystemExit(f"[snapshot] FATAL: the reference tree is not at the base {ref['base']}")
    git("-C", str(tree), "apply", "--index", "--binary", "--whitespace=nowarn", str(patch))
    tree_hash = git("-C", str(tree), "write-tree")
    if tree_hash != ref["target_tree"]:
        raise SystemExit(f"[snapshot] FATAL: base + the reference patch hashes to {tree_hash}, not the closure tree "
                         f"{ref['target_tree']} that oracle.lock records")
    violations = native_source_violations(tree, ref["base"], ref["native_source_rule"])
    if violations:
        raise SystemExit("[snapshot] FATAL: the reference patch changes native source, which the prepared base's compiled "
                         "products would not serve:\n  " + "\n  ".join(violations))
    products = []
    for rel in ignored_products(app, ref.get("build_product_roots", []), ref.get("build_product_exclude", [])):
        src, dst = app / rel, tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        products.append({"path": rel, "bytes": dst.stat().st_size, "sha256": sha256_file(dst)})
    links = link_cute(tree, ref.get("links", {}))
    if subprocess.run(["git", "-C", str(tree), "diff", "--quiet"]).returncode != 0:
        raise SystemExit("[snapshot] FATAL: copying the build products changed a tracked file")
    compiled = compile_tree(tree, ref.get("pythonpath", ["python"])) if ref.get("compile", True) else None
    snap = {"schema": "infb-reference-snapshot/1", "base": ref["base"], "patch": str(patch),
            "patch_sha256": got, "tree": tree_hash, "app": str(app), "products": products, "links": links,
            "compiled": compiled, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.time() - t0, 3)}
    (root / "snapshot.json").write_text(json.dumps(snap, indent=1))
    print(f"[snapshot] reference tree {tree_hash} at {tree} (base {ref['base'][:12]} + {patch.name}, "
          f"{len(products)} build product(s), {len(links)} link(s)) in {snap['seconds']} s", flush=True)
    for prod in products:
        print(f"[snapshot]   product {prod['path']} {prod['bytes']} B {prod['sha256'][:16]}", flush=True)
    for link in links:
        print(f"[snapshot]   link {link['link']} -> {link['realpath']}", flush=True)
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
