#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Build the reference tree into the verifier image: the tests/Dockerfile tail step.

    RUN chmod +x /tests/test.sh && test -s /tests/hidden/<each> ... && python3 /tests/infb_efficiency/snapshot.py --config /tests/config.json

At verifier-image build time `/app` is the pristine task image -- no agent has touched it, no `model.patch` has been
applied -- so the reference is built here, never at grading time.

  1. Independent Git copy of `/app` into `<root>/tree` at the prepared base (`/app/.git` is not modified).
  2. `tests/reference/solution.patch` -- byte-identical to `solution/solution.patch` -- checked against the digest in
     `config.json`, then applied.
  3. **The identity guard, on trees**: the result, staged, must hash to the closure tree `config.json` records (the
     `oracle.lock` closure tree plus the evaluation commit's `eval/`). Anything else fails the image build.
  4. The base's fetched external sources (`/app/.deps/*-src`) are copied into the tree's `.deps`, so CMake reuses
     them offline (fully disconnected; the base's sub-build directories record `/app` paths and are not copied).
  5. The documented editable build (`/public/build.sh`: offline, `--no-index`, the recipe's build variables) runs in
     the tree. The editable install re-points the venv's `vllm` at the tree, so the venv's own install of `/app` (its
     dist-info, editable hook files and console script) is saved first and restored after; `import vllm` must then
     resolve to `/app` again. The reference imports its engine through `PYTHONPATH` and the isolation hook instead.
  6. The tree's Python is byte-compiled once by the interpreter that serves it, so serving writes nothing into it.
  7. `<root>/snapshot.json` records every git-ignored build product under the tree's `vllm/` with its sha256, the
     build's wall time (it sets the verifier-image build allowance), and the checks; `reference.py` checks
     the tree and the products against it at grading time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VENV = Path(os.environ.get("INFB_VENV", "/opt/vllm"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fatal(msg: str) -> None:
    raise SystemExit(f"[snapshot] FATAL: {msg}")


def git(*args: str, cwd: Path | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if p.returncode != 0:
        fatal(f"git {' '.join(args)} failed ({p.returncode}): {p.stderr.strip()[-2000:]}")
    return (p.stdout or "").strip()


def site_packages() -> Path:
    out = subprocess.run([str(VENV / "bin/python"), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return Path(out)


def install_files(site: Path) -> list[Path]:
    """The venv's current install of `vllm`: its dist-info, what its RECORD lists outside the package, the console
    script."""
    dist = sorted(site.glob("vllm-*.dist-info"))
    if len(dist) != 1:
        fatal(f"expected one vllm dist-info in {site}, found {[d.name for d in dist]}")
    files = [dist[0]]
    record = dist[0] / "RECORD"
    for line in record.read_text().splitlines():
        rel = line.split(",", 1)[0]
        path = (site / rel).resolve()
        if path.exists() and not str(path).startswith(str((site / "vllm").resolve())) and dist[0] not in path.parents:
            files.append(path)
    return files


def save_install(site: Path, stash: Path) -> list[dict]:
    stash.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, path in enumerate(install_files(site)):
        dst = stash / f"{i:03d}-{path.name}"
        if path.is_dir():
            shutil.copytree(path, dst, symlinks=True)
        else:
            shutil.copy2(path, dst, follow_symlinks=False)
        saved.append({"path": str(path), "stash": str(dst), "dir": path.is_dir()})
    return saved


def restore_install(site: Path, saved: list[dict]) -> None:
    for path in install_files(site):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for row in saved:
        dst = Path(row["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if row["dir"]:
            shutil.copytree(row["stash"], dst, symlinks=True)
        else:
            shutil.copy2(row["stash"], dst, follow_symlinks=False)


def engine_file(env_extra: dict | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(env_extra or {})
    return subprocess.run([str(VENV / "bin/python"), "-c", "import os, vllm; print(os.path.realpath(vllm.__file__))"],
                          capture_output=True, text=True, env=env, cwd="/", check=True).stdout.strip()


def ignored_products(tree: Path, roots: list[str]) -> list[str]:
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(tree), "status", "--porcelain", "--ignored",
                          "--untracked-files=all", "-z", "--", *roots], capture_output=True, text=True,
                         check=True).stdout
    paths = []
    for entry in out.split("\0"):
        if not entry.startswith("!! "):
            continue
        full = tree / entry[3:]
        files = [full] if full.is_file() else sorted(p for p in full.rglob("*") if p.is_file()) if full.is_dir() else []
        for f in files:
            rel = str(f.relative_to(tree))
            if "__pycache__" in f.parts or rel.endswith((".pyc", ".pyo")):
                continue
            paths.append(rel)
    return sorted(set(paths))


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
        fatal(f"{patch} has sha256 {got}, config.json names {ref['patch_sha256']}")
    if git("--no-optional-locks", "-C", str(app), "status", "--porcelain"):
        fatal(f"{app} is not clean; the reference must come from the pristine image")
    if tree.exists():
        fatal(f"{tree} already exists; the reference tree is built once, from the pristine task image")
    root.mkdir(parents=True, exist_ok=True)
    git("clone", "--quiet", "--no-hardlinks", "--no-checkout", str(app), str(tree))
    if (tree / ".git/objects/info/alternates").exists():
        fatal("reference objects must not depend on the candidate repository")
    git("-C", str(tree), "-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", ref["base"])
    if git("-C", str(tree), "rev-parse", "HEAD") != ref["base"]:
        fatal(f"the reference tree is not at the base {ref['base']}")
    git("-C", str(tree), "apply", "--index", "--binary", "--whitespace=nowarn", str(patch))
    tree_hash = git("-C", str(tree), "write-tree")
    if tree_hash != ref["target_tree"]:
        fatal(f"base + the reference patch hashes to {tree_hash}, not the closure tree {ref['target_tree']}")
    t_copy = time.time()
    deps = tree / ".deps"
    deps.mkdir()
    fetched = sorted(p.name for p in (app / ".deps").glob("*-src") if p.is_dir())
    for name in fetched:
        shutil.copytree(app / ".deps" / name, deps / name, symlinks=True)
    deps_s = time.time() - t_copy
    site = site_packages()
    saved = save_install(site, root / "venv-install-of-app")
    t_build = time.time()
    env = dict(os.environ, INFB_WORKSPACE=str(tree), FETCHCONTENT_BASE_DIR=str(deps),
               CMAKE_ARGS=(os.environ.get("CMAKE_ARGS", "") + " -DFETCHCONTENT_FULLY_DISCONNECTED=ON").strip())
    p = subprocess.run(["/bin/bash", "/public/build.sh"], env=env, cwd=str(tree), capture_output=True, text=True)
    build_s = time.time() - t_build
    (root / "build.log").write_text(p.stdout[-200000:] + "\n--- stderr\n" + p.stderr[-200000:])
    restore_install(site, saved)
    if p.returncode != 0:
        fatal(f"the reference build failed (rc {p.returncode}); see {root / 'build.log'}:\n{p.stderr[-4000:]}")
    back = engine_file()
    if back != str((app / "vllm/__init__.py").resolve()):
        fatal(f"after restoring the venv's install, vllm resolves to {back}, not {app}")
    if subprocess.run(["git", "-C", str(tree), "diff", "--quiet"]).returncode != 0:
        fatal("the build changed a tracked file of the reference tree")
    t_comp = time.time()
    c = subprocess.run([str(VENV / "bin/python"), "-m", "compileall", "-q", "-j", "0", str(tree / "vllm")],
                       capture_output=True, text=True)
    compiled = {"rc": c.returncode, "seconds": round(time.time() - t_comp, 3),
                "output": (c.stdout + c.stderr)[-2000:] if c.returncode else ""}
    products = [{"path": rel, "bytes": (tree / rel).stat().st_size, "sha256": sha256_file(tree / rel)}
                for rel in ignored_products(tree, ["vllm"])]
    snap = {"schema": "infb-m3-reference-snapshot/1", "base": ref["base"], "patch": str(patch), "patch_sha256": got,
            "tree": tree_hash, "app": str(app), "deps_sources": fetched, "deps_copy_s": round(deps_s, 3),
            "build_s": round(build_s, 3),
            "build_command": "/public/build.sh (INFB_WORKSPACE=<tree>, FETCHCONTENT_BASE_DIR=<tree>/.deps, "
                             "CMAKE_ARGS=-DFETCHCONTENT_FULLY_DISCONNECTED=ON)", "venv_install_restored": saved,
            "products": products, "compiled": compiled,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 3)}
    (root / "snapshot.json").write_text(json.dumps(snap, indent=1))
    print(f"[snapshot] reference tree {tree_hash} at {tree} (base {ref['base'][:12]} + {patch.name}): built in "
          f"{build_s:.0f} s, {len(products)} build product(s), {snap['seconds']:.0f} s in all", flush=True)
    return snap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="/tests/config.json")
    args = ap.parse_args(argv)
    build(Path(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
