#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Build the engine tree offline, as the grader builds it: the editable SGLang install and the vendored FlashMLA.

    python /app/eval/deepseek_v4_flash/kernels/build.py --tree /app [--vendor-only] [--output /tmp/build.json]

  * The editable install is SGLang's own, offline and without build isolation, exactly the check in
    `../tests/check_native.py` (`sglang` must then resolve from `<tree>/python`).
  * The vendored FlashMLA at `<tree>/third_party/flash_mla` is rebuilt from its source: every stale build product is
    removed first (`build/`, `*.egg-info`, and every compiled object under the tree -- a generated binary is never a
    substitute for its source), then `setup.py build_ext --inplace` runs offline with this task's flags
    (`FLASH_MLA_DISABLE_SM90=1 MAX_JOBS=8 NVCC_THREADS=4`; the task targets SM100). Its CUTLASS is the generic checkout
    `csrc/cutlass` links to. The extension must then import from the tree, and the imported module must be the one
    this build wrote.

The whole command is one `kernel_build` operation (`../limits.json`: 3,600 s from build dispatch through usable
imports); the grader runs this same file on the committed tree before any other check, and a failure there fails the
build check. The output JSON records the tree, each step's duration and result, and the built products' digests.
Nothing is fetched, and nothing outside `<tree>` is written except the `--output` file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMITS = json.loads((HERE.parent / "limits.json").read_text())["operations"]
LIMIT_S = int(LIMITS["kernel_build"]["limit_s"])
VENDOR = Path("third_party") / "flash_mla"
BUILD_ENV = {"FLASH_MLA_DISABLE_SM90": "1", "MAX_JOBS": "8", "NVCC_THREADS": "4"}
PRODUCT_SUFFIXES = (".so", ".o", ".a", ".cubin", ".fatbin", ".ptx")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _native():
    spec = importlib.util.spec_from_file_location("deepseek_v4_flash_eval_check_native",
                                                  HERE.parent / "tests" / "check_native.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clear_products(vendor: Path) -> list[str]:
    """Remove every stale build product under the vendored tree; the removed paths, relative to it.

    The walk never follows a symlink, so the generic CUTLASS that `csrc/cutlass` links to is never touched."""
    removed = []
    for rel in ("build", "dist"):
        target = vendor / rel
        if target.is_symlink() or target.is_file():
            target.unlink()
            removed.append(rel)
        elif target.is_dir():
            shutil.rmtree(target)
            removed.append(rel + "/")
    for root, dirs, files in os.walk(vendor, followlinks=False):
        here = Path(root)
        for d in list(dirs):
            if (here / d).is_symlink():
                dirs.remove(d)
            elif d.endswith(".egg-info"):
                shutil.rmtree(here / d)
                removed.append(str((here / d).relative_to(vendor)) + "/")
                dirs.remove(d)
        for f in files:
            path = here / f
            if not path.is_symlink() and path.suffix in PRODUCT_SUFFIXES:
                path.unlink()
                removed.append(str(path.relative_to(vendor)))
    return sorted(removed)


def vendor_build(tree: Path, py: str, deadline: float) -> dict:
    vendor = tree / VENDOR
    rec: dict = {"tree": str(vendor), "env": dict(BUILD_ENV)}
    if not (vendor / "setup.py").is_file():
        rec["failure"] = f"no vendored FlashMLA at {vendor}"
        return rec
    t0 = time.time()
    rec["removed"] = clear_products(vendor)
    env = dict(os.environ, **BUILD_ENV, PIP_NO_INDEX="1")
    left = max(1, int(deadline - time.time()))
    log = vendor.parent / ".flash_mla-build.log"
    try:
        with log.open("w") as fh:
            p = subprocess.run([py, "setup.py", "build_ext", "--inplace"], cwd=str(vendor), env=env,
                               stdout=fh, stderr=subprocess.STDOUT, timeout=left)
        rec["rc"] = p.returncode
    except subprocess.TimeoutExpired:
        rec["rc"] = None
        rec["failure"] = f"the vendored FlashMLA build exceeded the kernel_build limit ({LIMIT_S} s)"
    text = log.read_text(errors="replace") if log.exists() else ""
    log.unlink(missing_ok=True)
    rec["build_s"] = round(time.time() - t0, 1)
    rec["log_tail"] = text[-6000:]
    if rec.get("failure") or rec["rc"] != 0:
        rec.setdefault("failure", f"the vendored FlashMLA build failed (rc {rec['rc']})")
        return rec
    built = sorted(p for p in (vendor / "flash_mla").glob("cuda*.so") if p.stat().st_mtime >= t0 - 1)
    if len(built) != 1:
        rec["failure"] = f"the build left {len(built)} fresh flash_mla/cuda*.so, expected exactly one"
        return rec
    rec["products"] = [{"path": str(built[0].relative_to(tree)), "bytes": built[0].stat().st_size,
                        "sha256": _sha256(built[0])}]
    probe = ("import json, os, flash_mla, flash_mla.cuda; "
             "print('INFB_BUILD ' + json.dumps({'package': os.path.realpath(flash_mla.__file__), "
             "'extension': os.path.realpath(flash_mla.cuda.__file__)}))")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(vendor)
    left = max(1, int(deadline - time.time()))
    try:
        p = subprocess.run([py, "-c", probe], cwd="/", env=env, capture_output=True, text=True, timeout=left)
    except subprocess.TimeoutExpired:
        rec["failure"] = f"importing the built extension exceeded the kernel_build limit ({LIMIT_S} s)"
        return rec
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("INFB_BUILD ")]
    if p.returncode != 0 or not lines:
        rec["failure"] = "the built flash_mla does not import:\n" + (p.stdout + p.stderr)[-4000:]
        return rec
    got = json.loads(lines[-1][len("INFB_BUILD "):])
    rec["import"] = got
    if not got["package"].startswith(str(vendor.resolve()) + os.sep):
        rec["failure"] = f"flash_mla resolved outside the tree: {got['package']}"
    elif got["extension"] != str(built[0].resolve()):
        rec["failure"] = f"flash_mla.cuda is {got['extension']}, not the extension this build wrote ({built[0]})"
    return rec


def build(tree: Path, vendor_only: bool = False, py: str = sys.executable) -> dict:
    t0 = time.time()
    deadline = t0 + LIMIT_S
    out: dict = {"tree": str(tree), "limit_s": LIMIT_S, "operation": "kernel_build"}
    if not vendor_only:
        e0 = time.time()
        failures = _native().editable_install(tree, py)
        out["editable_install"] = {"seconds": round(time.time() - e0, 1), "failures": failures}
        if failures:
            out["failure"] = failures[0]
            out["seconds"] = round(time.time() - t0, 1)
            return out
    out["vendor_build"] = vendor_build(tree, py, deadline)
    if out["vendor_build"].get("failure"):
        out["failure"] = out["vendor_build"]["failure"]
    out["seconds"] = round(time.time() - t0, 1)
    if out["seconds"] > LIMIT_S and not out.get("failure"):
        out["failure"] = f"the build took {out['seconds']} s, over the kernel_build limit ({LIMIT_S} s)"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", default="/app", help="the engine tree to build (default /app)")
    ap.add_argument("--vendor-only", action="store_true", help="only the vendored FlashMLA, not the editable install")
    ap.add_argument("--output", default=None, help="write the build record here as JSON")
    args = ap.parse_args(argv)
    rec = build(Path(args.tree).resolve(), args.vendor_only)
    text = json.dumps(rec, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n")
    summary = {k: rec[k] for k in ("tree", "seconds", "limit_s") if k in rec}
    vb = rec.get("vendor_build", {})
    summary.update({"vendor_build_s": vb.get("build_s"), "products": vb.get("products")})
    print(json.dumps(summary, indent=1))
    if rec.get("failure"):
        print("build FAILED: " + rec["failure"], file=sys.stderr)
        return 1
    print("build holds: the editable tree and the vendored FlashMLA built offline and import from the tree"
          if not args.vendor_only else "build holds: the vendored FlashMLA built offline and imports from the tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
