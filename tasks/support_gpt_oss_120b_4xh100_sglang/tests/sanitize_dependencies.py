#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Remove duplicate target GPU implementations before copying the final venv.

Run in an intermediate build stage after the locked wheel installation. The
final stage copies only that stage's sanitized filesystem. Neither the original
wheels nor their earlier layers belong to the agent image.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sysconfig
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_gpu_casts(path: Path) -> dict:
    original = digest(path)
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    names = {"_fp4_packed_to_bf16", "triton_f4_to_bf16_kernel", "triton_f4_to_scaled_bf16_kernel",
             "triton_f4_to_bf16", "triton_f4_to_scaled_bf16"}
    functions = [fn for fn in ast.walk(ast.parse(text)) if isinstance(fn, ast.FunctionDef) and fn.name in names]
    if {fn.name for fn in functions} != names:
        raise ValueError("TorchAO source shape differs from its locked version")
    removed = []
    for fn in sorted(functions, key=lambda f: f.lineno, reverse=True):
        kernel = any(ast.unparse(d).startswith("triton.jit") for d in fn.decorator_list)
        statement = ('tl.static_assert(False, "packed FP4 conversion is not implemented")'
                     if kernel else 'raise NotImplementedError("packed FP4 conversion is not implemented")')
        lines[fn.body[0].lineno - 1:fn.end_lineno] = [" " * fn.body[0].col_offset + statement + "\n"]
        removed.append(fn.name)
    path.write_text("".join(lines))
    ast.parse(path.read_text())
    return {"before": original, "after": digest(path), "functions": sorted(removed)}


def sanitize(root: Path, record: Path) -> None:
    duplicates = root / "triton_kernels"
    files = sorted(duplicates.rglob("*.py"))
    if len(files) != 38:
        raise ValueError(f"expected the locked wheel's 38 duplicate kernel modules, found {len(files)}")
    out = {"duplicate_kernel_files": {str(p.relative_to(root)): digest(p) for p in files}}
    shutil.rmtree(duplicates)
    for metadata in root.glob("triton_kernels-*.dist-info"):
        shutil.rmtree(metadata)
    path = root / "torchao/prototype/mx_formats/custom_cast.py"
    out["torchao"] = remove_gpu_casts(path)
    removed = []
    for relative in ("transformers/models/gpt_oss/modeling_gpt_oss.py",
                     "transformers/models/gpt_oss/modular_gpt_oss.py", "transformers/integrations/mxfp4.py"):
        path = root / relative
        if path.is_file():
            removed.append({"path": relative, "sha256": digest(path)})
            path.unlink()
    out["model_implementations"] = removed
    # Setuptools' test fixtures embed zipped and unpacked example eggs. They
    # are not runtime dependencies and do not belong in the offline surface.
    fixtures = root / "pkg_resources/tests"
    if fixtures.is_dir():
        out["package_test_fixtures"] = str(fixtures.relative_to(root))
        shutil.rmtree(fixtures)
    # All bytecode is regenerated only from sanitized files. This also removes
    # orphan pycs and code objects compiled while pip inspected dependencies.
    for path in list(root.rglob("__pycache__")):
        shutil.rmtree(path)
    for suffix in ("*.pyc", "*.pyo"):
        for path in root.rglob(suffix):
            path.unlink()
    if list(root.rglob("*.whl")) or list(root.rglob("*.egg")):
        raise ValueError("an installed wheel/archive carrier remains in the venv")
    if list(root.rglob("modeling_gpt_oss.py")) or list(root.rglob("modular_gpt_oss.py")):
        raise ValueError("a second model implementation remains")
    if (root / "triton_kernels").exists():
        raise ValueError("a duplicate kernel package remains")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site-packages", type=Path, default=Path(sysconfig.get_path("purelib")))
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()
    sanitize(args.site_packages, args.record)
