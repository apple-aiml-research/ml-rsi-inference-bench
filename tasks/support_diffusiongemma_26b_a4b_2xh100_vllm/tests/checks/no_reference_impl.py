#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if an importable DiffusionGemma implementation is installed anywhere.

Transformers 5.11.0 and later carry ``transformers/models/diffusion_gemma/``, a complete, working
implementation of the model this task asks the agent to write, and the very implementation the served
denoiser is compared with. The engine's Transformers is 5.5.3, which does not; supplying the v5.13.0
files under ``/app/eval/diffusion_gemma/reference/model_author/`` to read is the task, shipping a
runtime to delegate to is not. Asserted in every interpreter of the image:

* ``transformers.models.diffusion_gemma`` is not importable;
* no file on any interpreter's path defines ``DiffusionGemmaForBlockDiffusion`` except the reading
  copies, and no installed package carries a ``diffusion_gemma`` module or directory;
* the reading copies' directory is on no interpreter's path.

This runs at image build, so a later lock change that quietly reintroduces the model cannot reopen the
hole silently.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

INTERPRETERS = ("/opt/vllm/bin/python", "/opt/infb/tools/bin/python")
READING = pathlib.Path("/app/eval/diffusion_gemma/reference/model_author")
TARGET = "class DiffusionGemmaForBlockDiffusion"
PROBE = (
    "import importlib.util, json, sys\n"
    "try:\n"
    "    spec = importlib.util.find_spec('transformers.models.diffusion_gemma') is not None\n"
    "except Exception:\n"
    "    spec = False\n"
    "print(json.dumps({'path': [p for p in sys.path if p], 'importable': spec}))\n"
)


def main() -> int:
    roots: set[pathlib.Path] = set()
    importable: dict[str, bool] = {}
    for interpreter in INTERPRETERS:
        if not pathlib.Path(interpreter).exists():
            continue
        out = json.loads(subprocess.run([interpreter, "-c", PROBE], capture_output=True, text=True,
                                        check=True).stdout)
        importable[interpreter] = out["importable"]
        roots.update(pathlib.Path(p) for p in out["path"])

    on_path = sorted(str(r) for r in roots if r == READING or READING in r.parents)
    defines: list[str] = []
    named: list[str] = []
    for root in sorted(roots):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if READING in path.parents or not path.is_file():
                continue
            try:
                if TARGET in path.read_text(encoding="utf-8", errors="ignore"):
                    defines.append(str(path))
            except OSError:
                continue
        if not str(root).startswith("/app"):
            named += [str(p) for p in root.rglob("*diffusion_gemma*")]

    import transformers

    print(f"transformers resolved by this interpreter: {transformers.__version__}")
    print(f"interpreter path entries searched: {len(roots)}")
    print(f"transformers.models.diffusion_gemma importable: {importable}")
    print(f"files defining {TARGET.split()[1]} outside the reading copies: {defines or 'none'}")
    print(f"installed diffusion_gemma modules: {named or 'none'}")
    print(f"reading copies on an interpreter's path: {on_path or 'no'}")
    if any(importable.values()) or defines or named or on_path:
        print("FATAL: an importable DiffusionGemma implementation is installed in this image; the task "
              "supplies the model-author source to read, not a runtime to delegate to", file=sys.stderr)
        return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
