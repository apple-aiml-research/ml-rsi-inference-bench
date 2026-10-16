#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if an importable model-author MiniMax-M3 or MSA implementation is installed anywhere.

Transformers 5.12.0 and later carry ``transformers/models/minimax_m3_vl/`` (a working implementation of the model
this task asks for, and of its sparse attention) and ``integrations/msa_attention.py``; the engine pins 5.5.3,
which carries neither. The MSA package is withheld at every revision. Supplying the v5.13.0
modeling source under ``/public/reference-impl`` to READ is part of the task; an installed runtime to delegate to
is not.

Two paths are excluded, deliberately: ``/app`` (the engine tree the agent edits; after a submission, whatever
the candidate wrote there) and ``/public`` (source on no interpreter's path).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

INTERPRETERS = ("/opt/vllm/bin/python", "/opt/infb/tools/bin/python")
#: The MSA package's module name, spelled in two parts so this file does not contain the identifier
#: literally.
MSA_MODULE = "fmha" + "_sm100"
EXCLUDED = ("/app", "/public")
NAME_GLOBS = ("*minimax_m3*", f"*{MSA_MODULE}*", "msa_attention*", "msa", "msa-*")
PROBE = r"""
import importlib.util, json, sys
out = {}
for name in ("transformers.models.minimax_m3_vl", "transformers.integrations.msa_attention", sys.argv[1],
             "msa", "kernels"):
    try:
        out[name] = importlib.util.find_spec(name) is not None
    except Exception:
        out[name] = False
try:
    import transformers
    out["transformers"] = transformers.__version__
except Exception:
    out["transformers"] = None
print(json.dumps({"importable": out, "path": [p for p in sys.path if p]}))
"""


def main() -> int:
    hits: list[str] = []
    roots: set[str] = set()
    for interpreter in INTERPRETERS:
        if not pathlib.Path(interpreter).exists():
            continue
        probe = json.loads(subprocess.run([interpreter, "-c", PROBE, MSA_MODULE], capture_output=True, text=True,
                                          check=True, cwd="/").stdout)
        roots.update(probe["path"])
        print(f"{interpreter}: transformers {probe['importable'].pop('transformers')}; importable {probe['importable']}")
        hits += [f"{interpreter}: importable {m}" for m, ok in probe["importable"].items() if ok]
    for root in sorted(roots):
        path = pathlib.Path(root)
        if not path.is_dir() or str(path).startswith(EXCLUDED):
            continue
        for pattern in NAME_GLOBS:
            hits += [str(p) for p in path.rglob(pattern)]
    print(f"paths searched: {len(roots)}; hits: {hits or 'none'}")
    if hits:
        print("FATAL: an importable model-author or MSA implementation is installed in this image; the task "
              "supplies the model-author source under /public to read, not a runtime to delegate to",
              file=sys.stderr)
        return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
