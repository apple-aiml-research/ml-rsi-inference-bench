#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if an importable qwen4_exp implementation is installed anywhere.

The EvalScope set pins transformers 5.17.0, the release that carries
``transformers/models/qwen4_exp/`` -- a complete, working implementation of the model this task asks
the agent to write, and the very implementation the fidelity gate measures agreement against.
Supplying ``modeling_qwen4_exp.py`` under ``/public`` to read and port is the task; shipping an
installed runtime to delegate to is not.

Two paths are excluded, both deliberately:

* ``/app`` -- the engine tree, where ``vllm/models/qwen4_exp/`` holds the supplied Triton kernels
  and, after a submission, whatever the candidate wrote there. That IS the task.
* ``/public`` -- the model-author source, which sits on no interpreter's path.

This runs at image build, where it is the durable half of the fix: a later lock bump that quietly
reintroduces 5.17.0 cannot reopen the hole silently.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import site
import subprocess
import sys

INTERPRETERS = ("/opt/vllm/bin/python", "/opt/evalscope/bin/python", "/opt/infb/tools/bin/python")
EXCLUDED = ("/app", "/public")


def main() -> int:
    roots: set[str] = set(site.getsitepackages())
    for interpreter in INTERPRETERS:
        if not pathlib.Path(interpreter).exists():
            continue
        out = subprocess.run(
            [interpreter, "-c", "import sys, json; print(json.dumps([p for p in sys.path if p]))"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        roots.update(json.loads(out))

    hits: list[str] = []
    for root in sorted(roots):
        path = pathlib.Path(root)
        if not path.is_dir() or str(path).startswith(EXCLUDED):
            continue
        hits += [str(p) for p in path.rglob("*qwen4_exp*")]

    try:
        spec = importlib.util.find_spec("transformers.models.qwen4_exp")
    except Exception:
        spec = None

    import transformers

    print(f"transformers resolved by this interpreter: {transformers.__version__}")
    print(f"paths searched: {len(roots)}")
    print(f"qwen4_exp on a site-packages path: {hits or 'none'}")
    print(f"importable as transformers.models.qwen4_exp: {spec}")

    if hits or spec is not None:
        print(
            "FATAL: an importable qwen4_exp implementation is installed in this image; the task "
            "supplies the model-author source under /public to read, not a runtime to delegate to",
            file=sys.stderr,
        )
        return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
