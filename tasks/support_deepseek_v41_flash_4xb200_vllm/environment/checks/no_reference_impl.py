#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if an importable DeepSeek-V4.1 implementation is installed anywhere.

The model-author implementation ships with the checkpoint (`inference/model.py`, `engram.py`,
`kernel.py`), which is on no interpreter's path: it is there to read, and the fidelity gate measures
agreement against it. What must not exist is an installed runtime a candidate could delegate to:

* no Transformers release carries the model (checked: no `transformers.models.deepseek_v41`);
* nothing on any interpreter's path outside the engine tree names it (`deepseek_v41`,
  `deepseek_v4_1`);
* the engine tree at the prepared base carries no V4.1 model implementation: that IS the task. The
  base does carry the model's FRONTENDS (tokenizer and prompt encoder, parsers, the configuration
  class) and one file under the package directory, `vllm/models/deepseek_v4_1/common/mm_preprocess.py`
  (image-token preprocessing, added with the frontends); those are upstream history at the base and
  allowed by name, and nothing else under that directory is.

This runs at image build, where it is the durable half of the rule: a later lock bump that quietly
installs one cannot reopen the hole silently.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

INTERPRETERS = ("/opt/vllm/bin/python", "/opt/infb/tools/bin/python")
EXCLUDED = ("/app", "/public", "/model", "/hf")
NAMES = ("*deepseek_v41*", "*deepseek_v4_1*")


def main() -> int:
    roots: set[str] = set()
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
        for pattern in NAMES:
            hits += [str(p) for p in path.rglob(pattern)]

    try:
        spec = importlib.util.find_spec("transformers.models.deepseek_v41")
    except Exception:
        spec = None

    allowed = {"common/mm_preprocess.py"}
    app_pkg = []
    for pkg in pathlib.Path("/app/vllm/models").glob("deepseek_v4*1*"):
        for f in pkg.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.relative_to(pkg).as_posix() not in allowed:
                app_pkg.append(str(f))

    import transformers

    print(f"transformers resolved by this interpreter: {transformers.__version__}")
    print(f"paths searched: {len(roots)}")
    print(f"deepseek_v41 on a site-packages path: {hits or 'none'}")
    print(f"importable as transformers.models.deepseek_v41: {spec}")
    print(f"V4.1 package in the engine tree at the prepared base: {app_pkg or 'none'}")

    if hits or spec is not None or app_pkg:
        print(
            "FATAL: an importable DeepSeek-V4.1 implementation is installed in this image; the "
            "task supplies the model-author source with the checkpoint to read, not a runtime to "
            "delegate to",
            file=sys.stderr,
        )
        return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
