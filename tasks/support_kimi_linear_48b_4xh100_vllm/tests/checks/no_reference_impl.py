#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if an importable Kimi-Linear implementation is installed anywhere.

The model-author implementation is `modeling_kimi.py`, shipped under `/app/eval/kimi_linear/reference/`
(a tracked link onto the checkpoint's own file) to READ AND PORT -- that is the task. An installed runtime to DELEGATE to is not. Transformers is pinned at 4.57.1 and no released
transformers through v5.2.0 carries a native `models/kimi_linear` (GitHub contents API, 2026-09-23;
`main` does). But that is an argument, and an argument is not a guard: this check makes it a build
assertion, so a later lock bump that reaches such a release cannot reopen the hole silently.

Three things are looked for, each for its own reason:

* `kimi_linear` / `modeling_kimi` on an interpreter's path -- the model itself, in either the
  Transformers spelling or the model author's;
* `fla` / `fla_core` -- the model author's kernel library. It is deliberately absent: vLLM vendors
  its own copy of the delta-rule ops under `vllm/model_executor/layers/fla/`, in the tree, and those
  are the kernels the task supplies. An installed `fla-core` would be a second, unvendored source of
  the same kernels and is not what the agent is asked to build against. The fidelity reference DOES
  need it.

One path is excluded, deliberately:

* `/app` -- the engine tree, where `vllm/model_executor/layers/fla/` holds the supplied Triton ops
  and, after a submission, whatever the candidate wrote. That IS the task. Its evaluation tree
  `/app/eval/kimi_linear/` holds the model-author source to read, and sits on no interpreter's path.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import site
import subprocess
import sys

INTERPRETERS = ("/opt/vllm/bin/python", "/opt/infb/tools/bin/python")
EXCLUDED = ("/app",)
#: Names whose presence on an interpreter's path would be an implementation of this model, or the
#: model author's kernel library. Globs, matched against every path under every site directory.
FORBIDDEN_GLOBS = ("*kimi_linear*", "*modeling_kimi*", "fla", "fla_core", "fla_core-*")
#: Modules whose importability is the same question asked of the interpreter rather than the tree.
FORBIDDEN_MODULES = ("transformers.models.kimi_linear", "fla", "fla_core")


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
        for pattern in FORBIDDEN_GLOBS:
            hits += [str(p) for p in path.rglob(pattern)]

    importable: dict[str, str] = {}
    for name in FORBIDDEN_MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is not None:
            importable[name] = str(spec.origin)

    import transformers

    print(f"transformers resolved by this interpreter: {transformers.__version__}")
    print(f"paths searched: {len(roots)}")
    print(f"kimi_linear / modeling_kimi / fla on a site path: {hits or 'none'}")
    print(f"importable: {importable or 'none'}")

    if hits or importable:
        print(
            "FATAL: an importable Kimi-Linear implementation or the model author's kernel library is "
            "installed in this image; the task supplies the model-author source under "
            "/app/eval/kimi_linear/reference/ to read, not a runtime to delegate to",
            file=sys.stderr,
        )
        return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
