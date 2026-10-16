#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fail the build if an implementation of the withheld work is importable in any interpreter.

Checked in every interpreter the image carries (the engine's, the tooling venv's, the system one):

  * no importable ``transformers.models.inkling`` -- the engine's Transformers is 5.12.1, which does not
    include the model; releases from 5.14.0 do, and the model-author source is supplied under
    /public/reference-impl to read, not as a runtime to delegate to;
  * no importable ``sglang.jit_kernel.flash_attn`` -- the landing's in-tree FlashAttention-4 fork;
  * no FlashAttention-4 ``flash_attn_varlen_func`` that accepts ``rel_bias``: the supplied
    ``flash-attn-4`` must be exactly 4.0.0b15, which has none;
  * none of tml-fa4's distribution (``flash-attn-4`` with a ``-forward`` suffix, spelled apart in this file
    because the build's marker scan looks for the joined name), vLLM or TensorRT-LLM installed;
  * Transformers exactly 5.12.1 in the engine interpreter.

Paths under /app (the engine tree, the agent's workspace) and /public (read-only material on no
interpreter's path) are not site-packages and are not searched.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

INTERPRETERS = ("/opt/sglang/bin/python", "/opt/infb/tools/bin/python", "/usr/bin/python3")
PROBE = r"""
import ast, importlib.metadata as md, importlib.util, json, os, sys
out = {"python": sys.executable}
def spec(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
out["transformers.models.inkling"] = spec("transformers.models.inkling")
# The fork by the import system and by the file system: any sglang/jit_kernel/flash_attn on this path.
fork_dirs = []
for root in [p for p in sys.path if p and os.path.isdir(p)] + ["/app/python"]:
    d = os.path.join(root, "sglang", "jit_kernel", "flash_attn")
    if os.path.isdir(d):
        fork_dirs.append(d)
out["sglang.jit_kernel.flash_attn dirs"] = sorted(set(fork_dirs))
dists = {}
for name in ("transformers", "flash-attn-4", "flash-attn-4" + "-forward", "vllm", "tensorrt-llm", "tensorrt_llm"):
    try:
        dists[name] = md.version(name)
    except md.PackageNotFoundError:
        dists[name] = None
out["distributions"] = dists
# Import-free: read the installed FlashAttention-4 interface and look for a rel_bias parameter.
rel_bias = None
if dists["flash-attn-4"]:
    files = [f for f in (md.distribution("flash-attn-4").files or []) if str(f).replace(os.sep, "/").endswith("flash_attn/cute/interface.py")]
    if len(files) != 1:
        rel_bias = "error: interface.py not found exactly once in the flash-attn-4 distribution"
    else:
        tree = ast.parse(open(files[0].locate(), encoding="utf-8").read())
        fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "flash_attn_varlen_func"]
        if not fns:
            rel_bias = "error: flash_attn_varlen_func not defined in interface.py"
        else:
            names = {a.arg for fn in fns for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
            rel_bias = "rel_bias" in names
out["flash_attn.cute.flash_attn_varlen_func accepts rel_bias"] = rel_bias
print(json.dumps(out))
"""


def main() -> int:
    failures: list[str] = []
    reports = []
    for interpreter in INTERPRETERS:
        if not pathlib.Path(interpreter).exists():
            continue
        res = subprocess.run([interpreter, "-c", PROBE], capture_output=True, text=True)
        if res.returncode != 0:
            failures.append(f"{interpreter}: probe failed: {res.stderr.strip()[-500:]}")
            continue
        r = json.loads(res.stdout)
        reports.append(r)
        if r["transformers.models.inkling"]:
            failures.append(f"{interpreter}: transformers.models.inkling is importable")
        if r["sglang.jit_kernel.flash_attn dirs"]:
            failures.append(f"{interpreter}: sglang.jit_kernel.flash_attn present: {r['sglang.jit_kernel.flash_attn dirs']}")
        accepts = r["flash_attn.cute.flash_attn_varlen_func accepts rel_bias"]
        if accepts is True or isinstance(accepts, str):
            failures.append(f"{interpreter}: flash_attn_varlen_func rel_bias check: {accepts}")
        d = r["distributions"]
        for name in ("flash-attn-4" + "-forward", "vllm", "tensorrt-llm", "tensorrt_llm"):
            if d.get(name):
                failures.append(f"{interpreter}: {name} {d[name]} is installed")
        if d.get("flash-attn-4") not in (None, "4.0.0b15"):
            failures.append(f"{interpreter}: flash-attn-4 is {d['flash-attn-4']}, not 4.0.0b15")
        if interpreter == "/opt/sglang/bin/python":
            if d.get("transformers") != "5.12.1":
                failures.append(f"{interpreter}: transformers is {d.get('transformers')}, not 5.12.1")
            if d.get("flash-attn-4") != "4.0.0b15" or accepts is not False:
                failures.append(f"{interpreter}: the engine's FlashAttention-4 is not the plain 4.0.0b15 wheel")
    print(json.dumps(reports, indent=1))
    if failures:
        print("FATAL: an implementation of the withheld work is importable in this image:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 7
    print("no reference implementation importable in", ", ".join(r["python"] for r in reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
