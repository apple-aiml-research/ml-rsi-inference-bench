#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Assert the engine stack is installed and is the stack the lock names (image build only).

Why this is not simply `import sgl_kernel, triton_kernels.matmul_ogs, flashinfer`: those packages link
against **`libcuda.so.1`**, which is the host's NVIDIA *driver* library. It is not in a CUDA image and
never can be — the container runtime bind-mounts it at run time — so importing them during a build
fails with `ImportError: libcuda.so.1: cannot open shared object file` on a machine that has GPUs
sitting right there.

So the split is: at BUILD time, assert that every wheel actually landed and at the version the lock
pins, and that each kernel package is locatable without executing its extension module. And at
GRADING time the verifier's preflight imports them again (`config.json` `infb.importable`), so a
broken kernel package is an environment fault with the −1 sentinel rather than a candidate's failure.
"""
from __future__ import annotations

import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version

# distribution name -> the version the lock pins. A mismatch means the image is not the image.
EXPECTED = {
    "torch": "2.8.0+cu126",
    "torchvision": "0.23.0+cu126",
    "torchaudio": "2.8.0+cu126",
    "triton": "3.4.0",
    "sgl-kernel": "0.3.2",
    "flashinfer-python": "0.2.10",
    "transformers": "4.55.0",
    "openai": None,          # pinned by a range, not a version: report what landed
    "openai-harmony": "0.0.3",
    "compressed-tensors": "0.10.2",
    "pandas": None,
    "pytest": None,
}

# Importable-but-driver-linked: located, never imported here.
LOCATABLE = ["sgl_kernel", "triton_kernels", "flashinfer", "sglang"]


def main() -> int:
    failures: list[str] = []
    for dist, want in EXPECTED.items():
        try:
            got = version(dist)
        except PackageNotFoundError:
            failures.append(f"{dist} is not installed")
            continue
        mark = "ok  "
        if want is not None and got != want:
            failures.append(f"{dist} is {got}, the lock pins {want}")
            mark = "FAIL"
        print(f"[verify_stack] {mark} {dist}=={got}")

    for mod in LOCATABLE:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            failures.append(f"{mod} is not on the interpreter path")
            print(f"[verify_stack] FAIL {mod} not found")
        else:
            print(f"[verify_stack] ok   {mod} at {spec.origin}")

    if failures:
        for f in failures:
            print(f"[verify_stack] FAILURE: {f}", file=sys.stderr)
        return 1
    print("[verify_stack] the stack is the one the lock names; the driver-linked imports are the "
          "verifier preflight's to prove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
