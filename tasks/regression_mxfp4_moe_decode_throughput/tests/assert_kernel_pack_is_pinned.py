#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Assert this image can express the defect at all, before anything is benched.

RUN BY tests/test.sh, on the GPU, before the rounds rig starts. It lives under tests/ rather than
in the image because stating the condition means naming the fix, and everything under environment/
is the docker build context.

WHAT THIS CATCHES, and why it is worth a separate file. The scored path runs through
`triton_kernels`, which is vendored inside the `sgl_kernel` wheel rather than installed as a
distribution of its own. There are two ways for the environment to be wrong, and the second
survives every check the first fails:

  SHAPE A -- the package is absent. `is_triton_kernels_available()` is a bare
  `find_spec("triton_kernels")`, so the engine picks a different MoE backend, serves this model
  correctly, and both arms read the same number. The image asserts this one at build time, and
  _candidate_runner.py asserts the engine's own startup banner at run time.

  SHAPE B -- the package is present but too NEW. From the triton_kernels shipped with sgl-kernel
  >= 0.4.4 (triton >= 3.6), `matmul_ogs` computes

      can_use_split_k = scatter_indx is None and not x_has_mx and not w_has_mx

  which forbids a split reduction for mxfp4 weights UNCONDITIONALLY. The behaviour this task
  measures then does not exist. Measured on a 0.5.15 / triton 3.6 environment: applying and
  reverting the reference repair gave 1.00x at EVERY batch size from 1 to 64, while the package
  imported, the backend was selected, the banner printed and the server served. Only the number
  changed -- and it changed to "this task is broken".

So `sgl-kernel==0.3.20` is part of the task definition rather than a version floor, and a future
"bump it so it installs" would delete the thing being measured while leaving the image green.

THE CHECK IS STRUCTURAL, NOT A SECOND READING OF THE VERSION STRING. environment/Dockerfile already
asserts the exact sgl-kernel version; repeating that here would prove nothing new. What this
asserts is the shape of `make_opt_flags`, which is the case a version check cannot see: a wheel
rebuilt under the same version number.

Exit codes: 0 = the environment can express the defect, 4 = it cannot (shape A or B), 2 = the check
itself could not run.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys


def main() -> int:
    spec = importlib.util.find_spec("triton_kernels")
    if spec is None:
        print(
            "FATAL (shape A): triton_kernels is not importable, so the engine will fall back to "
            "another MoE backend. It will serve correctly and both arms will read the same number. "
            "This is an environment fault, not a slow candidate.",
            file=sys.stderr,
        )
        return 4
    print(f"[kernel-pack] triton_kernels resolves to {spec.origin}")

    try:
        from triton_kernels.matmul_ogs_details import opt_flags
    except Exception as exc:  # noqa: BLE001 -- any import failure here is fatal and worth showing
        print(f"FATAL: triton_kernels is present but unusable: {exc!r}", file=sys.stderr)
        return 2

    try:
        params = list(inspect.signature(opt_flags.make_opt_flags).parameters)
    except (AttributeError, TypeError, ValueError) as exc:
        print(f"FATAL: cannot inspect make_opt_flags: {exc!r}", file=sys.stderr)
        return 2

    if "can_use_split_k" in params:
        print(
            "FATAL (shape B): this triton_kernels is newer than the one this task is pinned "
            "against. Its make_opt_flags takes can_use_split_k, which disables the launch "
            "configuration choice this task measures, for this quantisation format, "
            "unconditionally. The oracle would score 1.00x and every other check would pass.\n"
            f"  signature: {params}",
            file=sys.stderr,
        )
        return 4

    if not hasattr(opt_flags, "update_opt_flags_constraints"):
        print(
            "FATAL: triton_kernels has no update_opt_flags_constraints, which is the entry point "
            "the fix uses. The oracle could not apply against this build.",
            file=sys.stderr,
        )
        return 4

    print(
        f"[kernel-pack] make_opt_flags signature is the pinned one "
        f"({len(params)} parameters, no split-k predicate); this environment can express the defect"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
