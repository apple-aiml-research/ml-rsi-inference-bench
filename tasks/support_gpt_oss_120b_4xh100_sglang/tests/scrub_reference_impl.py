#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Remove transformers' own gpt-oss MODEL implementation, and its MXFP4 expert integration, from the image;
assert both are gone.

No importable implementation of the model the task asks the agent to write may exist in the graded
image. Here that hole arrives through the engine's OWN dependency rather than a verifier venv --
transformers 4.55.0 is what SGLang's pyproject pins at the base commit, and it ships
``transformers/models/gpt_oss/modeling_gpt_oss.py``, a complete working implementation. The
distinction that decides it: supplied SOURCE to read and port is the task
(/public carries the model-author reference code); an installed RUNTIME to delegate to is not.

What stays, and why it is not optional:

* ``configuration_gpt_oss.py``. SGLang's ``get_config`` calls ``AutoConfig.from_pretrained`` FIRST,
  and the landing registers no ``gpt_oss`` entry in SGLang's own ``_CONFIG_REGISTRY`` (measured at
  the reference), so without the config class the checkpoint does not parse at all and the base's
  failure witness changes from ``Unknown quantization method: mxfp4`` to an unrecognised
  architecture. The instruction names it as supplied.
* ``quantizers/quantizer_mxfp4.py``. transformers imports it itself, eagerly:
  ``quantizers/auto.py`` has ``from .quantizer_mxfp4 import Mxfp4HfQuantizer`` at module level, and
  ``modeling_utils`` imports ``quantizers``, so every process that imports ``modeling_utils`` loads
  it, in the base tree and in the landing alike. Removing it would break transformers, and the
  engine with it. It reaches the integration only from inside its methods, which SGLang never
  calls, and the one piece of the kernel recipe it spells out is the ``PrecisionConfig``/``FlexCtx``
  form, which is the kernel's own API.

What goes besides the modeling files, and why:

* ``integrations/mxfp4.py``: Hugging Face's ``triton_kernels`` integration of this model's MXFP4
  experts ("Copied from GPT_OSS repo and vllm") -- the packed-block and scale layouts
  ``matmul_ogs`` wants, the swizzle, and the experts' forward through it. It is not the model, but
  it is a worked recipe for the hardest step the task asks for.

Nothing in the reference needs the removed files: the landing defines its OWN
``GptOssConfig(PretrainedConfig)`` in ``models/gpt_oss.py`` and imports only ``PretrainedConfig``
from transformers, and executes its experts with its own MXFP4 method. That is why the removal
cannot change the oracle's behaviour.

The assertions matter as much as the removal: they make a later lock bump that reintroduces the file
fail the BUILD rather than quietly reopen the hole.
"""
from __future__ import annotations

import glob
import importlib
import os
import sys

KEEP = ["__init__.py", "configuration_gpt_oss.py"]
# Beyond the model package: paths relative to the transformers package directory.
REMOVE = ["modeling_gpt_oss.py", "modular_gpt_oss.py"]
REMOVE_OTHER = ["integrations/mxfp4.py"]
KEEP_OTHER = ["quantizers/quantizer_mxfp4.py"]


def main() -> int:
    import transformers

    sp = os.path.dirname(transformers.__file__)
    pkg = os.path.join(sp, "models", "gpt_oss")
    if not os.path.isdir(pkg):
        raise SystemExit(f"transformers at {sp} has no models/gpt_oss package; the pin changed shape")

    for name in REMOVE:
        p = os.path.join(pkg, name)
        if os.path.exists(p):
            os.remove(p)
            print(f"[scrub] removed {p}")
    for cache in glob.glob(os.path.join(pkg, "__pycache__", "*")):
        os.remove(cache)
    try:
        os.rmdir(os.path.join(pkg, "__pycache__"))
    except OSError:
        pass

    left = sorted(os.path.basename(p) for p in glob.glob(os.path.join(pkg, "*")))
    if left != KEEP:
        raise SystemExit(f"[scrub] unexpected gpt_oss package contents after the removal: {left}")

    # Every interpreter path, not only this package: nothing anywhere may serve the model.
    stray: list[str] = []
    for d in sys.path:
        if not d:
            continue
        for pat in ("**/modeling_gpt_oss.py", "**/modeling_gpt_oss*.so", "**/modular_gpt_oss.py"):
            stray += glob.glob(os.path.join(d, pat), recursive=True)
    if stray:
        raise SystemExit(f"[scrub] a gpt-oss modeling module is still on an interpreter path: {sorted(set(stray))}")

    # The config must still resolve, through the same mapping AutoConfig.from_pretrained uses.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.gpt_oss import GptOssConfig

    mapped = CONFIG_MAPPING["gpt_oss"]
    if mapped is not GptOssConfig:
        raise SystemExit(f"[scrub] CONFIG_MAPPING['gpt_oss'] is {mapped!r}, not the kept GptOssConfig")

    # And the model classes must not be reachable.
    tf = importlib.import_module("transformers")
    reachable = []
    for name in ("GptOssForCausalLM", "GptOssModel", "GptOssPreTrainedModel", "GptOssForSequenceClassification"):
        try:
            getattr(tf, name)
        except Exception:  # noqa: BLE001 - any failure to resolve is the outcome we want
            continue
        reachable.append(name)
    if reachable:
        raise SystemExit(f"[scrub] still importable from transformers: {reachable}")

    # The MXFP4 integration: removed with its bytecode, absent from every interpreter path, not importable.
    for rel in REMOVE_OTHER:
        p = os.path.join(sp, rel)
        if os.path.exists(p):
            os.remove(p)
            print(f"[scrub] removed {p}")
        stem = os.path.splitext(os.path.basename(rel))[0]
        for pyc in glob.glob(os.path.join(sp, os.path.dirname(rel), "__pycache__", stem + ".*")):
            os.remove(pyc)
    stray = []
    for d in sys.path:
        if d:
            stray += glob.glob(os.path.join(d, "**", "transformers", "integrations", "mxfp4.py"), recursive=True)
    if stray:
        raise SystemExit(f"[scrub] the transformers MXFP4 integration is still on an interpreter path: {sorted(set(stray))}")
    try:
        importlib.import_module("transformers.integrations.mxfp4")
    except ImportError:
        pass
    else:
        raise SystemExit("[scrub] transformers.integrations.mxfp4 still imports")

    # What stays must still work without it: the quantizer transformers imports eagerly, modeling_utils
    # through it, and the config class (checked above).
    for rel in KEEP_OTHER:
        if not os.path.isfile(os.path.join(sp, rel)):
            raise SystemExit(f"[scrub] {rel} is gone; transformers imports it eagerly (quantizers/auto.py)")
    importlib.import_module("transformers.modeling_utils")
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer  # noqa: F401

    print(f"[scrub] ok: {left} kept, {REMOVE} and {REMOVE_OTHER} removed, {KEEP_OTHER} kept and importing, "
          "GptOssConfig resolves, no model class reachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
