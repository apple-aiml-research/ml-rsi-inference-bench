#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The witness observer's entry point in every engine process of the (untimed) witness launch only."""
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("INFB_GLM53_WITNESS_DIR"):
    try:
        import glm53_witness

        glm53_witness.install()
    except Exception as _exc:  # noqa: BLE001 - never break the engine process from here
        try:
            os.makedirs(os.environ["INFB_GLM53_WITNESS_DIR"], exist_ok=True)
            with open(os.path.join(os.environ["INFB_GLM53_WITNESS_DIR"], f"install-error-{os.getpid()}.txt"), "a") as _f:
                _f.write(repr(_exc) + "\n")
        except OSError:
            pass

_rest = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.origin and os.path.dirname(os.path.abspath(_spec.origin)) != _HERE:
    try:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
    except Exception:  # noqa: BLE001 - the image's own hook is best-effort there too
        pass
