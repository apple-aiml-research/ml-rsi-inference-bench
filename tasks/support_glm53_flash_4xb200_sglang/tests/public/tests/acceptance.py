#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The maintainers' acceptance classes in `test_glm53_flash_b200.py`, bound to a server that is already running."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACCEPTANCE = HERE / "test_glm53_flash_b200.py"
MODEL = "/model/zai-org/GLM-5.3-Flash"
CLASSES = {"high_throughput": "TestGLM53FlashB200HighThroughput", "low_latency": "TestGLM53FlashB200LowLatency"}


def _module():
    spec = importlib.util.spec_from_file_location("test_glm53_flash_b200", ACCEPTANCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acceptance_class(name: str):
    return getattr(_module(), CLASSES[name])


def server_args(name: str) -> list[str]:
    return list(acceptance_class(name).server_args)


def bound(cls, base_url: str, model: str = MODEL):
    class Bound(cls):
        @classmethod
        def setUpClass(klass):
            klass.base_url, klass.model, klass.process = base_url.rstrip("/"), model, None

        @classmethod
        def tearDownClass(klass):
            pass

    Bound.__name__ = Bound.__qualname__ = cls.__name__
    return Bound


def run_methods(cls, names: list[str]) -> unittest.TestResult:
    return unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cls(name) for name in names))
