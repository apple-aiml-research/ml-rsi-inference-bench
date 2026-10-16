#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0, build and registry: the committed tree installs and the engine resolves the model natively."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
PY = sys.executable
_spec = importlib.util.spec_from_file_location("deepseek_v4_flash_eval_check_native",
                                               APP_DIR / "eval" / "deepseek_v4_flash" / "tests" / "check_native.py")
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)
_bspec = importlib.util.spec_from_file_location("deepseek_v4_flash_eval_kernels_build",
                                                APP_DIR / "eval" / "deepseek_v4_flash" / "kernels" / "build.py")
kbuild = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(kbuild)
BUILD_RECORD = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier")) / "records" / "l0-build.json"


def test_build_editable_install():
    """The tree installs editable, offline, without build isolation (the backend is in the image), the package
    resolves from /app/python rather than from any cached copy, and the vendored FlashMLA builds from the committed
    source and imports from /app/third_party/flash_mla."""
    rec = kbuild.build(APP_DIR, vendor_only=False, py=PY)
    BUILD_RECORD.parent.mkdir(parents=True, exist_ok=True)
    BUILD_RECORD.write_text(json.dumps(rec, indent=1))
    assert not rec.get("failure"), rec["failure"]


def test_registry_resolves_native_class():
    """The checkpoint's config parses under the pinned Transformers through the engine (the base
    cannot: Transformers 5.12.1 does not know the model type) and the engine's own registry resolves
    the declared architecture to a native SGLang model class, not the Transformers fallback."""
    failures, _ = native.native_resolution(X["model_dir"], X["architectures"][0], X["model_type"], PY)
    assert not failures, failures[0]
