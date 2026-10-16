#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""L0, build and registry: the committed tree installs and the engine resolves the model natively."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
PY = sys.executable
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from glm53_protocols.native import (install_editable, native_problems, resolution_record,  # noqa: E402
                                    resolve_architecture, sglang_location)


def test_build_editable_install():
    """The tree installs editable, offline, without build isolation (the backend is in the image),
    and the package resolves from /app/python rather than from any cached copy."""
    r = install_editable(APP_DIR, PY)
    assert r.returncode == 0, "editable install failed:\n" + r.stdout[-4000:] + r.stderr[-4000:]
    r = sglang_location(PY)
    assert r.returncode == 0, "import sglang failed:\n" + r.stdout[-2000:] + r.stderr[-4000:]
    assert str((APP_DIR / "python").resolve()) in r.stdout, f"sglang resolved outside the tree: {r.stdout.strip()}"


def test_registry_resolves_native_class():
    """The checkpoint's config parses under the pinned Transformers through the engine (the base
    cannot: Transformers 5.12.1 does not know the model type) and the engine's own registry resolves
    the declared architecture to a native SGLang model class, not the Transformers fallback."""
    r = resolve_architecture(X["model_dir"], PY)
    assert r.returncode == 0, "config did not parse or the architecture did not resolve:\n" + r.stdout[-3000:] + r.stderr[-8000:]
    info = resolution_record(r.stdout)
    assert info, "no resolution record in output:\n" + r.stdout[-3000:]
    problems = native_problems(info, X["architectures"][0], X["model_type"])
    assert not problems, "; ".join(problems)
