#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The adapter that lets the supplied upstream acceptance test run on this tree, offline. The two upstream files beside
it (`test_deepseek_v4_flash_fp4_b200.py`, `server_sanity_kit.py`) are byte for byte the maintainers'."""
from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = "sglang.test.kits.server_sanity_kit"


def supply_sanity_kit() -> str:
    """Make `sglang.test.kits.server_sanity_kit` importable; returns where it came from."""
    try:
        importlib.import_module(KIT)
        return "the tree's own"
    except ModuleNotFoundError as exc:
        if exc.name not in (KIT, "sglang.test.kits"):
            raise
    try:
        importlib.import_module("sglang.test.kits")
    except ModuleNotFoundError as exc:
        if exc.name != "sglang.test.kits":
            raise
        pkg = types.ModuleType("sglang.test.kits")
        pkg.__path__ = []
        sys.modules["sglang.test.kits"] = pkg
    spec = importlib.util.spec_from_file_location(KIT, HERE / "server_sanity_kit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[KIT] = module
    spec.loader.exec_module(module)
    sys.modules["sglang.test.kits"].server_sanity_kit = module
    return str(HERE / "server_sanity_kit.py")


def supply_gsm8k(dst: Path = Path("/tmp/test.jsonl")) -> None:
    src = HERE / "fixtures" / "gsm8k_test.jsonl"
    if src.is_file() and not dst.exists():
        shutil.copyfile(src, dst)


supply_sanity_kit()
supply_gsm8k()
