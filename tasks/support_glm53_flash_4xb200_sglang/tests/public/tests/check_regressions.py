#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Existing engine behaviour this task keeps: the engine's own unit tests listed in `regressions.txt`.

    python3 /app/eval/glm53_flash/tests/check_regressions.py

It runs the listed tests (the pool configurator's, the chat-template parser's and the EAGLE worker's fast path, all CPU
only) from `/app/test/registered` with pytest, and prints each one's outcome. They pass on the starting commit.
Exit 0 when every listed test passes. `check_native.py` covers the editable install.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--app", default="/app")
    a = ap.parse_args()
    ids = [ln.strip() for ln in (HERE / "regressions.txt").read_text().splitlines()
           if ln.strip() and not ln.startswith("#")]
    cwd = Path(a.app) / "test" / "registered"
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rA", "-q",
                        f"--rootdir={Path(a.app) / 'test'}", *ids], cwd=str(cwd))
    print(f"{len(ids)} listed tests; pytest rc {r.returncode}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
