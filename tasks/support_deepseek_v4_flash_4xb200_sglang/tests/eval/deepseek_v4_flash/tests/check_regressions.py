#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The regression behaviour the task preserves: the 104 unit tests in `regressions.json` that pass on the prepared base
and on a complete implementation, run from the engine's native test paths under `/app/test`, with the GPUs visible and no
server running. The grader restores the engine's test packages (`python/sglang/test`, `test/`) to the base before
it runs them, so edits there do not change the result; run this after committing your work."""
from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="run the 104 preserved regression tests")
    ap.add_argument("--selection", default=str(Path(__file__).resolve().parent / "regressions.json"))
    args = ap.parse_args(argv)
    sel = json.loads(Path(args.selection).read_text())
    root = Path(sel["rootdir"])
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        env = dict(os.environ, PYTHONPATH=str(root))
        r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--continue-on-collection-errors",
                            f"--rootdir={root}", "-q", *sel["ids"], f"--junitxml={junit}"],
                           cwd=str(root), env=env)
        outcomes = {}
        if junit.is_file():
            for case in ET.parse(junit).getroot().iter("testcase"):
                nid = f"{case.get('classname')}.{case.get('name')}"
                bad = case.find("failure") is not None or case.find("error") is not None
                outcomes[nid] = "skipped" if case.find("skipped") is not None else "failed" if bad else "passed"
    passed = sum(1 for v in outcomes.values() if v == "passed")
    print(f"regressions: {passed} of {len(sel['ids'])} passed (pytest rc {r.returncode})")
    return 0 if passed == len(sel["ids"]) and r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
