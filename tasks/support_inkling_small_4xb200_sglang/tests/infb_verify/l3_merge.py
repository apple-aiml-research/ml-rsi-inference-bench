#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The six L3 test files run one pytest process each (test.sh); this merges their reports into the one the grader
reads (`junit-l3.xml`) and, when the step's wall stopped a file within its own limit (censored) or left files unrun
(blocked), writes `l3-wall.json` with the ids concerned for `disposition.py`."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def ids_of(path: str, p2p: list[str]) -> list[str]:
    prefix = path.removesuffix(".py").replace("/", ".") + "."
    return [i for i in p2p if i.startswith(prefix)]


def main(argv: list[str]) -> int:
    vdir, cfg = Path(argv[1]), json.loads(Path(argv[2]).read_text())
    censored = argv[3] if len(argv) > 3 else ""
    not_run = [a for a in argv[4:] if a]
    merged = ET.Element("testsuites")
    n = 0
    for part in sorted(vdir.glob("junit-l3-*.xml")):
        root = ET.parse(part).getroot()
        for suite in (root.iter("testsuite") if root.tag == "testsuites" else [root]):
            merged.append(suite)
            n += 1
    ET.ElementTree(merged).write(vdir / "junit-l3.xml", encoding="utf-8", xml_declaration=True)
    print(f"[l3] merged {n} suite(s) from {len(list(vdir.glob('junit-l3-*.xml')))} file report(s) into junit-l3.xml")
    if censored or not_run:
        p2p = cfg["p2p_node_ids"]
        wall = {"censored_by": censored or None, "censored_ids": ids_of(censored, p2p) if censored else [],
                "not_run_files": not_run, "not_run": [i for f in not_run for i in ids_of(f, p2p)]}
        (vdir / "l3-wall.json").write_text(json.dumps(wall, indent=1))
        print(f"[l3] the step's wall: {json.dumps(wall)[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
