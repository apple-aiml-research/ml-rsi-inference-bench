#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Merge the per-operation JUnit reports of one step into the report the grader reads."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(argv: list[str]) -> int:
    vdir, prefix = Path(argv[1]), argv[2]
    merged = ET.Element("testsuites")
    parts = sorted(vdir.glob(f"junit-{prefix}-*.xml"))
    n = 0
    for part in parts:
        try:
            root = ET.parse(part).getroot()
        except ET.ParseError as exc:
            print(f"[{prefix}] {part.name} is not a JUnit report ({exc}); its ids count as missing")
            continue
        for suite in (root.iter("testsuite") if root.tag == "testsuites" else [root]):
            merged.append(suite)
            n += 1
    ET.ElementTree(merged).write(vdir / f"junit-{prefix}.xml", encoding="utf-8", xml_declaration=True)
    print(f"[{prefix}] merged {n} suite(s) from {len(parts)} report(s) into junit-{prefix}.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
