#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The reference phase, run by test.sh after preflight and before L0, while no candidate code has
been imported: check the reference snapshot, collect the `attend` reference timings in their own
process, then serve the reference tree and record its efficiency draws.

    python3 -I /tests/infb_efficiency/reference.py --config /tests/config.json

Any failure exits 3 (a verifier fault: the reward is withheld, never a candidate zero).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TESTS = Path(os.environ.get("TESTS_DIR", "/tests"))
OUT = Path(os.environ.get("INFB_EFFICIENCY_DIR", "/logs/verifier/efficiency"))
PY = os.environ.get("INFB_PYTHON", "/opt/vllm/bin/python")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_snapshot(ref: dict) -> tuple[Path, dict]:
    root = Path(ref["root"])
    tree = root / ref.get("tree_dir", "tree")
    snap = json.loads((root / "snapshot.json").read_text())
    if snap.get("patch_sha256") != ref["patch_sha256"] or snap.get("tree") != ref["target_tree"]:
        raise SystemExit(f"[reference] FATAL: snapshot {snap.get('tree')} / {snap.get('patch_sha256')} is not the "
                         f"configured reference {ref['target_tree']} / {ref['patch_sha256']}")
    for prod in snap.get("products", []):
        got = sha256_file(tree / prod["path"])
        if got != prod["sha256"]:
            raise SystemExit(f"[reference] FATAL: {prod['path']} changed since the image build ({got[:12]})")
    for link in snap.get("links", []):
        if os.path.realpath(tree / link["link"]) != link["realpath"]:
            raise SystemExit(f"[reference] FATAL: {link['link']} no longer points at {link['realpath']}")
    return tree, snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(TESTS / "config.json"))
    args = ap.parse_args()
    ref = json.loads(Path(args.config).read_text())["infb"]["efficiency"]["reference"]
    tree, snap = check_snapshot(ref)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[reference] snapshot {snap['tree']} ok ({len(snap.get('products', []))} products)", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
    arm = subprocess.run([PY, "-B", str(TESTS / "infb_verify/fa4_arm.py"), "--tree", str(tree),
                          "--out", str(OUT / "attend_reference.json")], cwd=str(tree), env=env)
    if arm.returncode != 0:
        raise SystemExit(f"[reference] FATAL: the component comparator arm exited {arm.returncode}")
    for p in (str(TESTS / "eval"), str(TESTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("INFB_REFERENCE_TREE", str(tree))
    from infb_verify import efficiency  # noqa: PLC0415

    rec = efficiency.reference_phase()
    print("[reference] " + json.dumps({d: {"ready_s": r["ready_s"], "c1": round(r["c1"]["value"], 2),
                                            "c4": round(r["c4"]["value"], 2)} for d, r in rec.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
