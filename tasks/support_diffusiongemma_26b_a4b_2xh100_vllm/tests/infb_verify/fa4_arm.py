#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The component comparator arm: the `attend` timing collection over
the reference tree's oracle adapter, in its own process, before any candidate code is imported.

    python -B /tests/infb_verify/fa4_arm.py --tree /opt/infb/reference/app --out DIR/attend_reference.json

The import path is the reference tree first (so `vllm` is the reference's, asserted), then the trusted
evaluation code under /tests. Each workload's draw size N is recorded with its times; the candidate's
collection reuses the same N (test_attend.timing), so both arms time the same number of calls.
Expected: completes on every verifier run; its record is what the three efficiency verdicts compare to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", default=os.environ.get("TESTS_DIR", "/tests"))
    args = ap.parse_args()
    tree = Path(args.tree).resolve()
    sys.path[:0] = [str(tree), args.tests]
    import vllm  # noqa: PLC0415

    src = Path(vllm.__file__).resolve()
    if tree not in src.parents:
        raise SystemExit(f"[fa4_arm] FATAL: vllm imports from {src}, not the reference tree {tree}")
    from eval.diffusion_gemma.attend import check, measure  # noqa: PLC0415

    impl = check.load_impl(str(tree / "diffusion_gemma_attend.py"))
    t0 = time.monotonic()
    coll = measure.collection(impl)
    ns = {f"{key}/{lk}": row["per_layer"][lk]["n"] for key, row in coll.items() for lk in row["per_layer"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "diffusion-gemma-attend-reference/1", "tree": str(tree), "vllm": str(src),
                               "seconds": round(time.monotonic() - t0, 1), "n": ns, "collection": coll}, indent=1))
    print(f"[fa4_arm] reference collection in {time.monotonic() - t0:.1f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
