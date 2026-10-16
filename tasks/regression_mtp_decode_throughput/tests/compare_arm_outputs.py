#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Compare the two arms' greedy generations and report how far apart they are."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def compare(base: dict, cand: dict) -> dict[str, object]:
    b, c = base.get("token_ids") or [], cand.get("token_ids") or []
    n = min(len(b), len(c))
    exact = sum(1 for i in range(n) if b[i] == c[i])
    matched = compared = 0
    prefixes: list[float] = []
    first_div: list[int | None] = []
    for i in range(n):
        bi, ci = b[i], c[i]
        m = min(len(bi), len(ci))
        compared += m
        same = 0
        div: int | None = None
        for j in range(m):
            if bi[j] == ci[j]:
                same += 1
            elif div is None:
                div = j
        matched += same
        prefixes.append((div if div is not None else m) / m if m else 1.0)
        first_div.append(div)
    return {
        "sequences_compared": n,
        "exact_sequences": exact,
        "exact_fraction": (exact / n) if n else 0.0,
        "token_agreement": (matched / compared) if compared else 0.0,
        "prefix_agreement": (sum(prefixes) / len(prefixes)) if prefixes else 0.0,
        "first_divergence_per_sequence": first_div,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, cand = _load(Path(args.baseline)), _load(Path(args.candidate))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if base is None or cand is None or not base.get("ok") or not cand.get("ok"):
        payload = {
            "rc": 2,
            "verdict": "unusable: an arm produced no generations",
            "baseline_ok": bool(base and base.get("ok")),
            "candidate_ok": bool(cand and cand.get("ok")),
            "baseline_error": (base or {}).get("error"),
            "candidate_error": (cand or {}).get("error"),
        }
    else:
        payload = {"rc": 0, **compare(base, cand)}
        payload["verdict"] = (
            f"{payload['exact_sequences']}/{payload['sequences_compared']} sequences identical, "
            f"token agreement {payload['token_agreement']:.4f}, "
            f"prefix agreement {payload['prefix_agreement']:.4f}"
        )

    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[equivalence] rc={payload['rc']} {payload['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
