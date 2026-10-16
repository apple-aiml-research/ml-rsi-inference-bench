#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Did the workload make the GDN decode kernel recompile, and did the candidate stop it?"""
from __future__ import annotations

import argparse
import json
import pathlib

KERNEL = "fused_recurrent_gated_delta_rule_fwd_kernel"

CANDIDATE_MAX_FRACTION = 0.25


def count(cache_dir: pathlib.Path) -> tuple[int, int]:
    """(compiles of the GDN kernel, compiles of anything) inside one arm's Triton cache."""
    if not cache_dir.is_dir():
        return -1, -1
    gdn = len(list(cache_dir.rglob(f"{KERNEL}.json")))
    # `__grp__<name>.json` is the group manifest that sits beside the real one; counting it would
    # double every kernel.
    allk = len([p for p in cache_dir.rglob("*.json") if not p.name.startswith("__grp__")])
    return gdn, allk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-cache", required=True)
    ap.add_argument("--candidate-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_gdn, base_all = count(pathlib.Path(args.baseline_cache))
    cand_gdn, cand_all = count(pathlib.Path(args.candidate_cache))

    if base_gdn < 0 or base_all == 0:
        rc, verdict = 2, "unusable: the baseline arm's Triton cache is missing or empty"
    elif base_gdn <= 1:
        rc, verdict = 4, (
            f"NOT EXERCISED: the baseline arm compiled {KERNEL} {base_gdn} time(s). The workload "
            f"never varied the batch shape, so the defect had no opportunity to cost anything and "
            f"the throughput ratio is measuring something else."
        )
    elif cand_gdn > base_gdn * CANDIDATE_MAX_FRACTION:
        rc, verdict = 5, (
            f"NOT THIS DEFECT: the baseline arm compiled {KERNEL} {base_gdn} times and the candidate "
            f"still compiles it {cand_gdn} times ({cand_gdn / base_gdn:.0%} of baseline, ceiling "
            f"{CANDIDATE_MAX_FRACTION:.0%}). Whatever recovered the throughput, it was not this "
            f"defect -- measured: both legitimate fixes reach 0.8%, while one agent's repair that "
            f"patched the cudagraph capture ladder instead reached 98% and still beat the gate by 39%."
        )
    else:
        rc, verdict = 0, (
            f"exercised and fixed: the baseline arm compiled {KERNEL} {base_gdn} times, "
            f"the candidate {cand_gdn} ({cand_gdn / base_gdn:.1%} of baseline)"
        )

    payload = {
        "rc": rc,
        "verdict": verdict,
        "kernel": KERNEL,
        "baseline_gdn_compiles": base_gdn,
        "baseline_total_compiles": base_all,
        "candidate_gdn_compiles": cand_gdn,
        "candidate_total_compiles": cand_all,
        "gdn_compile_reduction": (
            round(base_gdn / cand_gdn, 3) if cand_gdn and cand_gdn > 0 and base_gdn >= 0 else None
        ),
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(verdict)
    print(f"  baseline: {base_gdn} GDN / {base_all} total compiles")
    print(f"  candidate: {cand_gdn} GDN / {cand_all} total compiles")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
