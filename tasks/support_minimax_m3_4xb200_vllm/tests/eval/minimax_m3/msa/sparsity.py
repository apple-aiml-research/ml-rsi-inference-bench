#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Sparsity realized (README "Context scaling"): the attention operators' cost per query must not grow with context
length beyond the selected blocks.

Two ratios, BF16 KV cache, GPU-time medians by the timing protocol of ``measure.py``, with the reference selections
of the same inputs (16 blocks per query):

  R_dec = t(attend_decode, 16 requests at 262,144 tokens) / t(attend_decode, 16 requests at 16,384 tokens)
          (one query per request, at the last position)
  R_pre = t(attend_prefill, 8,192 queries ending at 262,144) / t(attend_prefill, 8,192 queries ending at 32,768)

A sparse implementation's per-query cost is O(k * block), flat in context; attention over all visible tokens grows
with the context. This tool reports the two ratios; the acceptance bounds are the grader's.

The poisoned cache (``POISON``, ``poisoned_case``): the attention operators read only the causally visible tokens of
the selected blocks, so after a case is installed, every page no query of the case selects is overwritten with NaN
in K and in V (the index keys are untouched; selection reads every block's index keys by definition) and the case is
called once. A sparse implementation never reads those pages: its output still meets the case's correctness policy
(check.attention_policy). An implementation whose arithmetic touches any of them returns NaN rows (0 x NaN = NaN) or
changed values. FP8 pages are poisoned with 0x7F, E4M3's NaN; an implementation that reads that byte as a finite value
and gives it zero weight is not seen by the FP8 geometry, which is why the BF16 geometries carry the check.

  python /app/eval/minimax_m3/run.py msa-sparsity --output /tmp/s.json [--poison]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import check, measure

PAIRS = {
    "decode": ("attend_decode_bf16", {"requests": 16, "seq_len": 262_144}, {"requests": 16, "seq_len": 16_384}),
    "prefill": ("attend_prefill_bf16", {"context": 262_144 - 8_192, "queries": 8_192},
                {"context": 32_768 - 8_192, "queries": 8_192}),
}


def measure_pair(module, which: str, seed: int, device="cuda") -> dict:
    family, long_w, short_w = PAIRS[which]
    phase = measure.FAMILIES[family][0]
    out = {}
    for label, w in (("long", long_w), ("short", short_w)):
        kw = {"prefill": w} if phase == "prefill" else {"decode": w}
        case = measure.workload_case(family, seed, device, **kw)
        rec = measure.measure_case(module, case, measure.CALLS[phase], device)
        rec["workload"] = w
        out[label] = rec
        if rec["status"] != "measured":
            return {"family": family, "status": rec["status"], **out}
    ratio = out["long"]["gpu_ms_median"] / out["short"]["gpu_ms_median"]
    return {"family": family, "status": "measured", "ratio": ratio, **out}


#: The poisoned-cache geometries: the decode pair's two lengths and the efficiency workload's in both KV dtypes, and a
#: prefill of 256 queries at the end of a 131,072-token request (with 8,192 queries every block is some query's).
POISON = {
    "decode_16k": ("attend_decode_bf16", {"requests": 16, "seq_len": 16_384}),
    "decode_64k_bf16": ("attend_decode_bf16", {"requests": 16, "seq_len": 65_536}),
    "decode_64k_fp8": ("attend_decode_fp8", {"requests": 16, "seq_len": 65_536}),
    "decode_256k": ("attend_decode_bf16", {"requests": 16, "seq_len": 262_144}),
    "prefill_256q_bf16": ("attend_prefill_bf16", {"context": 130_816, "queries": 256}),
    "prefill_256q_fp8": ("attend_prefill_fp8", {"context": 130_816, "queries": 256}),
}


def selected_pages(case: check.Case) -> torch.Tensor:
    """The physical pages some query of the case selects."""
    sel = case.selection.long()
    table = case.block_table.long()
    if case.prefill:
        return table[0, sel[sel >= 0].unique()].unique()
    rows = torch.arange(sel.shape[0], device=sel.device)[:, None].expand_as(sel)
    ok = sel >= 0
    return table[rows[ok], sel[ok]].unique()


def poison_unselected(module, cache, case: check.Case) -> int:
    """Overwrite K and V of every page no query of the case selects (NaN; FP8 0x7F) through the contract's views."""
    k, v, _ = module.cache_views(cache)
    keep = torch.zeros(case.num_pages, dtype=torch.bool, device=case.selection.device)
    keep[selected_pages(case)] = True
    idx = (~keep).nonzero().squeeze(1)
    for view in (k, v):
        if view.element_size() == 1:
            view.view(torch.uint8)[idx] = 0x7F
        else:
            view[idx] = float("nan")
    torch.cuda.synchronize()
    return int(idx.numel())


def poisoned_case(module, name: str, seed: int, device="cuda") -> dict:
    """One poisoned-cache geometry: install, poison the unselected pages, call once, apply the case's structure and
    correctness policy. ``pass`` is the verdict; a candidate exception is its failure, recorded verbatim."""
    family, w = POISON[name]
    phase = measure.FAMILIES[family][0]
    case = measure.workload_case(family, seed, device, **{phase: w})
    rec = {"geometry": name, "family": family, "workload": w, "pages": case.num_pages, "status": None, "pass": False}
    try:
        cache, problems = check.install(module, case, device)
        if problems:
            rec.update(status="candidate_fail", structure=problems)
            return rec
        rec["poisoned_pages"] = poison_unselected(module, cache, case)
        out = check.call(module, cache, case)
        torch.cuda.synchronize()
        problems = check.structure(case, out)
    except Exception as exc:  # the candidate's own exception is its failure
        rec.update(status="candidate_fail", exception=f"{type(exc).__name__}: {exc}"[:2000])
        return rec
    if problems:
        rec.update(status="candidate_fail", structure=problems)
        return rec
    verdict = check.attention_policy(case, out)
    rec.update(policy=verdict, status="pass" if verdict["pass"] else "candidate_fail", **{"pass": bool(verdict["pass"])})
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", type=Path, default=check.CONTRACT)
    ap.add_argument("--seed", type=int, default=measure.PUBLIC_SEED)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--poison", action="store_true", help="also run the poisoned-cache geometries")
    args = ap.parse_args(argv)
    module = check.load_contract(args.contract)
    res = {k: measure_pair(module, k, args.seed) for k in ("decode", "prefill")}
    out = {"contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
           "R_dec": res["decode"].get("ratio"), "R_pre": res["prefill"].get("ratio"), "pairs": res,
           "note": "the candidate's own ratios; the grader applies its own bounds"}
    if args.poison:
        out["poisoned"] = {name: poisoned_case(module, name, args.seed) for name in POISON}
    args.output.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"R_dec {out['R_dec']}  R_pre {out['R_pre']}")
    for name, rec in out.get("poisoned", {}).items():
        print(f"poisoned {name}: {rec['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
