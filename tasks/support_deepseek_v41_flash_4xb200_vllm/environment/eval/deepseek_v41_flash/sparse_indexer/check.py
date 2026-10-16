#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run the public examples through the four dependency operations and the two entry points, and report every
discrepancy with its continuous errors (README.md, "Examples and the checking script").

    python check.py [--cases examples.json] [--device cuda:0] [--output report.json]

It checks the small example records only, against the FP32 mathematics of reference.py and the expected selections
the records carry. A clean report establishes only the checks performed on these inputs; it is not a task verdict.
Exit status 1 means at least one discrepancy was found.

PyTorch and the engine are imported inside the functions, so the module imports with the standard library alone.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import contract  # noqa: E402
import reference  # noqa: E402

DTYPES = {"int8": "int8", "uint8": "uint8", "int32": "int32", "bfloat16": "bfloat16"}


def tensor(record: dict, device: str):
    """A tensor from its literal record: little-endian bytes, base64, with dtype and shape."""
    import torch

    raw = base64.b64decode(record["data"])
    dtype = getattr(torch, DTYPES[record["dtype"]])
    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype)
    return flat.view(*record["shape"]).to(device)


def packed_step(case: dict, device: str) -> contract.PackedStep:
    t = {k: tensor(v, device) for k, v in case["tensors"].items()}
    cand = t["candidate_storage"][:, : contract.NUM_CANDIDATES]  # a strided view of a wider buffer
    return contract.PackedStep(q=t["q"], q_scale=t["q_scale"], k_quant=t["k_quant"], k_scale=t["k_scale"],
                               weights=t["weights"], k_start=t["k_start"], k_end=t["k_end"], candidate_blocks=cand)


def paged_step(case: dict, device: str) -> contract.PagedStep:
    import torch

    t = {k: tensor(v, device) for k, v in case["tensors"].items()}
    page_kv, stride = int(case["page_kv"]), int(case["page_stride"])
    storage = t["page_storage"]
    kv = storage.as_strided((storage.shape[0], page_kv, 1, contract.PAGE_RECORD_BYTES),
                            (stride, contract.PAGE_RECORD_BYTES, contract.PAGE_RECORD_BYTES, 1))
    return contract.PagedStep(q=t["q"], q_scale=t["q_scale"], kv_cache=kv, weights=t["weights"],
                              context_lens=t["context_lens"], block_table=t["block_table"],
                              row_indices=t["row_indices"], candidate_blocks=t["candidate_blocks"])


def sparse_block_row(pool, visible: int, base: int):
    """The dependency ABI's sparse block ids for one row: the valid pool blocks inside the row's range, sorted, as
    absolute ids (``base`` = ``ks // 8`` for packed rows, 0 for paged rows), padded by repeating the last valid one."""
    import torch

    blocks = pool[(pool >= 0) & (pool < math.ceil(visible / contract.CANDIDATE_BLOCK))].to(torch.int64)
    blocks = torch.sort(blocks).values + base
    n = blocks.numel()
    row = torch.full((contract.NUM_CANDIDATES,), int(blocks[-1]) if n else base, dtype=torch.int32,
                     device=pool.device)
    row[:n] = blocks.to(torch.int32)
    return row, n


def check_selection(out_row, allowed, visible, row_scores, expected) -> dict:
    import torch

    rec = {"problems": reference.selection_problems(out_row, allowed, visible)}
    got = out_row[out_row >= 0].to(torch.int64)
    want = torch.tensor(expected, dtype=torch.int64, device=out_row.device)
    rec["agree_with_expected"] = int(torch.isin(got, want).sum())
    rec["expected"] = int(want.numel())
    delta = reference.SOURCE_DELTA * float(row_scores.abs().max()) if row_scores.numel() else 0.0
    rec["within_slack"] = reference.within_selection_slack(out_row, allowed, row_scores, delta)
    if not rec["within_slack"]:
        rec["problems"].append("an excluded position scores more than 2 delta above the smallest selected one")
    return rec


def run_packed(case: dict, device: str) -> dict:
    import torch

    s = packed_step(case, device)
    report = {"id": case["id"], "interface": contract.packed_problems(s), "rows": []}
    prefill, _ = contract.load_entry_points()
    out = contract.output_buffer(s.rows, device)
    try:
        contract.run_prefill(prefill, s, out)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        report["entry_point_error"] = f"{type(e).__name__}: {e}"
        out = None
    kd = reference.dequantize(s.k_quant, s.k_scale)
    dep = contract.load_dependency()
    for c in range(contract.NUM_CONSUMERS):
        qd = reference.dequantize(s.q[c], s.q_scale[c])
        for r in range(s.rows):
            ks, ke = int(s.k_start[r]), int(s.k_end[r])
            visible = ke - ks
            row_scores = reference.scores(qd[r : r + 1], s.weights[c, r : r + 1], kd[ks:ke])[0]
            allowed = reference.candidate_positions(s.candidate_blocks[r], visible)
            rec = {"consumer": c, "row": r}
            if out is not None:
                rec.update(check_selection(out[c, r], allowed, visible, row_scores, case["expected"][c][r]))
            report["rows"].append(rec)
        # the dependency operations directly, on this consumer's rows
        try:
            rows = [sparse_block_row(s.candidate_blocks[r], int(s.k_end[r] - s.k_start[r]),
                                     int(s.k_start[r]) // contract.CANDIDATE_BLOCK) for r in range(s.rows)]
            idx = torch.stack([row for row, _ in rows])
            meta = dep["get_sparse_mqa_logits_metadata"](s.k_start, s.k_end, s.k_quant.shape[0], idx, torch.int8,
                                                         contract.SPARSE_BLOCK_KV, use_unaligned_ks=True)
            logits = dep["fp8_fp4_sparse_mqa_logits"]((s.q[c], s.q_scale[c]), (s.k_quant, s.k_scale),
                                                      s.weights[c], meta, idx.shape[1], contract.SPARSE_BLOCK_KV,
                                                      use_unaligned_ks=True)
            torch.cuda.synchronize()
            errs = []
            for r, (row, n) in enumerate(rows):
                ks, ke = int(s.k_start[r]), int(s.k_end[r])
                tok = (row[:n].to(torch.int64)[:, None] * 8 + ks % 8
                       + torch.arange(8, device=row.device)).flatten()
                valid = tok < ke
                want = reference.scores(qd[r : r + 1], s.weights[c, r : r + 1], kd[tok[valid]])[0]
                errs.append(reference.row_error(logits[r, : tok.numel()][valid], want))
            report.setdefault("logits", []).append({"consumer": c, "p99": max(e[0] for e in errs),
                                                    "max": max(e[1] for e in errs)})
        except Exception as e:  # noqa: BLE001
            report.setdefault("dependency_error", []).append(f"consumer {c}: {type(e).__name__}: {e}")
    return report


def run_paged(case: dict, device: str) -> dict:
    import torch

    s = paged_step(case, device)
    report = {"id": case["id"], "interface": contract.paged_problems(s), "rows": []}
    _, decode = contract.load_entry_points()
    out = contract.output_buffer(s.rows, device)
    try:
        contract.run_decode(decode, s, out)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        report["entry_point_error"] = f"{type(e).__name__}: {e}"
        out = None
    dep = contract.load_dependency()
    page_kv = s.kv_cache.shape[1]
    keys = []
    for r in range(s.rows):
        values, words = reference.page_tokens(s.kv_cache, s.block_table[r], int(s.context_lens[r]))
        keys.append(reference.dequantize(values, words))
    for c in range(contract.NUM_CONSUMERS):
        qd = reference.dequantize(s.q[c, :, 0], s.q_scale[c, :, 0])
        for r in range(s.rows):
            visible = int(s.context_lens[r])
            row_scores = reference.scores(qd[r : r + 1], s.weights[c, r : r + 1], keys[r])[0]
            allowed = reference.candidate_positions(s.candidate_blocks[r], visible)
            rec = {"consumer": c, "row": r}
            if out is not None:
                rec.update(check_selection(out[c, r], allowed, visible, row_scores, case["expected"][c][r]))
            report["rows"].append(rec)
        try:
            rows = [sparse_block_row(s.candidate_blocks[r], int(s.context_lens[r]), 0) for r in range(s.rows)]
            idx = torch.stack([row for row, _ in rows])
            meta = dep["get_paged_sparse_mqa_logits_metadata"](s.context_lens, s.block_table, s.row_indices,
                                                               page_kv, idx, torch.int8, contract.SPARSE_BLOCK_KV)
            logits = dep["fp8_fp4_paged_sparse_mqa_logits"]((s.q[c], s.q_scale[c]), s.kv_cache, s.weights[c], meta,
                                                            idx.shape[1], contract.SPARSE_BLOCK_KV)
            torch.cuda.synchronize()
            errs = []
            for r, (row, n) in enumerate(rows):
                tok = (row[:n].to(torch.int64)[:, None] * 8 + torch.arange(8, device=row.device)).flatten()
                valid = tok < int(s.context_lens[r])
                want = reference.scores(qd[r : r + 1], s.weights[c, r : r + 1], keys[r][tok[valid]])[0]
                errs.append(reference.row_error(logits[r, : tok.numel()][valid], want))
            report.setdefault("logits", []).append({"consumer": c, "p99": max(e[0] for e in errs),
                                                    "max": max(e[1] for e in errs)})
        except Exception as e:  # noqa: BLE001
            report.setdefault("dependency_error", []).append(f"consumer {c}: {type(e).__name__}: {e}")
    return report


def discrepancies(report: dict) -> list[str]:
    out = [f"{report['id']}: {p}" for p in report["interface"]]
    for key in ("entry_point_error", "dependency_error"):
        if report.get(key):
            out.append(f"{report['id']}: {report[key]}")
    for rec in report["rows"]:
        for p in rec.get("problems", []):
            out.append(f"{report['id']} consumer {rec['consumer']} row {rec['row']}: {p}")
    for rec in report.get("logits", []):
        if not (rec["p99"] < reference.LOGIT_P99 and rec["max"] < reference.LOGIT_MAX):
            out.append(f"{report['id']} consumer {rec['consumer']}: logits p99 {rec['p99']:.4g}, "
                       f"max {rec['max']:.4g}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, default=HERE / "examples.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args(argv)
    import torch

    torch.cuda.set_device(a.device)
    records = json.loads(a.cases.read_text())
    reports = [run_packed(c, a.device) if c["kind"] == "packed" else run_paged(c, a.device)
               for c in records["checks"]]
    found = [d for r in reports for d in discrepancies(r)]
    for r in reports:
        logits = ", ".join(f"c{x['consumer']} p99 {x['p99']:.2e} max {x['max']:.2e}" for x in r.get("logits", []))
        print(f"{r['id']}: {len(r['rows'])} consumer rows; logits {logits or 'not computed'}")
    for d in found:
        print("DISCREPANCY " + d)
    print(f"{len(found)} discrepancies (public examples only; not a task verdict)")
    if a.output:
        a.output.write_text(json.dumps({"reports": reports, "discrepancies": found}, indent=1) + "\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
