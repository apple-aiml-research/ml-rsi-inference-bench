#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verdicts of the six component nodes, computed from an arm's
recorded outputs, in a process that never imports an implementation under test. The inputs are regenerated from
`kernel_cases` (and checked against the fixture manifest again); the expected values come from `kernel_reference`'s
float64 equations; every comparison is the published comparator (`check.compare`, the trusted mirror of the public
`kernels/check.py`, under README "Numerical tolerances")."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_cases as kc  # noqa: E402
import kernel_reference as ref  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("infb_kernel_check_v", HERE.parent / "eval" / "deepseek_v4_flash" /
                                               "kernels" / "check.py")
chk = importlib.util.module_from_spec(_spec)
sys.modules["infb_kernel_check_v"] = chk
_spec.loader.exec_module(chk)

ROWS_PER_CHUNK = 16


def _fail(problem: str) -> dict:
    return {"ok": False, "problems": [problem]}


def _typed(name: str, t, dtype, shape) -> dict | None:
    if not isinstance(t, torch.Tensor):
        return _fail(f"{name}: no tensor")
    if t.dtype != dtype:
        return _fail(f"{name}: dtype {t.dtype}, the interface returns {dtype}")
    if tuple(t.shape) != tuple(shape):
        return _fail(f"{name}: shape {tuple(t.shape)}, the interface returns {tuple(shape)}")
    return None


def _stat(name, got, want, policy, dtype, shape) -> dict:
    bad = _typed(name, got, dtype, shape)
    return bad or chk.compare(got, want, policy, expect_nonfinite=~torch.isfinite(want))




def verdict_bf16(case, t, o) -> dict:
    q, kv, idx, tl, sink = t["q"], t["kv"], t["indices"][:, 0, :].long(), t["topk_length"].long(), t["attn_sink"]
    n, k = idx.shape
    kv_len = kv.shape[0]
    valid = (idx >= 0) & (idx < kv_len) & (torch.arange(k).unsqueeze(0) < tl.unsqueeze(1))
    keys = kv[:, 0, :][idx.clamp(0, kv_len - 1)]
    r = ref.sparse_attention(q, keys, valid, sink, kc.SCALE)
    out_bad = _typed("output", o.get("out"), torch.bfloat16, (n, kc.H, kc.D))
    return {"output": out_bad or chk.compare(o["out"], r["out"], "bf16_out"),
            "max_logits": _stat("max_logits", o.get("max_logits"), r["max_logits"], "bf16_stat", torch.float32,
                                (n, kc.H)),
            "lse": _stat("lse", o.get("lse"), r["lse"], "bf16_stat", torch.float32, (n, kc.H))}


def packed_reference(case, t) -> dict:
    """The float64 attention of a packed case, keys decoded from the bytes of every present pool."""
    pages = kc.packed_pages(case)
    b, sq = t["q"].shape[:2]
    n = b * sq
    q = t["q"].view(n, kc.H, kc.D)
    outs, lses = [], []
    for a in range(0, n, ROWS_PER_CHUNK):
        rows = range(a, min(n, a + ROWS_PER_CHUNK))
        keys, valid = [], []
        for name, page in pages.items():
            idx = t[f"{name}.indices"].view(n, -1)[a: a + len(rows)].long()
            cap = idx.shape[-1]
            lens = t[f"{name}.lengths"].long().repeat_interleave(sq)[a: a + len(rows)]
            keys.append(ref.decode_model1(t[f"{name}.raw"], page, idx.clamp_min(0)))
            valid.append((idx >= 0) & (torch.arange(cap).unsqueeze(0) < lens.unsqueeze(1)))
        r = ref.sparse_attention(q[a: a + len(rows)], torch.cat(keys, 1), torch.cat(valid, 1), t["attn_sink"], kc.SCALE)
        outs.append(r["out"])
        lses.append(r["lse"])
    return {"out": torch.cat(outs), "lse": torch.cat(lses)}


def verdict_packed(case, t, o) -> dict:
    b, sq = t["q"].shape[:2]
    r = packed_reference(case, t)
    out_bad = _typed("output", o.get("out"), torch.bfloat16, (b, sq, kc.H, kc.D))
    want_lse = r["lse"].view(b, sq, kc.H).transpose(1, 2).contiguous()
    return {"output": out_bad or chk.compare(o["out"].view(b * sq, kc.H, kc.D), r["out"], "packed_out"),
            "lse": _stat("lse", o.get("lse"), want_lse, "packed_lse", torch.float32, (b, kc.H, sq))}




def _emissions(rows, keys, expected_keys, x_of, ape, ratio, hd, label) -> dict:
    """The recorded emissions against the float64 history: exactly the expected (request, position) events, each row
    within the values policy."""
    got_keys = [tuple(k) for k in keys.tolist()]
    if sorted(got_keys) != sorted(expected_keys):
        missing = sorted(set(expected_keys) - set(got_keys))[:8]
        extra = sorted(set(got_keys) - set(expected_keys))[:8]
        return _fail(f"{label}: emitted events differ from the history's (missing {missing}, unexpected {extra})")
    if len(got_keys) != len(set(got_keys)):
        return _fail(f"{label}: an event was emitted twice")
    bad = _typed(label, rows, torch.float32, (len(got_keys), hd))
    if bad:
        return bad
    want = torch.stack([ref.compress_emissions(x_of(i), ape, ratio, hd, [p])[0] for i, p in got_keys]) if got_keys \
        else torch.empty(0, hd, dtype=torch.float64)
    return chk.compare(rows, want, "values")


def _untouched(u, label) -> dict:
    if not isinstance(u, torch.Tensor) or u.numel() != 2:
        return _fail(f"{label}: no guard record")
    g, idle = int(u[0]), int(u[1])
    if g or idle:
        return _fail(f"{label}: {g} guard element(s) and {idle} element(s) of windows no request addresses changed")
    return {"ok": True, "problems": []}


def verdict_state(case, t, o) -> dict:
    p = case.p
    ratio, hd, _, _ = kc.FAMILIES[p["family"]]
    totals = [a + p["extend"] + p["decode_steps"] for a in p["prefixes"]]
    expected = [(i, pos) for i, n in enumerate(totals) for pos in range(n) if (pos + 1) % ratio == 0]
    checks = {}
    for mode in ("oneshot", "chunked"):
        checks[f"{mode}.emissions"] = _emissions(o.get(f"{mode}.rows"), o[f"{mode}.keys"], expected,
                                                 lambda i: t[f"x{i}"], t["ape"], ratio, hd, mode)
        checks[f"{mode}.storage"] = _untouched(o.get(f"{mode}.untouched"), mode)
    return checks


def verdict_branch(case, t, o) -> dict:
    p = case.p
    ratio, hd, _, _ = kc.FAMILIES[p["family"]]
    hist = torch.cat([t["prefix"], t["suffix"][: p["accepted"]], t["new"]])
    expected = [(0, pos) for pos in range(hist.shape[0]) if (pos + 1) % ratio == 0]
    keys = torch.stack([torch.zeros_like(o["keys"]), o["keys"]], dim=1) if o["keys"].numel() else o["keys"].view(-1, 2)
    return {"emissions": _emissions(o.get("rows"), keys, expected, lambda i: hist, t["ape"], ratio, hd, "branch"),
            "storage": _untouched(o.get("untouched"), "branch")}




def _freqs_real() -> torch.Tensor:
    return torch.view_as_real(kc.rope_table()).flatten(-2)


def verdict_normrope(case, t, o) -> dict:
    p = case.p
    ratio, hd, _, _ = kc.FAMILIES[p["family"]]
    reqs = kc.normrope_requests(ratio, p["phase"])
    x = t["x"]
    bad = _typed("rows", o.get("y"), torch.float32, tuple(x.shape))
    if bad:
        return {"rows": bad}
    rows, pos = [], []
    if p["phase"] == "prefill":
        j = 0
        for a, e in reqs:
            for off in range(e):
                if (a + off + 1) % ratio == 0:
                    rows.append(j)
                    pos.append(a + off + 1 - ratio)
                j += 1
    else:
        for j, (n, _) in enumerate(reqs):
            if n % ratio == 0:
                rows.append(j)
                pos.append(n - ratio)
    want = ref.norm_rope(x[rows], kc.norm_weight(hd), kc.NORM_EPS, _freqs_real(), pos)
    return {"rows": chk.compare(o["y"][rows], want, "values")}


def _store_bytes(typ: str, page: int, pool, pool0, rows_x, slots: list[int], label: str) -> dict:
    """Exact destination bytes for `rows_x` stored at `slots`; every other byte of the pool unchanged except the
    reserved padding slot 0 and the MODEL1 scale row's padding byte."""
    idx = torch.tensor(slots, dtype=torch.int64)
    problems = []
    if typ == "flashmla":
        got = ref.read_model1_rows(pool, page, idx)
        want = ref.store_flashmla_expected(rows_x)
        nd = int((got["data"] != want["data"]).sum())
        ns = int((got["scale"] != want["scale"]).sum())
        if nd:
            problems.append(f"{label}: {nd} data byte(s) (E4M3 values or BF16 rotary bits) differ from the store rule")
        if ns:
            problems.append(f"{label}: {ns} E8M0 scale byte(s) differ from the store rule")
        spans = ref.row_byte_spans_model1(page, torch.tensor(slots + [0]))
    else:
        got = ref.read_indexer_rows(pool, page, idx)
        want = ref.store_indexer_expected(rows_x)
        nd, ties = chk.indexer_payload_mismatches(got["data"], rows_x, want["scale"])
        ulp = (got["scale"].view(torch.int32).to(torch.int64) - want["scale"].view(torch.int32).to(torch.int64)).abs()
        if nd:
            problems.append(f"{label}: {nd} FP8 byte(s) differ from the store rule ({ties} value(s) at a rounding tie "
                            f"accept either neighbour)")
        if int(ulp.max()) > 1:
            problems.append(f"{label}: an FP32 scale is {int(ulp.max())} ULP from the formula (one allowed)")
        spans = ref.row_byte_spans_indexer(page, torch.tensor(slots + [0]))
    allowed = torch.zeros_like(pool, dtype=torch.bool)
    for pg, a, b in spans:
        allowed[pg, a:b] = True
    changed = int(((pool != pool0) & ~allowed).sum())
    if changed:
        problems.append(f"{label}: {changed} byte(s) outside the destinations and the padding slot changed")
    return {"ok": not problems, "problems": problems, "rows": len(slots)}


def verdict_store(case, t, o) -> dict:
    p = case.p
    pool0 = t["pool"]
    bad = _typed("pool", o.get("pool"), torch.uint8, tuple(pool0.shape))
    if bad:
        return {"store": bad}
    dest = t["dest"].tolist()
    live = [(j, s) for j, s in enumerate(dest) if s != 0]
    x = t["x"] if p["type"] == "flashmla" else t["x"].to(torch.bfloat16).float()
    return {"store": _store_bytes(p["type"], p["page"], o["pool"], pool0, x[[j for j, _ in live]],
                                  [s for _, s in live], "store")}


def verdict_compose(case, t, o) -> dict:
    ratio, hd, typ, page = kc.FAMILIES[case.p["family"]]
    _, _, emits = kc.compose_layout(case.p["family"])
    post, pool = o.get("post"), o.get("pool")
    bad = _typed("post", post, torch.float32, (len(emits), hd)) or _typed("pool", pool, torch.uint8,
                                                                         tuple(t["pool"].shape))
    if bad:
        return {"compose": bad}
    pre = ref.compress_emissions(t["x"], t["ape"], ratio, hd, emits)
    want = ref.norm_rope(pre, kc.norm_weight(hd), kc.NORM_EPS, _freqs_real(), [p + 1 - ratio for p in emits])
    stored_from = post if typ == "flashmla" else post.to(torch.bfloat16).float()
    return {"intermediate": chk.compare(post, want, "values"),
            "stored": _store_bytes(typ, page, pool, t["pool"], stored_from, t["dest"].tolist(), "stored")}


VERDICTS = {"bf16": verdict_bf16, "packed": verdict_packed, "state": verdict_state, "branch": verdict_branch,
            "normrope": verdict_normrope, "store": verdict_store, "compose": verdict_compose}




def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def correctness_verdict(node: str, outputs: Path, records: Path, manifest: dict) -> dict:
    """{'status': 'pass'|'fail'|'fault', 'cases': [...], 'problems': [...]}"""
    recs = read_records(records)
    faults = [r for r in recs if r.get("status") == "verifier_fault"]
    if faults:
        return {"status": "fault", "problems": [f.get("failure", "") for f in faults], "cases": []}
    worker = {r["case"]: r for r in recs if "case" in r}
    head = [r for r in recs if r.get("worker") == "start"]
    load_fail = [r for r in head if r.get("status") in ("error", "unavailable")]
    out, problems = [], []
    for case in kc.cases(node):
        t = kc.inputs(case)
        problem = kc.check_digest(case, t, manifest)
        if problem:
            return {"status": "fault", "problems": [problem], "cases": out}
        rec = {"case": case.id, "kind": case.kind}
        w = worker.get(case.id)
        if load_fail:
            rec.update(ok=False, problems=[f"the tree's interfaces did not load: {load_fail[0].get('failure')}"])
        elif w is None:
            rec.update(ok=False, problems=["the worker recorded no result for this case (it stopped before it)"])
        elif w.get("status") != "ran":
            rec.update(ok=False, problems=[f"{w.get('status')}: {w.get('error')}"])
        else:
            try:
                o = torch.load(outputs / (case.id.replace("/", "__") + ".pt"), weights_only=True, map_location="cpu")
                checks = VERDICTS[case.kind](case, t, o)
                rec["checks"] = checks
                rec["ok"] = all(c.get("ok") for c in checks.values())
                rec["problems"] = [x for c in checks.values() for x in c.get("problems", [])]
            except Exception as exc:  # noqa: BLE001 - a malformed output is the arm's failure
                rec.update(ok=False, problems=[f"outputs unreadable or malformed: {type(exc).__name__}: {exc}"[:2000]])
            rec.update(elapsed_s=w.get("elapsed_s"), peak_allocated_bytes=w.get("peak_allocated_bytes"))
        out.append(rec)
        if not rec["ok"]:
            problems.append(f"{case.id}: " + "; ".join(rec["problems"])[:600])
    return {"status": "pass" if not problems else "fail", "cases": out, "problems": problems}


def efficiency_verdict(node: str, cand_records: Path, ref_records: Path, rho: float) -> dict:
    phase = kc.EFFICIENCY_NODES[node]
    fams = kc._timing_families(phase)

    def final(path):
        recs = read_records(path)
        by = {}
        for r in recs:
            if "family" in r and r.get("status") not in ("started", "profiled", "profile_error"):
                by[r["family"]] = r
        head = [r for r in recs if r.get("worker") == "start" and r.get("status") not in (None, "loaded")]
        return by, head

    cand, cand_head = final(cand_records)
    refs, ref_head = final(ref_records)
    out, problems, faults = [], [], []
    if ref_head or any(r.get("status") == "verifier_fault" for r in read_records(cand_records)):
        faults += [h.get("failure", "") for h in ref_head]
        faults += [r.get("failure", "") for r in read_records(cand_records) if r.get("status") == "verifier_fault"]
    for fam in fams:
        r, c = refs.get(fam), cand.get(fam)
        rec = {"family": fam}
        if r is None or r.get("status") != "measured":
            faults.append(f"{fam}: the reference arm's measurement is {'missing' if r is None else r.get('status')}")
            rec.update(ok=None)
        elif c is None or c.get("status") != "measured":
            why = (cand_head[0].get("failure") if cand_head else None) or (c or {}).get("failure") or "no record"
            rec.update(ok=False, problems=[f"not measured: {(c or {}).get('status', 'missing')}: {why}"[:600]])
        elif r.get("seed") != c.get("seed"):
            faults.append(f"{fam}: the arms measured different inputs (seed {r.get('seed')} vs {c.get('seed')})")
            rec.update(ok=None)
        else:
            rw, rg = r["wall_s"] / c["wall_s"], r["gpu_s"] / c["gpu_s"]
            rec.update(ratio_wall=rw, ratio_gpu=rg, reference={"wall_s": r["wall_s"], "gpu_s": r["gpu_s"]},
                       candidate={"wall_s": c["wall_s"], "gpu_s": c["gpu_s"]}, ok=(rw >= rho and rg >= rho),
                       candidate_peak_allocated_bytes=c.get("peak_allocated_bytes"),
                       elapsed_s=c.get("elapsed_s"))
            if not rec["ok"]:
                rec["problems"] = [f"{fam}: ratio wall {rw:.3f}, GPU {rg:.3f} (both must be >= the private bound)"]
        out.append(rec)
        if rec.get("ok") is False:
            problems += rec["problems"]
    if faults:
        return {"status": "fault", "families": out, "problems": faults}
    return {"status": "pass" if not problems else "fail", "families": out, "problems": problems}


def main(argv: list[str]) -> int:
    node, outputs, records, manifest = argv[1], Path(argv[2]), Path(argv[3]), json.loads(Path(argv[4]).read_text())
    v = correctness_verdict(node, outputs, records, manifest)
    text = json.dumps(v, indent=1, default=str)
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(text + "\n")
    print(json.dumps({"node": node, "status": v["status"], "problems": v["problems"][:20]}, default=str))
    return {"pass": 0, "fail": 1, "fault": 3}[v["status"]]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
