#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'Port-agnostic evidence that an FP8 SD3 DiT really computes in FP8: the instruments, as testable functions.'


from __future__ import annotations

import ast
import math
import re
from collections import Counter

BLOCK = 128
FP8_DTYPES = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
SIXTEEN_BIT = ("bfloat16", "float16")
GEMM_OP_RE = re.compile(r"^aten\.(mm|addmm|bmm|baddbmm|matmul|linear|_scaled_mm)\.")


SD3_Q_SHAPES = frozenset({(1536, 1536), (6144, 1536), (1536, 6144)})
SD3_Q_SHAPES_FUSED = frozenset({(4608, 1536)})


def q_shapes_for(tp: int = 1, fused_qkv: bool = True) -> frozenset:
    base = set(SD3_Q_SHAPES) | (set(SD3_Q_SHAPES_FUSED) if fused_qkv else set())
    out = set()
    for n, k in base:
        out.add((n, k))
        if tp > 1:
            if n % tp == 0:
                out.add((n // tp, k))
            if k % tp == 0:
                out.add((n, k // tp))
    return frozenset(out)


KERNEL_RULES = (

    ("fp8_aux", "act_quant_per_token_group", r"per_token_group_quant"),
    ("fp8_aux", "act_quant_flashinfer_scale_1x128", r"scale_1x128_kernel"),
    ("fp8_aux", "act_quant_other", r"per_token_quant|scaled_fp8_quant|quant_8bit|_per_token_group_quant_fp8"),
    ("fp8_aux", "fp8_cast", r"float8_copy"),
    ("fp8_aux", "fp8_fill", r"FillFunctor<c10::Float8"),

    ("fp8_gemm", "deepgemm_sm90", r"deep_gemm::sm90_fp8_gemm"),
    ("fp8_gemm", "deepgemm_flashinfer", r"deep_gemm::fp8_gemm_kernel"),
    ("fp8_gemm", "cutlass_blockwise_fp8", r"BlockwiseFP8"),


    ("fp8_gemm", "cutlass_sm90_fp8_gmma", r"cutlass.*MMA_\d+x\d+x\d+_F32E[45]M[23]E[45]M[23]"),
    ("fp8_gemm", "triton_w8a8_block", r"^_w8a8_block_fp8_matmul"),
    ("fp8_gemm", "cublaslt_fp8", r"^nvjet_qq"),
    ("fp8_gemm", "cublas_xmma_e4m3", r"sm90_xmma_gemm_e4m3"),

    ("w8a16_gemm", "marlin", r"marlin::Marlin<"),

    ("gemm16", "cublaslt_16bit", r"^nvjet_tst_"),
    ("gemm16", "cublas_xmma_16bit", r"sm\d+_xmma_gemm_(bf16|f16)"),
    ("gemm16", "ampere_16bit", r"^ampere_(bf16|fp16|h)\w*gemm"),

    ("attention", "flash", r"flash_fwd|flash_attn|fmha|FlashAttn"),
)
_KERNEL_RX = tuple((cat, fam, re.compile(rx)) for cat, fam, rx in KERNEL_RULES)
_GEMMISH = re.compile(r"gemm|Gemm|GEMM|matmul|Matmul|_mma|wgmma|hgmma|qgmma", re.I)


def classify_kernel(name: str) -> tuple[str, str]:
    """-> (category, family). category in fp8_aux / fp8_gemm / w8a16_gemm / gemm16 / attention /
    unclassified_gemm / other. An unrecognised kernel whose name looks like a GEMM is 'unclassified_gemm', which a
    gate must REPORT and never count as FP8 compute."""
    for cat, fam, rx in _KERNEL_RX:
        if rx.search(name):
            return cat, fam
    if _GEMMISH.search(name):
        return "unclassified_gemm", "unknown"
    return "other", "other"


def kernel_census(kernel_counts: dict) -> dict:
    """kernel_counts: {kernel name: launch count} for one forward. Returns counts per category and family, and the
    unclassified GEMM-looking names. Unknown kernels need direct evidence before receiving execution credit."""
    by_cat, by_fam, unclassified = Counter(), Counter(), {}
    for name, cnt in kernel_counts.items():
        n = int(cnt["count"] if isinstance(cnt, dict) else cnt)
        cat, fam = classify_kernel(name)
        by_cat[cat] += n
        by_fam[f"{cat}:{fam}"] += n
        if cat == "unclassified_gemm":
            unclassified[name[:200]] = n
    return {"by_category": dict(by_cat), "by_family": dict(by_fam), "unclassified_gemm": unclassified,
            "fp8_gemm_launches": by_cat.get("fp8_gemm", 0), "w8a16_launches": by_cat.get("w8a16_gemm", 0)}


def parse_legacy_gemm_key(key: str) -> dict:
    """Parse the legacy 'op|dt1,dt2,...|[(s1), (s2), ...]' record into op, dtypes and shapes."""
    op, dts, shapes = key.split("|", 2)
    sh = ast.literal_eval(shapes)
    return {"op": op, "dtypes": [d for d in dts.split(",") if d], "shapes": [list(s) for s in sh]}


def _nk_m(op: str, dtypes, shapes):
    """(N, K, M_total, kind) of one GEMM-like record, or None. kind in fp8 / 16bit / 32bit / other."""
    fp8 = [i for i, d in enumerate(dtypes) if d in FP8_DTYPES and i < len(shapes) and len(shapes[i]) == 2]
    if len(fp8) >= 2:
        a, b = shapes[fp8[0]], shapes[fp8[1]]
        m, k = a
        if b[1] == k:
            n = b[0]
        elif b[0] == k:
            n = b[1]
        else:
            return None
        return n, k, m, "fp8"
    base = op.split(".")[1] if op.startswith("aten.") else ""
    mats = [(d, s) for d, s in zip(dtypes, shapes) if len(s) in (2, 3)]
    if base in ("mm", "addmm", "bmm", "baddbmm"):
        if len(mats) < 2:
            return None
        (da, a), (_, b) = mats[-2], mats[-1]
        if len(a) == 3 and len(b) == 3:
            m, k, n = a[0] * a[1], a[2], b[2]
        elif len(a) == 2 and len(b) == 2:
            m, k, n = a[0], a[1], b[1]
        else:
            return None
        kind = "16bit" if da in SIXTEEN_BIT else ("32bit" if da == "float32" else "other")
        return n, k, m, kind
    if base == "linear" and len(mats) >= 2:
        (da, x), (_, w) = mats[0], mats[1]
        m = math.prod(x[:-1])
        kind = "16bit" if da in SIXTEEN_BIT else ("32bit" if da == "float32" else "other")
        return w[0], w[1], m, kind
    return None


def op_recorder(on_fp8=None):
    'A TorchDispatchMode that records every GEMM-like op (aten GEMMs and any op with two float8 matrix operands,'
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class OpRecorder(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            tens = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
            n_fp8 = sum(1 for t in tens if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and t.dim() == 2)
            if GEMM_OP_RE.match(name) or n_fp8 >= 2:
                key = (name, tuple(str(t.dtype).replace("torch.", "") for t in tens),
                       tuple(tuple(t.shape) for t in tens))
                self.counts[key] += 1
            res = func(*args, **kwargs)
            if on_fp8 is not None and n_fp8 >= 2 and fp8_gemm_like(name, tens):
                try:
                    on_fp8(name, tens, res)
                except Exception:  # noqa: BLE001 - the recorder must never change what the model computes
                    pass
            return res

        def records(self):
            return [{"op": op, "dtypes": list(dts), "shapes": [list(sh) for sh in shs], "count": c}
                    for (op, dts, shs), c in self.counts.items()]

    return OpRecorder()


def fp8_gemm_like(name: str, tensors) -> bool:
    """Is this call an FP8 GEMM: two float8 matrix operands that share K ([M, K] x [N, K] or [K, N]); an aten
    op only when it is a GEMM by name (a float8 copy_, cat or clone of weights is not a GEMM)? `tensors` are torch
    tensors or {"dtype", "shape"} dicts (a Triton launch's recorded arguments)."""
    def meta(t):
        if isinstance(t, dict):
            return str(t.get("dtype")), list(t.get("shape") or [])
        return str(t.dtype).replace("torch.", ""), list(t.shape)
    if name.startswith("aten.") and not GEMM_OP_RE.match(name):
        return False
    mats = [sh for dt, sh in map(meta, tensors) if dt in FP8_DTYPES and len(sh) == 2]
    if len(mats) < 2:
        return False
    a, b = mats[0], mats[1]
    return a[1] in (b[0], b[1])


def _asset_list(v) -> list:
    """ptr_to_asset's value: the asset module(s) a held float8 tensor is -- a list (a fused tensor's slabs), or one
    name; never iterated character by character."""
    return [v] if isinstance(v, str) else list(v or ())


def direct_channel_evidence(rec: dict, ptr_to_asset: dict) -> dict | None:
    """Read result-consumption evidence from the per-layer fp8_direct records."""


    d = rec.get("fp8_direct")
    if not isinstance(d, dict):
        return None
    out = {"recorded": True, "layers_with_events": 0, "evidenced_layers": 0, "evidenced_layer_names": [],
           "evidenced_asset_modules": [], "by_channel": {}, "show_layers": [], "unseen_layers": [],
           "weight_not_asset_layers": [],
           "events_total": int(d.get("events_total") or 0),
           "events_outside_layers": int(d.get("events_outside_layers") or 0)}
    covered = set()
    for L, s in sorted((d.get("layers") or {}).items()):
        if not s.get("events"):
            continue
        out["layers_with_events"] += 1
        if not s.get("consumed"):
            (out["unseen_layers"] if s.get("io_seen") is False else out["show_layers"]).append(L)
            continue
        wp = s.get("weight_ptr")
        if wp is None or int(wp) not in ptr_to_asset:
            out["weight_not_asset_layers"].append(L)
            continue
        out["evidenced_layers"] += 1
        out["evidenced_layer_names"].append(L)
        covered.update(_asset_list(ptr_to_asset[int(wp)]))
        ch = str(s.get("consumed_channel") or "?")
        out["by_channel"][ch] = out["by_channel"].get(ch, 0) + 1
    out["evidenced_asset_modules"] = sorted(covered)
    return out


def tp_row_sum_reading(crosses: list) -> str:
    """A TP row-parallel layer's output, read on every rank at the same positions (the tap's "cross" sample): whether
    every rank holds the same output and it IS the ranks' FP8 GEMM results summed, with the layer's bias once (B0's
    RowParallelLinear adds it in tp rank 0's epilogue): 'sum_of_rank_results' (in rank order in the result's dtype,
    one float32 sum rounded once, or the bias added after the reduction), else 'missing' (a rank made no consumed-
    shaped GEMM), 'positions_differ', 'ranks_disagree' or 'different'."""
    import torch
    if not crosses or any(c is None for c in crosses):
        return "missing"
    if any(c["idx"] != crosses[0]["idx"] for c in crosses):
        return "positions_differ"
    ydt = getattr(torch, crosses[0].get("y_dtype") or "float32", torch.float32)
    cdt = getattr(torch, crosses[0].get("c_dtype") or "float32", torch.float32)
    ys = [torch.tensor(c["y"], dtype=torch.float32).to(ydt) for c in crosses]
    if any(not torch.equal(ys[0], y) for y in ys[1:]):
        return "ranks_disagree"
    cs = [torch.tensor(c["c"], dtype=torch.float32).to(cdt) for c in crosses]
    b = crosses[0].get("bias")
    bt = torch.tensor(b, dtype=torch.float32).to(cdt) if b is not None else torch.zeros_like(cs[0])
    first = cs[0] + bt
    for c in cs[1:]:
        first = first + c
    after = cs[0]
    for c in cs[1:]:
        after = after + c
    for cand in (first, (sum(c.float() for c in cs) + bt.float()).to(cdt), after + bt):
        if torch.equal(ys[0], cand.to(ydt)):
            return "sum_of_rank_results"
    return "different"


def tp_flashinfer_row_sum(recs: list, row_parallel: set) -> dict:
    """{row-parallel layer: tp_row_sum_reading} over the ranks' FlashInfer `cross` samples (the tap reads each rank's
    call result when it returns and the layer's output at the same positions). A TP row-parallel layer's output is the
    ranks' results summed, so no single rank can read it against its own call."""
    reads = {}
    for L in sorted(row_parallel):
        per = [((r or {}).get("flashinfer_layers") or {}).get(L) for r in recs]
        if not any(per):
            continue
        reads[L] = tp_row_sum_reading([(p or {}).get("cross") for p in per])
    return reads


def with_tp_flashinfer_provenance(rec: dict, reads: dict) -> dict:
    """A copy of one rank's tap record whose row-parallel FlashInfer layers carry the cross-rank reading as their
    provenance: 'sum_of_rank_results' (credited), 'unread_missing' / 'unread_positions_differ' (no reading: no credit,
    no failure), else 'row_sum_<reading>' (observed: the output is not the ranks' results summed)."""
    layers = dict(rec.get("flashinfer_layers") or {})
    for L, r in reads.items():
        if L not in layers:
            continue
        meta = dict(layers[L])
        meta["provenance_rank_local"] = meta.get("provenance")
        meta["provenance"] = r if r == "sum_of_rank_results" else (
            f"unread_{r}" if r in ("missing", "positions_differ") else f"row_sum_{r}")
        layers[L] = meta
    return dict(rec, flashinfer_layers=layers)


def tp_direct_evidence(recs: list, ptr_to_asset: list, row_parallel: set) -> list:
    """Result-consumption evidence under TP, one direct_channel_evidence dict per rank (None for a missing record). A
    column-parallel layer is judged on its own rank, as at TP=1. A ROW-parallel layer's output is the ranks' partial
    results summed, so it is judged across ranks: consumed on a rank when that rank's FP8 GEMM used its own shard and
    rows AND the output every rank holds is the sum of the ranks' results (tp_row_sum_reading) -- so every rank's
    contribution must reach the output."""
    reads = {}
    for L in sorted(row_parallel):
        reads[L] = tp_row_sum_reading([(((r or {}).get("fp8_direct") or {}).get("layers") or {}).get(L, {}).get("cross")
                                       for r in recs])
    tally = {v: list(reads.values()).count(v) for v in sorted(set(reads.values()))}
    outs = []
    for k, rec in enumerate(recs):
        d = (rec or {}).get("fp8_direct")
        if not isinstance(d, dict):
            outs.append(None)
            continue
        layers = {}
        for L, s in (d.get("layers") or {}).items():
            if L in row_parallel:
                s = dict(s)
                own = any(r.get("on_layer_weight") and r.get("on_layer_rows") for r in s.get("readings") or [])
                s["consumed"] = bool(own and reads.get(L) == "sum_of_rank_results")
                s["consumed_channel"] = (s.get("cross") or {}).get("channel") if s["consumed"] else None
            layers[L] = s
        ev = direct_channel_evidence({"fp8_direct": dict(d, layers=layers)}, ptr_to_asset[k])
        ev["tp_row_sum"] = dict(tally, row_parallel_layers=len(reads))
        outs.append(ev)
    return outs


def gemm_census(records, q_shapes=SD3_Q_SHAPES, min_tokens: int = 64) -> dict:
    'gemm census.'
    out = Counter()
    flops = Counter()
    unparsed = Counter()
    for r in records:
        if isinstance(r, tuple):
            key, cnt = r
            r = dict(parse_legacy_gemm_key(key), count=cnt)
        cnt = int(r.get("count", 1))
        res = _nk_m(r["op"], r["dtypes"], r["shapes"])
        if res is None:
            if GEMM_OP_RE.match(r["op"]) or "gemm" in r["op"].lower() or "scaled_mm" in r["op"]:
                unparsed[r["op"]] += cnt
            continue
        n, k, m, kind = res
        f = 2 * n * k * m * cnt
        if kind == "fp8":
            out["fp8_gemm_calls"] += cnt
            flops["fp8"] += f
            if (n, k) in q_shapes:
                out["fp8_q_shape_calls"] += cnt
        else:
            flops[kind] += f
            out[f"{kind}_gemm_calls"] += cnt
            if (n, k) in q_shapes and m >= min_tokens:
                out["nonfp8_q_shape_token_calls"] += cnt
                out[f"{kind}_q_shape_token_calls"] += cnt
    return {**{k: int(v) for k, v in out.items()}, "flops": {k: int(v) for k, v in flops.items()},
            "unparsed": dict(unparsed), "min_tokens": min_tokens,
            "q_shapes": sorted(list(s) for s in q_shapes)}


def find_fp8_layers(model, q_shapes=SD3_Q_SHAPES) -> list[dict]:
    """Modules that directly hold a 2-D float8 tensor whose shape, in either orientation, is a quantized layer's
    (N, K), plus a float32 tensor shaped like its 128x128 block scale (either orientation). Identification is by
    shape and dtype only: no module, class or attribute NAME is assumed. The orientation is NOT decided here --
    [1536, 6144] is a legal [N, K] and a legal [K, N] -- but at capture time, from the layer's own input width."""
    import torch
    found = []
    for mname, mod in model.named_modules():
        tens = [(n, t) for n, t in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False))
                if isinstance(t, torch.Tensor)]
        for wname, w in tens:
            if w.dtype != torch.float8_e4m3fn or w.dim() != 2:
                continue
            r, c = w.shape
            nks = sorted({nk for nk in ((r, c), (c, r)) if nk in q_shapes})
            if not nks:
                continue
            want = {(-(-a // BLOCK), -(-b // BLOCK)) for a, b in ((r, c), (c, r))}
            scales = [sn for sn, s in tens if s.dtype == torch.float32 and tuple(s.shape) in want]
            found.append({"module": mname, "weight": wname, "nk_candidates": nks, "scales": scales, "obj": mod})
    return found


def capture_layer_io(model, layers: list[dict], run, max_rows: int = 256) -> dict:
    """Run `run()` (one forward) with hooks on the given layers; keep max_rows evenly spaced rows of each layer's
    first input and first output, flattened to 2-D. Returns {module name: (x, y)}, the orientation fixed by what
    the layer actually consumed and produced. A layer that never fired is absent -- the caller must treat that as
    'not evaluated', never as a pass."""
    import torch
    got, hooks = {}, []

    def first_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)):
            for o in obj:
                if isinstance(o, torch.Tensor):
                    return o
        return None

    for L in layers:
        def hook(mod, args, out, name=L["module"], nks=tuple(L["nk_candidates"])):
            if name in got:
                return
            x, y = first_tensor(args), first_tensor(out)
            if x is None or y is None or (y.shape[-1], x.shape[-1]) not in nks:
                return
            k, n = x.shape[-1], y.shape[-1]
            x2 = x.detach().reshape(-1, k)
            y2 = y.detach().reshape(-1, n)
            idx = torch.linspace(0, x2.shape[0] - 1, steps=min(max_rows, x2.shape[0]), device=x2.device).round().long()
            got[name] = (x2[idx].clone(), y2[idx].clone())
        hooks.append(L["obj"].register_forward_hook(hook))
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    return got


def layer_beta(x, y, w_fp8, scale, block: int = BLOCK, input_dtype=None) -> dict:
    """beta of one layer, read on ONE candidate operand. x [M, K] the layer's captured input, y [M, N] its output,
    w_fp8/scale its OWN float8 weight and float32 block scale, stored in any orientation. Both references are built
    from those bytes in float32, on x cast to `input_dtype` first when given, else on x as captured: y16 = x @ deq(w)^T,
    y8 = deq(q(x)) @ deq(w)^T, q the declared per-token 1x128 E4M3 quantiser. Which operand the kernel quantised is
    layer_beta_on_quantised_operand's question. Columns are centred first, so a bias -- fused or not, present or not --
    cannot leak into the projection. Where the stored orientation is ambiguous (square blocks), the one whose W8A16
    reference is closest to y wins: a wrong one is off by ~100%, a right one by the few percent that separate W8A8
    from W8A16."""
    import torch
    import fp8_torch_ref as T
    n, k = y.shape[-1], x.shape[-1]
    xf = (x.to(input_dtype) if input_dtype is not None else x).float()
    yf = y.float()
    ws = [W for W in (w_fp8, w_fp8.t()) if tuple(W.shape) == (n, k)]
    ss = [S for S in (scale, scale.t()) if tuple(S.shape) == (-(-n // block), -(-k // block))]
    if not ws or not ss:
        raise ValueError(f"no orientation of weight {tuple(w_fp8.shape)} / scale {tuple(scale.shape)} fits [N={n}, K={k}]")
    q, sa = T.quant_act_groups(xf, block)
    xa = T.dequant_act_groups(q, sa, block)
    c = yf - yf.mean(0, keepdim=True)
    cands = []
    for wi, W in enumerate(ws):
        for si, S in enumerate(ss):
            wd = T.dequant_block(W.contiguous(), S.contiguous(), block)
            y16 = xf @ wd.t()
            y8 = xa @ wd.t()
            a = y16 - y16.mean(0, keepdim=True)
            b = y8 - y8.mean(0, keepdim=True)
            fit = float(torch.linalg.vector_norm(c - a) / torch.linalg.vector_norm(a))
            cands.append((fit, W is not w_fp8, S is not scale, a, b))
    fit, w_t, s_t, a, b = min(cands, key=lambda t: t[0])
    d = (b - a).flatten().double()
    e = (c - a).flatten().double()
    dd = float(d @ d)
    beta = float(e @ d) / dd if dd else float("nan")
    resid = float(torch.linalg.vector_norm(e - beta * d)) / math.sqrt(dd) if dd else float("nan")
    return {"beta": beta, "orthogonal_residual_over_d": resid, "rel_l2_vs_w8a16": fit,
            "d_rel": math.sqrt(dd) / float(torch.linalg.vector_norm(a.flatten().double())),
            "weight_transposed": w_t, "scale_transposed": s_t, "n": n, "k": k,
            "input_dtype": str(input_dtype if input_dtype is not None else x.dtype).replace("torch.", "")}


def layer_beta_on_quantised_operand(x, y, w_fp8, scale, operand_dtypes, block: int = BLOCK) -> dict:
    'beta read on the operand THE KERNEL QUANTISED.'


    reads, ops = [], []
    for dt in operand_dtypes:
        op = (x.to(dt) if dt is not None else x).float()
        if any(op.shape == o.shape and bool((op == o).all()) for o in ops):
            continue
        ops.append(op)
        reads.append(layer_beta(x, y, w_fp8, scale, block, input_dtype=dt))
    if not reads:
        raise ValueError("no candidate operand")

    def key(r):
        v = r["orthogonal_residual_over_d"]
        return v if isinstance(v, float) and math.isfinite(v) else math.inf

    best = min(reads, key=key)
    return dict(best, alternatives=[{k: r[k] for k in ("input_dtype", "beta", "orthogonal_residual_over_d")}
                                    for r in reads if r is not best])


def effective_weight_report(w_eff, codes_fp8, scale, block: int = BLOCK) -> dict:
    """Where does a kernel's EFFECTIVE weight depart from the asset's dequantised weight?

    w_eff [N, K]: what the kernel multiplies by, recovered with an identity input (y = I_K @ W_eff^T, bias zero) --
    exact for a W8A16 kernel, and for a W8A8 one too, because every row of I quantises losslessly (one 1.0 per row,
    all-zero groups). codes_fp8 / scale: the asset's float8 weight [N, K] and block scale [N/128, K/128].

    Returns the relative error, the per-block least-squares ratio map r[i, j] = <W_eff, W_deq> / <W_deq, W_deq> over
    each 128x128 block, how much error is left once each block is rescaled by its own r (small = a pure per-block
    SCALE defect; large = an element-level one, e.g. a repack), and per block whether the kernel used its OWN scale
    (|r - 1| <= 2^-8), the TRANSPOSED block's scale (square grids), or another one ('other'; how many of those are
    still explained by SOME block's scale to 2^-8 is reported too)."""
    import torch
    import fp8_torch_ref as T
    W = w_eff.float()
    s = scale.float()
    wd = T.dequant_block(codes_fp8.contiguous(), s.contiguous(), block)
    n, k = wd.shape
    nb, kb = s.shape
    blk_e = W.reshape(nb, block, kb, block)
    blk_d = wd.reshape(nb, block, kb, block)
    num = (blk_e * blk_d).sum(dim=(1, 3))
    den = (blk_d * blk_d).sum(dim=(1, 3)).clamp_min(1e-30)
    r = num / den
    resc = (blk_d * r[:, None, :, None]).reshape(n, k)
    rel = float(torch.linalg.vector_norm(W - wd) / torch.linalg.vector_norm(wd))
    rel_after = float(torch.linalg.vector_norm(W - resc) / torch.linalg.vector_norm(wd))
    implied = (r * s).flatten()
    flat = s.flatten()


    tol = 2.0 ** -8
    own = (r.flatten() - 1).abs() <= tol
    src = torch.argmin((implied[:, None] - flat[None, :]).abs(), dim=1)
    ii, jj = torch.meshgrid(torch.arange(nb), torch.arange(kb), indexing="ij")
    explained = (implied - flat[src]).abs() <= tol * flat[src].abs()
    if nb == kb:
        t_src = (jj.flatten() * kb + ii.flatten())
        trans = ~own & ((implied - flat[t_src]).abs() <= tol * flat[t_src].abs())
    else:
        trans = torch.zeros_like(own)
    exact = own | explained
    bf16_s = s.to(torch.bfloat16).float()
    wd_bf16s = T.dequant_block(codes_fp8.contiguous(), bf16_s.contiguous(), block)
    out = {"rel_err_vs_dequant": rel, "rel_err_after_per_block_rescale": rel_after,
           "block_ratio": {"min": float(r.min()), "max": float(r.max()), "median": float(r.median()),
                           "frac_off_by_more_than_1pct": float(((r - 1).abs() > 0.01).float().mean())},
           "implied_scale_source": {"own": int(own.sum()), "transposed": int(trans.sum()),
                                    "other": int((~own & ~trans).sum()), "explained_within_2^-8": int(exact.sum()),
                                    "blocks": int(own.numel())},
           "rel_err_vs_dequant_with_bf16_scales": float(torch.linalg.vector_norm(W - wd_bf16s) / torch.linalg.vector_norm(wd))}
    if nb == kb:
        wd_t = T.dequant_block(codes_fp8.contiguous(), s.t().contiguous(), block)
        out["rel_err_vs_dequant_with_transposed_scales"] = float(torch.linalg.vector_norm(W - wd_t) / torch.linalg.vector_norm(wd))
    return out


FP8_DTYPES = ("float8_e4m3fn", "float8_e5m2")


FI_PROVENANCE_OK = ("equal", "equal_plus_bias", "sum_of_rank_results")
FI_PROVENANCE_UNREAD = ("no_output", "same_storage", "unread_missing", "unread_positions_differ")


def flashinfer_boundary(rec: dict, ptr_to_asset: dict, pinned: dict, provenance_ok=FI_PROVENANCE_OK) -> dict:
    'flashinfer boundary.'
    calls = rec.get("flashinfer_calls") or []
    layers = rec.get("flashinfer_layers") or {}
    version = rec.get("flashinfer_version")
    trust = rec.get("flashinfer_trust") or {}
    entries = sorted({c.get("entry") for c in calls if c.get("entry")})
    allowed = set(pinned.get(version) or ()) if version else set()
    need_jit = set(trust.get("jit_modules") or ())
    loads = [x for x in rec.get("flashinfer_jit_loads") or [] if isinstance(x, dict)]
    fresh = {x.get("name") for x in loads if x.get("fresh_in_verifier_workspace")}
    jit_ok = need_jit <= fresh
    out = {"patched": bool(rec.get("flashinfer_patched")), "version": version, "calls": len(calls),
           "entries": entries, "pinned": bool(entries) and set(entries) <= allowed,
           "trusted_at_bind": bool(trust.get("verified")), "instrument_fault": trust.get("fault"),
           "trust_mismatch": list(trust.get("mismatch") or [])[:8],
           "entry_code_is_pinned": trust.get("entry_code_is_pinned"),
           "jit_modules_required": sorted(need_jit), "jit_modules_fresh": sorted(fresh & need_jit), "jit_ok": jit_ok,
           "layers_with_calls": len(layers), "sixteen_bit_weight_calls": 0, "weight_not_asset_calls": 0,
           "untrusted_calls": 0, "code_altered_calls": 0, "globals_altered_calls": 0,
           "runner_getter_altered_calls": 0, "not_executed_calls": 0,
           "provenance": {}, "provenance_bad_layers": [], "provenance_unread_layers": [], "evidenced_layers": 0}
    by_layer: dict = {}
    by_layer_assets: dict = {}
    for c in calls:
        w = (c.get("kwargs") or {}).get("weight")
        if w is None:
            w = next((a for a in c.get("args") or [] if a.get("i") == 1), None)
        ok_dtype = w is not None and w.get("dtype") in FP8_DTYPES
        ok_asset = ok_dtype and int(w.get("data_ptr", -1)) in ptr_to_asset
        if w is not None and not ok_dtype:
            out["sixteen_bit_weight_calls"] += 1
        elif not ok_asset:
            out["weight_not_asset_calls"] += 1
        t = c.get("trusted") or {}
        if not t.get("at_bind"):
            out["untrusted_calls"] += 1
        if t and not t.get("code_unaltered", False):
            out["code_altered_calls"] += 1
        if t and not t.get("globals_unaltered", False):
            out["globals_altered_calls"] += 1
        if t and not t.get("runner_getter_unaltered", False):
            out["runner_getter_altered_calls"] += 1
        if t and not t.get("trusted_frame", False):
            out["not_executed_calls"] += 1
        trusted = bool(t.get("at_bind") and t.get("code_unaltered") and t.get("globals_unaltered")
                       and t.get("runner_getter_unaltered") and t.get("trusted_frame") and jit_ok)
        L = c.get("layer")
        if L is not None:
            by_layer.setdefault(L, []).append(ok_asset and trusted)
            if ok_asset:
                by_layer_assets.setdefault(L, set()).update(_asset_list(ptr_to_asset[int(w["data_ptr"])]))
    out["evidenced_layer_names"] = []
    covered = set()
    for L, meta in sorted(layers.items()):
        prov = meta.get("provenance")
        out["provenance"][L] = prov
        if prov in FI_PROVENANCE_UNREAD or prov is None:
            out["provenance_unread_layers"].append(L)
        elif prov not in provenance_ok:
            out["provenance_bad_layers"].append(L)
        elif by_layer.get(L) and all(by_layer[L]):
            out["evidenced_layers"] += 1
            out["evidenced_layer_names"].append(L)
            covered |= by_layer_assets.get(L, set())
    out["evidenced_asset_modules"] = sorted(covered)
    fired = set(rec.get("layers_fired") or [])
    out["unrecorded_fired_layers"] = len(fired - set(layers)) if fired else None
    return out
