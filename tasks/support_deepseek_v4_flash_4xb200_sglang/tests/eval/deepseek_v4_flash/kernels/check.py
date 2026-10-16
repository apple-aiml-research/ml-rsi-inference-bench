#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The kernel interfaces of this task, checked and measured as the grader checks and measures them.

    python /app/eval/deepseek_v4_flash/kernels/check.py --tree /app --example bf16 --mode correctness --output /tmp/bf16.json
    python /app/eval/deepseek_v4_flash/kernels/check.py --tree /app --example packed --mode measure --output /tmp/p.json
    python /app/eval/deepseek_v4_flash/kernels/check.py --tree /app --mode measure --workload prefill --output /tmp/pf.json
    python /app/eval/deepseek_v4_flash/kernels/check.py --tree /app --profile-map /tmp/component-map.json

`--example` runs one of the fixed examples in `examples/` (`bf16`, `packed`, `c4-core`, `c4-indexer`, `c128`):
`correctness` compares the tree's outputs with the example's expected values under the published tolerances
(README "Numerical tolerances") and reports every discrepancy; `measure` times the example's invocations. `--workload`
times the production-size component workloads of README "Component efficiency" (`prefill`: seven families; `decode`:
six), on inputs generated here from fixed public seeds, with the grader's timing procedure. The output JSON records the
input digests, the tree's source and build identity, structural validation, every metric and the operation durations.
It reports what it measured: passing these examples, or any timing, does not establish acceptance on held-out cases,
and no pass threshold of the hidden efficiency comparison is applied or known here. `--profile-map` runs one invocation
of every workload family under the PyTorch profiler and writes the tree's component map (README "Execution
attribution"): per scoped role the kernels its public calls launch, by base name, generic copy/fill/elementwise kernels
left out -- the identity set `tests/served_recipe.py --component-map` looks for in a served trace.

The same functions -- `compare`, the MODEL1 and indexer layouts, `Measure`, the workload builders -- are what the
grader's private drivers call with their own inputs; nothing here selects or generates the private cases.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMITS = json.loads((HERE.parent / "limits.json").read_text())["operations"]

POLICIES = {
    "bf16_out": {"atol": 8e-4, "rtol": 3.01 / 128, "cosine": 7e-6},
    "bf16_stat": {"atol": 1e-6, "rtol": 2.01 / 65536},
    "packed_out": {"atol": 1e-3, "rtol": 2.01 / 128, "cosine": 5e-6},
    "packed_lse": {"atol": 1e-6, "rtol": 8.01 / 65536},
    "values": {"atol": 5e-3, "rtol": 5e-3, "combined": True},
}
REL_EPS = 1e-6
COSINE_ZERO = 1e-12

D = 512
H = 64
D_NOPE, D_ROPE, GROUP, N_GROUPS = 448, 64, 64, 7
DATA_ROW, SCALE_ROW = 576, 8
INDEXER_D = 128
MAIN_PAGE, C4_PAGE, C128_PAGE = 256, 64, 2
SWA_PAGE = 256
RING = {4: 16, 128: 256}




@dataclass
class Arm:
    tree: Path
    flash_mla: object
    dsv4: object
    identity: dict = field(default_factory=dict)


def load_arm(tree: Path) -> Arm:
    """Import `flash_mla` and `sglang.jit_kernel.deepseek_v4` from `tree` (its vendored FlashMLA and its engine), ahead of
    anything installed, and record where they came from."""
    tree = Path(tree).resolve()
    for p in (tree / "python", tree / "third_party" / "flash_mla"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import flash_mla  # noqa: PLC0415
    import flash_mla.cuda  # noqa: PLC0415, F401
    from sglang.jit_kernel import deepseek_v4 as dsv4  # noqa: PLC0415
    ident = {"tree": str(tree), "flash_mla": os.path.realpath(flash_mla.__file__),
             "flash_mla_cuda": os.path.realpath(sys.modules["flash_mla.cuda"].__file__),
             "deepseek_v4": os.path.realpath(dsv4.__file__)}
    with contextlib.suppress(Exception):
        ident["head"] = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       timeout=60).stdout.strip()
    with contextlib.suppress(OSError):
        ident["flash_mla_cuda_sha256"] = _sha256_file(Path(ident["flash_mla_cuda"]))
    return Arm(tree, flash_mla, dsv4, ident)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"none")
            continue
        c = t.detach().contiguous().cpu()
        h.update(str((tuple(c.shape), str(c.dtype))).encode())
        h.update(c.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()




def compare(answer, reference, policy: str, expect_nonfinite=None) -> dict:
    """Every element of `answer` against `reference` under `policy`; the record, `ok` the verdict of this comparison.

    `expect_nonfinite` names the specified sentinels of a statistic: a boolean mask of elements whose reference is
    +/-inf; those must match exactly (same sign of infinity) and are excluded from the finite comparison. Any other
    non-finite value on either side fails. A zero reference passes through the elementwise rule only; the cosine uses
    float64 and the pinned near-zero convention."""
    import torch  # noqa: PLC0415

    pol = POLICIES[policy]
    a = answer.detach().to(torch.float64).cpu()
    r = reference.detach().to(torch.float64).cpu()
    rec: dict = {"policy": policy, "shape": list(a.shape), "ok": True, "problems": []}
    if tuple(a.shape) != tuple(r.shape):
        rec.update(ok=False, problems=[f"shape {tuple(a.shape)} != {tuple(r.shape)}"])
        return rec
    if expect_nonfinite is not None:
        mask = expect_nonfinite.cpu().bool()
        same = (a[mask] == r[mask]) | (torch.isnan(a[mask]) & torch.isnan(r[mask]))
        if not bool(same.all()):
            rec["ok"] = False
            rec["problems"].append(f"{int((~same).sum())} sentinel element(s) differ from the specified +/-inf")
        a, r = a[~mask], r[~mask]
    bad_a, bad_r = ~torch.isfinite(a), ~torch.isfinite(r)
    if bool(bad_r.any()):
        rec["ok"] = False
        rec["problems"].append(f"{int(bad_r.sum())} non-finite reference element(s) outside the specified sentinels")
    if bool(bad_a.any()):
        rec["ok"] = False
        rec["problems"].append(f"{int(bad_a.sum())} unexpected non-finite element(s)")
    fin = ~(bad_a | bad_r)
    a_f, r_f = a[fin], r[fin]
    if a_f.numel():
        abs_err = (a_f - r_f).abs()
        rel_err = abs_err / (r_f.abs() + REL_EPS)
        if pol.get("combined"):
            ok_el = abs_err <= pol["atol"] + pol["rtol"] * r_f.abs()
        else:
            ok_el = (abs_err <= pol["atol"]) | (rel_err <= pol["rtol"])
        worst = int(torch.argmax(abs_err))
        rec.update({"n": int(a_f.numel()), "max_abs": float(abs_err.max()), "max_rel": float(rel_err.max()),
                    "failing_elements": int((~ok_el).sum()), "worst_flat_index": worst,
                    "worst": {"answer": float(a_f[worst]), "reference": float(r_f[worst])}})
        if not bool(ok_el.all()):
            rec["ok"] = False
            rec["problems"].append(f"{int((~ok_el).sum())} of {a_f.numel()} elements outside the {policy} bounds "
                                   f"(max abs {float(abs_err.max()):.3g}, max rel {float(rel_err.max()):.3g})")
    if "cosine" in pol:
        cos = symmetric_cosine(answer.detach().to(torch.float64).cpu(), reference.detach().to(torch.float64).cpu())
        rec["cosine"] = cos
        if not (abs(cos) <= pol["cosine"]):
            rec["ok"] = False
            rec["problems"].append(f"symmetric cosine difference {cos:.3g} over the bound {pol['cosine']:g}")
    return rec


def symmetric_cosine(a, r) -> float:
    """1 - 2 sum(a r) / (sum(a^2) + sum(r^2)) in float64; 0 when sum(r^2) < 1e-12 (the pinned near-zero convention)."""
    import torch  # noqa: PLC0415

    a = a.to(torch.float64).reshape(-1)
    r = r.to(torch.float64).reshape(-1)
    fin = torch.isfinite(a) & torch.isfinite(r)
    a, r = a[fin], r[fin]
    rr = float((r * r).sum())
    if rr < COSINE_ZERO:
        return 0.0
    return float(1.0 - 2.0 * float((a * r).sum()) / (float((a * a).sum()) + rr))




def model1_page_bytes(page_size: int) -> int:
    """The MODEL1 page stride: all data rows (576 B each), then all scale rows (8 B each), padded to 576 B."""
    return math.ceil(584 * page_size / DATA_ROW) * DATA_ROW


def indexer_page_bytes(page_size: int) -> int:
    return (INDEXER_D + 4) * page_size


def new_model1_cache(num_pages: int, page_size: int, device="cuda"):
    """An empty MODEL1 pool as the engine views it: [num_pages, page_size, 1, 584] bytes over the padded page stride."""
    import torch  # noqa: PLC0415

    stride = model1_page_bytes(page_size)
    raw = torch.zeros(num_pages, stride, dtype=torch.uint8, device=device)
    return raw, raw[:, : page_size * 584].view(num_pages, page_size, 1, 584)


def pack_model1(values, raw, page_size: int, token_index) -> None:
    """Write float values [n, 512] into MODEL1 entries at physical token indices of the byte pool `raw`, with the
    FlashMLA test-data convention (per group: scale = 2 ** ceil(log2(max(max|x| / 448, 1e-4))), E4M3 values, rotary
    coordinates as BF16). This builds legal attention inputs; it is not the engine's cache-store rounding rule."""
    import torch  # noqa: PLC0415

    v = values.to(torch.float32)
    n = v.shape[0]
    g = v[:, :D_NOPE].reshape(n, N_GROUPS, GROUP)
    s = torch.pow(2.0, torch.ceil(torch.log2(torch.clamp_min(g.abs().amax(-1) / 448.0, 1e-4))))
    q = (g / s.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, D_NOPE)
    rope = v[:, D_NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(n, 2 * D_ROPE)
    sbytes = (torch.log2(s).round().to(torch.int64) + 127).to(torch.uint8)
    idx = token_index.reshape(-1).to(torch.int64).to(raw.device)
    page, off = idx // page_size, idx % page_size
    rows = torch.cat([q, rope], dim=-1).to(raw.device)
    cols = torch.arange(DATA_ROW, device=raw.device)
    raw[page.unsqueeze(1), (off * DATA_ROW).unsqueeze(1) + cols.unsqueeze(0)] = rows
    scol = torch.arange(N_GROUPS, device=raw.device)
    raw[page.unsqueeze(1), (DATA_ROW * page_size + off * SCALE_ROW).unsqueeze(1) + scol.unsqueeze(0)] = sbytes.to(raw.device)


def decode_model1(raw, page_size: int, token_index):
    """The float32 values of MODEL1 entries at physical token indices, decoded from the bytes (all 512 coordinates)."""
    import torch  # noqa: PLC0415

    idx = token_index.reshape(-1).to(torch.int64).to(raw.device)
    page, off = idx // page_size, idx % page_size
    cols = torch.arange(DATA_ROW, device=raw.device)
    data = raw[page.unsqueeze(1), (off * DATA_ROW).unsqueeze(1) + cols.unsqueeze(0)]
    scol = torch.arange(N_GROUPS, device=raw.device)
    sb = raw[page.unsqueeze(1), (DATA_ROW * page_size + off * SCALE_ROW).unsqueeze(1) + scol.unsqueeze(0)]
    scale = torch.pow(2.0, sb.to(torch.float32) - 127.0)
    nope = data[:, :D_NOPE].contiguous().view(torch.float8_e4m3fn).to(torch.float32).view(-1, N_GROUPS, GROUP)
    rope = data[:, D_NOPE:].contiguous().view(torch.bfloat16).to(torch.float32)
    out = torch.cat([(nope * scale.unsqueeze(-1)).reshape(-1, D_NOPE), rope], dim=-1)
    return out.view(*token_index.shape, D)


def decode_indexer(raw, page_size: int, token_index):
    """The float32 values of indexer entries: 128 E4M3 values per token, then a 4-byte FP32 scale per token."""
    import torch  # noqa: PLC0415

    idx = token_index.reshape(-1).to(torch.int64).to(raw.device)
    page, off = idx // page_size, idx % page_size
    cols = torch.arange(INDEXER_D, device=raw.device)
    data = raw[page.unsqueeze(1), (off * INDEXER_D).unsqueeze(1) + cols.unsqueeze(0)]
    scol = torch.arange(4, device=raw.device)
    sb = raw[page.unsqueeze(1), (INDEXER_D * page_size + off * 4).unsqueeze(1) + scol.unsqueeze(0)]
    scale = sb.contiguous().view(torch.float32).view(-1, 1)
    return (data.contiguous().view(torch.float8_e4m3fn).to(torch.float32) * scale).view(*token_index.shape, INDEXER_D)


def store_rule(x, store_type: str) -> dict:
    """README `fused_store_cache`: the bytes the store rule gives rows `x` (MODEL1: FP32 [n, 512] -> 576 data bytes and
    7 E8M0 bytes per row; indexer: BF16 [n, 128] -> 128 E4M3 bytes and the FP32 scale per row)."""
    import torch  # noqa: PLC0415

    v = x.detach().to(torch.float32).cpu()
    n = v.shape[0]
    if store_type == "indexer":
        scale = torch.clamp_min(v.abs().amax(-1, keepdim=True), 1e-4) / 448.0
        return {"data": (v / scale).to(torch.float8_e4m3fn).view(torch.uint8), "scale": scale.view(-1)}
    g = v[:, :D_NOPE].reshape(n, N_GROUPS, GROUP)
    raw = torch.clamp_min(g.abs().amax(-1), 1e-4) / 448.0
    e = torch.ceil(torch.log2(raw.to(torch.float64)))
    e = torch.where(torch.pow(2.0, e - 1) >= raw.to(torch.float64), e - 1, e)
    e = torch.where(torch.pow(2.0, e) < raw.to(torch.float64), e + 1, e)
    scale = torch.pow(2.0, e).to(torch.float32)
    q = (g / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, D_NOPE)
    rope = v[:, D_NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(n, 2 * D_ROPE)
    return {"data": torch.cat([q, rope], dim=-1), "scale": (e + 127).to(torch.uint8)}


#: README `fused_store_cache`: an E4M3 value whose unrounded quotient lies this close (relative) to a rounding tie may
#: take either neighbour -- the FP32 division lowering (x / scale or x * (1 / scale)) and the one-ULP scale decide it
TIE_ZONE = 2.0 ** -21


def indexer_payload_mismatches(stored, x, scale) -> tuple[int, int]:
    """(bytes that are neither the rounding of x / scale nor, inside a tie zone, its other neighbour; values inside a tie
    zone) for indexer rows `x` [n, 128] stored with FP32 `scale` [n]."""
    import torch  # noqa: PLC0415

    q = x.detach().to(torch.float64).cpu() / scale.detach().to(torch.float64).cpu().view(-1, 1)
    def e4m3(v):
        return v.to(torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    lo, hi, mid = e4m3(q * (1.0 - TIE_ZONE)), e4m3(q * (1.0 + TIE_ZONE)), e4m3(q)
    s = stored.detach().cpu().to(torch.uint8)
    tie = lo != hi
    ok = torch.where(tie, (s == lo) | (s == hi), s == mid)
    return int((~ok).sum()), int(tie.sum())


def stored_rows(raw, page_size: int, token_index, store_type: str) -> dict:
    """The data and scale bytes a pool holds at physical token indices (the MODEL1 scale row's padding byte left out)."""
    import torch  # noqa: PLC0415

    r = raw.detach().cpu()
    idx = token_index.reshape(-1).to(torch.int64).cpu()
    page, off = idx // page_size, idx % page_size
    width, srow, scount = (INDEXER_D, 4, 4) if store_type == "indexer" else (DATA_ROW, SCALE_ROW, N_GROUPS)
    cols = torch.arange(width)
    data = r[page.unsqueeze(1), (off * width).unsqueeze(1) + cols.unsqueeze(0)]
    scol = torch.arange(scount)
    sb = r[page.unsqueeze(1), (width * page_size + off * srow).unsqueeze(1) + scol.unsqueeze(0)]
    return {"data": data, "scale": sb.contiguous().view(torch.float32).view(-1) if store_type == "indexer" else sb}




@dataclass
class Request:
    """One logical request's SWA pages (physical, in order) for the compression state's address arithmetic."""
    pages: list[int]
    length: int = 0


def state_loc(req: Request, pos: int, ratio: int) -> int:
    """The state window of `pos`: (page(pos) * ring + pos % ring) // ratio, negative positions clamped to 0."""
    pos = max(pos, 0)
    ring = RING[ratio]
    return (req.pages[pos // SWA_PAGE] * ring + pos % ring) // ratio


def new_state(num_swa_pages: int, ratio: int, head_dim: int, device="cuda"):
    """A cleared compression state pool, [windows, ratio, (4 or 2) * head_dim] float32 (kv half 0, score half -inf)."""
    import torch  # noqa: PLC0415

    windows = num_swa_pages * RING[ratio] // ratio + 1
    last = (4 if ratio == 4 else 2) * head_dim
    buf = torch.zeros(windows, ratio, last, dtype=torch.float32, device=device)
    buf[..., last // 2:] = float("-inf")
    return buf


def prefill_metadata(reqs: list[Request], extend_lens: list[int], ratio: int, device="cuda"):
    """indices and extra for a prefill/extend call (README "Compression state"): per request the write window of the
    last position, and for C4 [load-overlap, load, write-overlap, write position], for C128 the load window."""
    import torch  # noqa: PLC0415

    write, extra = [], []
    for req, e in zip(reqs, extend_lens):
        seq_len = req.length + e
        prefix = req.length
        write_pos = ((seq_len - 1) // ratio) * ratio
        load_pos = ((prefix - 1) // ratio) * ratio
        v0, v1 = state_loc(req, load_pos, ratio), state_loc(req, write_pos, ratio)
        write.append(v1)
        if ratio == 4:
            v2, v3 = state_loc(req, load_pos - ratio, ratio), state_loc(req, write_pos - ratio, ratio)
            extra.append([v2, v0, v3, write_pos])
        else:
            extra.append(v0)
    w = torch.tensor(write, dtype=torch.int32, device=device)
    x = torch.tensor(extra, dtype=torch.int32, device=device)
    return w, (x.view(-1, 4).contiguous() if ratio == 4 else x.contiguous())


def decode_metadata(reqs: list[Request], ratio: int, device="cuda"):
    """indices and extra for a one-token decode step of each request, whose length already counts that token."""
    import torch  # noqa: PLC0415

    write, extra = [], []
    for req in reqs:
        wp = ((req.length - 1) // ratio) * ratio
        write.append(state_loc(req, wp, ratio))
        extra.append(state_loc(req, wp - ratio, ratio))
    w = torch.tensor(write, dtype=torch.int32, device=device)
    return w, (torch.tensor(extra, dtype=torch.int32, device=device).view(-1, 1) if ratio == 4 else None)


def rope_table(max_pos: int, base: float = 10000.0, device="cuda"):
    """A fixed rotary table [max_pos, 32] complex64 (the operation takes the table as an input)."""
    import torch  # noqa: PLC0415

    inv = 1.0 / (base ** (torch.arange(0, D_ROPE, 2, dtype=torch.float64) / D_ROPE))
    ang = torch.outer(torch.arange(max_pos, dtype=torch.float64), inv)
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64).to(device)


def emitting_rows_prefill(plan) -> list[int]:
    """The ragged row ids the prefill plan marks as compression events (the valid output rows)."""
    import torch  # noqa: PLC0415

    raw = plan.compress_plan.detach().cpu().contiguous().view(torch.int32).view(-1, 4)
    return [int(r) for r in raw[:, 0].tolist() if int(r) != -1 and int(r) != 0xFFFFFFFF]




def compress_step(arm: Arm, *, state, kv_score_input, ape, indices, plan, extra, head_dim: int, ratio: int,
                  norm_weight, eps: float, freqs, cache_raw=None, cache_page: int = 0, store_type: str = "flashmla",
                  store_loc=None) -> object:
    """compress_forward, compress_fused_norm_rope_inplace and fused_store_cache in the engine's order (the
    component's complete work); the compressed rows (float32, valid only at the plan's events)."""
    out = arm.dsv4.compress_forward(state, kv_score_input, ape, indices, plan, extra, head_dim=head_dim,
                                     compress_ratio=ratio)
    arm.dsv4.compress_fused_norm_rope_inplace(out, norm_weight, eps, freqs, plan)
    if cache_raw is not None:
        view = cache_raw if store_type == "indexer" else cache_raw
        src = out if store_type == "flashmla" else out.to(__import__("torch").bfloat16)
        arm.dsv4.fused_store_cache(src, view, store_loc, page_size=cache_page, type=store_type)
    return out




@dataclass
class Measure:
    """The timing procedure of one family: `warmup` untimed complete invocations, then `draws` draws of `reps`
    invocations, each preceded by `reset()` outside the timed interval (identical synchronization in every arm); a
    draw's latency is the mean over its invocations of the synchronized wall time and, separately, of the CUDA-event
    time; the family's value is the median over draws. `graph=True` captures one invocation after warm-up in a CUDA
    graph and times its replays (a capture failure is an operation failure)."""
    warmup: int = 10
    draws: int = 3
    reps: int = 100
    graph: bool = False

    def run(self, invoke, reset) -> dict:
        import torch  # noqa: PLC0415

        t_all = time.monotonic()
        for _ in range(self.warmup):
            reset()
            invoke()
        torch.cuda.synchronize()
        call = invoke
        if self.graph:
            reset()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                invoke()
            torch.cuda.current_stream().wait_stream(s)
            reset()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                invoke()
            torch.cuda.synchronize()
            call = g.replay
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        draws = []
        for _ in range(self.draws):
            walls, gpus = [], []
            for _ in range(self.reps):
                reset()
                torch.cuda.synchronize()
                e0.record()
                t0 = time.perf_counter()
                call()
                e1.record()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                walls.append(t1 - t0)
                gpus.append(e0.elapsed_time(e1) / 1e3)
            draws.append({"wall_s": statistics.fmean(walls), "gpu_s": statistics.fmean(gpus), "n": len(walls)})
        return {"draws": draws, "wall_s": statistics.median(d["wall_s"] for d in draws),
                "gpu_s": statistics.median(d["gpu_s"] for d in draws), "graph": self.graph,
                "warmup": self.warmup, "reps": self.reps, "operation_s": round(time.monotonic() - t_all, 3)}


def _gen(seed: int):
    import torch  # noqa: PLC0415

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _normal(shape, g, clamp: float = 3.0):
    import torch  # noqa: PLC0415

    return torch.randn(*shape, generator=g, dtype=torch.float32).clamp_(-clamp, clamp)


def _attention_bf16(shape, g):
    """Attention values at FlashMLA's own test scale (its tests/lib.py, the scale its tolerances were set on): BF16
    randn / 10 + (u - 0.5) / 10, u one uniform per tensor, clamped to [-10, 10]."""
    import torch  # noqa: PLC0415

    t = torch.randn(*shape, generator=g, dtype=torch.bfloat16) / 10 + (float(torch.rand((), generator=g)) - 0.5) / 10
    return t.clamp_(-10, 10)


def bf16_inputs(q_len: int, kv_len: int, topk: int, seed: int, device="cuda") -> dict:
    """A sparse BF16 prefill input: q [Q, 64, 512], kv [KV, 1, 512] at FlashMLA's test scale, distinct valid indices
    [Q, 1, K] in random order, a finite non-constant sink per head, full valid lengths."""
    import torch  # noqa: PLC0415

    g = _gen(seed)
    q = _attention_bf16((q_len, H, D), g)
    kv = _attention_bf16((kv_len, 1, D), g)
    idx = torch.stack([torch.randperm(kv_len, generator=g)[:topk] for _ in range(q_len)]).view(q_len, 1, topk)
    sink = torch.linspace(-2.0, 2.0, H, dtype=torch.float32)
    return {"q": q.to(device), "kv": kv.to(device), "indices": idx.to(torch.int32).to(device),
            "attn_sink": sink.to(device), "topk_length": None, "sm_scale": D ** -0.5}


def packed_inputs(batch: int, sq: int, extra_kind: str | None, main_cap: int, extra_cap: int, seed: int,
                  slots: int = 8192, device="cuda") -> dict:
    """A sparse packed decode input at the production page geometry: main pool page 256; C4 extra page 64; C128 extra
    page 2; `slots` logical entries per present pool, randomly placed; full valid lengths; a finite sink. Values at
    FlashMLA's decode-test scale (q randn clamped to [-1, 1]; keys randn / 10 clamped to [-1, 1])."""
    import torch  # noqa: PLC0415

    g = _gen(seed)
    q = _normal((batch, sq, H, D), g, clamp=1.0).to(torch.bfloat16)
    pools = {}
    for name, page, cap in (("main", MAIN_PAGE, main_cap),
                            *(((extra_kind, C4_PAGE if extra_kind == "c4" else C128_PAGE, extra_cap),) if extra_kind else ())):
        pages = math.ceil(slots / page) + 1
        raw, view = new_model1_cache(pages, page, device)
        phys = torch.randperm((pages - 1) * page, generator=g)[:slots] + page
        pack_model1((torch.randn(slots, D, generator=g) / 10).clamp_(-1, 1), raw, page, phys)
        sel = torch.stack([phys[torch.randperm(slots, generator=g)[:cap]] for _ in range(batch * sq)])
        pools[name] = {"raw": raw, "view": view, "indices": sel.view(batch, sq, cap).to(torch.int32).to(device),
                       "lengths": torch.full((batch,), cap, dtype=torch.int32, device=device), "page": page}
    return {"q": q.to(device), "pools": pools, "attn_sink": torch.linspace(-1.0, 1.0, H, dtype=torch.float32, device=device),
            "sm_scale": D ** -0.5}


def call_packed(arm: Arm, inp: dict, sched=None):
    """One sparse `flash_mla_with_kvcache` call on a packed input; (out, lse, sched)."""
    main = inp["pools"]["main"]
    extra = next((p for k, p in inp["pools"].items() if k != "main"), None)
    sched = sched or arm.flash_mla.get_mla_metadata()[0]
    out, lse = arm.flash_mla.flash_mla_with_kvcache(
        inp["q"], main["view"], None, None, D, sched, None, inp["sm_scale"], False, True,
        indices=main["indices"], attn_sink=inp["attn_sink"],
        extra_k_cache=extra["view"] if extra else None,
        extra_indices_in_kvcache=extra["indices"] if extra else None,
        topk_length=main["lengths"], extra_topk_length=extra["lengths"] if extra else None)
    return out, lse, sched


def call_bf16(arm: Arm, inp: dict):
    return arm.flash_mla.flash_mla_sparse_fwd(inp["q"], inp["kv"], inp["indices"], inp["sm_scale"], D,
                                              inp["attn_sink"], inp["topk_length"])


def compression_family(arm: Arm, ratio: int, head_dim: int, store_type: str, phase: str, seed: int,
                       device="cuda") -> tuple:
    """(invoke, reset, meta) of one compression family at the published sizes: prefill = one request of 4,096 projected
    tokens from an empty state; decode = 16 requests each from a 4,096-token accepted prefix, one complete cycle of 128
    advancing one-token steps (plans prepared before capture). Every invocation starts from its own restored state."""
    import torch  # noqa: PLC0415

    g = _gen(seed)
    coff = 2 if ratio == 4 else 1
    width = 2 * coff * head_dim
    ape = _normal((ratio * coff, head_dim), g).to(device)
    weight = (1.0 + 0.01 * (torch.arange(head_dim) % 17)).to(torch.float32).to(device)
    eps = 1e-6
    freqs = rope_table(4096 + 256, device=device)
    cache_page = C4_PAGE if ratio == 4 else C128_PAGE
    if phase == "prefill":
        reqs = [Request(pages=list(range(1, 1 + 4096 // SWA_PAGE + 1)))]
        n = 4096
        inp = _normal((n, width), g).to(device)
        state0 = new_state(1 + len(reqs[0].pages) + 1, ratio, head_dim, device)
        state = state0.clone()
        indices, extra = prefill_metadata(reqs, [n], ratio, device)
        plan = arm.dsv4.compress_plan(ratio, n, torch.tensor([n]), torch.tensor([n]), device)
        n_out = n // ratio
        pool_pages = n_out // cache_page + 2
        raw = torch.zeros(pool_pages, (model1_page_bytes(cache_page) if store_type == "flashmla"
                                       else indexer_page_bytes(cache_page)), dtype=torch.uint8, device=device)
        rows = torch.arange(n, device=device)
        loc = torch.where((rows + 1) % ratio == 0, (rows + 1) // ratio - 1 + cache_page, torch.zeros_like(rows)).to(torch.int32)

        def reset():
            state.copy_(state0)

        def invoke():
            compress_step(arm, state=state, kv_score_input=inp, ape=ape, indices=indices, plan=plan, extra=extra,
                          head_dim=head_dim, ratio=ratio, norm_weight=weight, eps=eps, freqs=freqs, cache_raw=raw,
                          cache_page=cache_page, store_type=store_type, store_loc=loc)
        return invoke, reset, {"ratio": ratio, "head_dim": head_dim, "store": store_type, "tokens": n}
    b, prefix, steps = 16, 4096, 128
    reqs = [Request(pages=[1 + r * 18 + k for k in range(18)], length=prefix) for r in range(b)]
    state0 = new_state(1 + b * 18 + 1, ratio, head_dim, device)
    pre = _normal((b * prefix, width), g).to(device)
    pre_reqs = [Request(pages=r.pages, length=0) for r in reqs]
    idx0, ex0 = prefill_metadata(pre_reqs, [prefix] * b, ratio, device)
    plan0 = arm.dsv4.compress_plan(ratio, b * prefix, torch.tensor([prefix] * b), torch.tensor([prefix] * b), device)
    arm.dsv4.compress_forward(state0, pre, ape, idx0, plan0, ex0, head_dim=head_dim, compress_ratio=ratio)
    torch.cuda.synchronize()
    state = state0.clone()
    step_inputs = [_normal((b, width), g).to(device) for _ in range(steps)]
    plans, metas, locs = [], [], []
    pool_pages = b * (steps // ratio + 2) // cache_page + 4
    raw = torch.zeros(pool_pages, (model1_page_bytes(cache_page) if store_type == "flashmla"
                                   else indexer_page_bytes(cache_page)), dtype=torch.uint8, device=device)
    for s in range(steps):
        for r in reqs:
            r.length = prefix + s + 1
        seq = torch.tensor([r.length for r in reqs], dtype=torch.int32, device=device)
        plans.append(arm.dsv4.compress_plan(ratio, b, seq, None, device))
        metas.append(decode_metadata(reqs, ratio, device))
        emit = [(r.length % ratio == 0) for r in reqs]
        locs.append(torch.tensor([(cache_page + i * (steps // ratio + 1) + (r.length - prefix) // ratio) if e else 0
                                  for i, (r, e) in enumerate(zip(reqs, emit))], dtype=torch.int32, device=device))

    def reset():
        state.copy_(state0)

    def invoke():
        for s in range(steps):
            compress_step(arm, state=state, kv_score_input=step_inputs[s], ape=ape, indices=metas[s][0], plan=plans[s],
                          extra=metas[s][1], head_dim=head_dim, ratio=ratio, norm_weight=weight, eps=eps, freqs=freqs,
                          cache_raw=raw, cache_page=cache_page, store_type=store_type, store_loc=locs[s])
    return invoke, reset, {"ratio": ratio, "head_dim": head_dim, "store": store_type, "requests": b, "steps": steps}


def workload_families(phase: str) -> list[dict]:
    """README "Component efficiency": the families of one phase (seven prefill, six decode)."""
    fams = []
    if phase == "prefill":
        fams.append({"family": "bf16", "kind": "bf16"})
    for sub, extra in (("main", None), ("main+c4", "c4"), ("main+c128", "c128")):
        fams.append({"family": f"packed-{sub}", "kind": "packed", "extra": extra,
                     "batch": 4096 if phase == "prefill" else 64})
    for name, ratio, hd, store in (("c4-core", 4, 512, "flashmla"), ("c4-indexer", 4, 128, "indexer"),
                                   ("c128", 128, 512, "flashmla")):
        fams.append({"family": name, "kind": "compress", "ratio": ratio, "head_dim": hd, "store": store})
    return fams


def measure_family(arm: Arm, fam: dict, phase: str, seed: int) -> dict:
    """One family measured as README "Component efficiency" specifies; the record."""
    import torch  # noqa: PLC0415

    t0 = time.monotonic()
    limit = int(LIMITS["invocation"]["limit_s"])
    if fam["kind"] == "bf16":
        inp = bf16_inputs(4096, 8192, 512, seed)
        m = Measure(reps=100, graph=False).run(lambda: call_bf16(arm, inp), lambda: None)
        meta = {"q": 4096, "kv": 8192, "topk": 512, "digest": tensor_digest(inp["q"], inp["kv"], inp["indices"])}
    elif fam["kind"] == "packed":
        inp = packed_inputs(fam["batch"], 1, fam["extra"], 128, 512, seed)
        sched = arm.flash_mla.get_mla_metadata()[0]
        call_packed(arm, inp, sched)
        m = Measure(reps=100, graph=(phase == "decode")).run(lambda: call_packed(arm, inp, sched), lambda: None)
        meta = {"batch": fam["batch"], "sq": 1, "extra": fam["extra"], "main_cap": 128, "extra_cap": 512,
                "digest": tensor_digest(inp["q"], *[p["raw"] for p in inp["pools"].values()])}
    else:
        invoke, reset, meta = compression_family(arm, fam["ratio"], fam["head_dim"], fam["store"], phase, seed)
        m = Measure(reps=10, graph=(phase == "decode")).run(invoke, reset)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    rec = {"family": fam["family"], "phase": phase, **m, "meta": meta, "elapsed_s": round(elapsed, 3), "limit_s": limit,
           "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    if elapsed > limit:
        rec["failure"] = f"the family measurement took {elapsed:.1f} s, over its {limit} s invocation limit"
    torch.cuda.reset_peak_memory_stats()
    return rec



GENERIC_NAMESPACES = ("at::native::", "at::cuda::", "c10::", "cub::", "thrust::")


def generic_kernel(base: str) -> bool:
    return base.startswith(GENERIC_NAMESPACES) or base.lower().startswith(("memcpy", "memset"))
ROLE_OF = {"bf16": "bf16_attention", "packed-main": "packed_attention", "packed-main+c4": "packed_attention",
           "packed-main+c128": "packed_attention", "c4-core": "compression_c4", "c4-indexer": "compression_c4_indexer",
           "c128": "compression_c128"}


def base_name(kernel: str) -> str:
    """A kernel's name without its template arguments, parameter list or leading `void`: every qualifier up to the first
    `<` or `(` that opens an argument list, `(anonymous namespace)` kept. Specializations of one kernel function (another
    tile, another graph shape) share it."""
    n = kernel.strip()
    if n.startswith("void "):
        n = n[5:]
    out, i = [], 0
    while i < len(n):
        if n.startswith("(anonymous namespace)", i):
            out.append("(anonymous namespace)")
            i += len("(anonymous namespace)")
            continue
        if n[i] in "<(":
            break
        out.append(n[i])
        i += 1
    return "".join(out).strip()


def profile_family(arm: Arm, fam: dict, phase: str, seed: int) -> dict:
    """One untimed invocation of a workload family under the PyTorch profiler (CUDA activity); the kernels it launched."""
    import torch  # noqa: PLC0415

    if fam["kind"] == "bf16":
        inp = bf16_inputs(4096, 8192, 512, seed)
        call = lambda: call_bf16(arm, inp)  # noqa: E731
        reset = lambda: None  # noqa: E731
    elif fam["kind"] == "packed":
        inp = packed_inputs(fam["batch"], 1, fam["extra"], 128, 512, seed)
        sched = arm.flash_mla.get_mla_metadata()[0]
        call_packed(arm, inp, sched)
        call = lambda: call_packed(arm, inp, sched)  # noqa: E731
        reset = lambda: None  # noqa: E731
    else:
        call, reset, _ = compression_family(arm, fam["ratio"], fam["head_dim"], fam["store"], phase, seed)
    reset()
    call()
    torch.cuda.synchronize()
    reset()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    names: dict[str, int] = {}
    for e in prof.events():
        if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA:
            names[e.name] = names.get(e.name, 0) + 1
    return {"family": fam["family"], "phase": phase, "role": ROLE_OF[fam["family"]], "kernels": names}


def component_map(profiles: list[dict]) -> dict:
    """Per role the base names of the non-generic kernels its families launched, with the raw names behind them."""
    roles: dict[str, dict] = {}
    for p in profiles:
        r = roles.setdefault(p["role"], {"base_names": set(), "kernels": {}, "families": set()})
        r["families"].add(f"{p['phase']}/{p['family']}")
        for name, count in p["kernels"].items():
            base = base_name(name)
            if not base or generic_kernel(base):
                continue
            r["base_names"].add(base)
            r["kernels"][name] = r["kernels"].get(name, 0) + count
    return {role: {"base_names": sorted(v["base_names"]), "families": sorted(v["families"]), "kernels": v["kernels"]}
            for role, v in sorted(roles.items())}




def _load_example(name: str) -> tuple[dict, dict]:
    from safetensors.torch import load_file  # noqa: PLC0415

    meta = json.loads((HERE / "examples" / f"{name}.json").read_text())
    tensors = load_file(str(HERE / "examples" / f"{name}.safetensors"))
    got = hashlib.sha256((HERE / "examples" / f"{name}.safetensors").read_bytes()).hexdigest()
    if got != meta["tensors_sha256"]:
        raise SystemExit(f"example {name}: tensors file is {got}, its record names {meta['tensors_sha256']}")
    return meta, tensors


def run_example(arm: Arm, name: str, mode: str) -> dict:
    """One fixed example: `correctness` against its expected values, or `measure` of its invocations."""
    import torch  # noqa: PLC0415

    meta, t = _load_example(name)
    dev = "cuda"
    t = {k: v.to(dev) for k, v in t.items()}
    rec: dict = {"example": name, "mode": mode, "what": meta.get("what"), "input_digest": meta["tensors_sha256"]}
    if name == "bf16":
        inp = {"q": t["q"], "kv": t["kv"], "indices": t["indices"], "attn_sink": t["attn_sink"],
               "topk_length": t.get("topk_length"), "sm_scale": meta["sm_scale"]}
        if mode == "measure":
            rec["timing"] = Measure(reps=100).run(lambda: call_bf16(arm, inp), lambda: None)
            return rec
        out, mx, lse = call_bf16(arm, inp)
        rec["checks"] = {"output": compare(out, t["expect_out"], "bf16_out"),
                         "max_logits": compare(mx, t["expect_max_logits"], "bf16_stat",
                                               expect_nonfinite=~torch.isfinite(t["expect_max_logits"])),
                         "lse": compare(lse, t["expect_lse"], "bf16_stat", expect_nonfinite=~torch.isfinite(t["expect_lse"]))}
    elif name == "packed":
        rec["subcases"] = {}
        for sub in meta["subcases"]:
            pools = {}
            for pname in sub["pools"]:
                pools[pname] = {"raw": t[f"{sub['id']}.{pname}.raw"], "page": sub["pools"][pname]["page"],
                                "indices": t[f"{sub['id']}.{pname}.indices"], "lengths": t[f"{sub['id']}.{pname}.lengths"]}
                pools[pname]["view"] = pools[pname]["raw"][:, : pools[pname]["page"] * 584].view(
                    pools[pname]["raw"].shape[0], pools[pname]["page"], 1, 584)
            inp = {"q": t[f"{sub['id']}.q"], "pools": pools, "attn_sink": t[f"{sub['id']}.attn_sink"],
                   "sm_scale": meta["sm_scale"]}
            if mode == "measure":
                sched = arm.flash_mla.get_mla_metadata()[0]
                call_packed(arm, inp, sched)
                rec["subcases"][sub["id"]] = {"timing": Measure(reps=100).run(lambda: call_packed(arm, inp, sched),
                                                                             lambda: None)}
                continue
            out, lse, _ = call_packed(arm, inp)
            exp_lse = t[f"{sub['id']}.expect_lse"]
            rec["subcases"][sub["id"]] = {"checks": {
                "output": compare(out, t[f"{sub['id']}.expect_out"], "packed_out"),
                "lse": compare(lse, exp_lse, "packed_lse", expect_nonfinite=~torch.isfinite(exp_lse))}}
    else:
        rec.update(run_compression_example(arm, meta, t, mode))
    if mode == "correctness":
        leaves = [c for c in _iter_checks(rec)]
        rec["conforms"] = all(c.get("ok") for c in leaves) and bool(leaves)
    return rec


def _iter_checks(rec):
    if isinstance(rec, dict):
        if "policy" in rec and "ok" in rec:
            yield rec
        for v in rec.values():
            yield from _iter_checks(v)


def run_compression_example(arm: Arm, meta: dict, t: dict, mode: str) -> dict:
    """A compression example: one extension of `prefill_tokens`, then `decode_steps` one-token steps, each with its
    post-processing and cache store; the emitted rows and the decoded cache values against the expected ones."""
    import torch  # noqa: PLC0415

    ratio, hd, store = meta["ratio"], meta["head_dim"], meta["store"]
    page = meta["cache_page"]
    req = Request(pages=meta["swa_pages"], length=0)
    state = new_state(max(req.pages) + 2, ratio, hd)
    raw = torch.zeros(meta["cache_pages"], (model1_page_bytes(page) if store == "flashmla" else indexer_page_bytes(page)),
                      dtype=torch.uint8, device="cuda")
    freqs = torch.view_as_complex(t["freqs"].view(-1, D_ROPE // 2, 2).contiguous())
    inputs = t["kv_score"]
    n0 = meta["prefill_tokens"]
    emitted = {}

    def one_call(rows_in, positions, is_prefill):
        if is_prefill:
            plan = arm.dsv4.compress_plan(ratio, len(positions), torch.tensor([positions[-1] + 1]),
                                          torch.tensor([len(positions)]), "cuda")
            idx, ex = prefill_metadata([req], [len(positions)], ratio)
            loc = torch.tensor([(meta["cache_first"] + (p + 1) // ratio - 1) if (p + 1) % ratio == 0 else 0
                                for p in positions], dtype=torch.int32, device="cuda")
        else:
            plan = arm.dsv4.compress_plan(ratio, 1, torch.tensor([positions[0] + 1], dtype=torch.int32, device="cuda"),
                                          None, "cuda")
            req.length = positions[0] + 1
            idx, ex = decode_metadata([req], ratio)
            loc = torch.tensor([(meta["cache_first"] + (positions[0] + 1) // ratio - 1) if (positions[0] + 1) % ratio == 0
                                else 0], dtype=torch.int32, device="cuda")
        out = compress_step(arm, state=state, kv_score_input=rows_in, ape=t["ape"], indices=idx, plan=plan, extra=ex,
                            head_dim=hd, ratio=ratio, norm_weight=t["norm_weight"], eps=meta["eps"], freqs=freqs,
                            cache_raw=raw, cache_page=page, store_type=store, store_loc=loc)
        for j, p in enumerate(positions):
            if (p + 1) % ratio == 0:
                emitted[p] = out[j].clone()
        req.length = positions[-1] + 1

    def run_all():
        emitted.clear()
        req.length = 0
        state.copy_(new_state(max(req.pages) + 2, ratio, hd))
        one_call(inputs[:n0], list(range(n0)), True)
        for p in range(n0, inputs.shape[0]):
            one_call(inputs[p:p + 1], [p], False)
        torch.cuda.synchronize()

    if mode == "measure":
        return {"timing": Measure(reps=10).run(run_all, lambda: None)}
    run_all()
    rec: dict = {"checks": {}}
    positions = sorted(emitted)
    if positions != meta["emitting_positions"]:
        rec["checks"]["events"] = {"policy": "values", "ok": False,
                                   "problems": [f"emitted at {positions}, the example emits at {meta['emitting_positions']}"]}
        return rec
    got = torch.stack([emitted[p] for p in positions])
    rec["checks"]["rows"] = compare(got, t["expect_rows"], "values")
    loc = torch.tensor([meta["cache_first"] + (p + 1) // ratio - 1 for p in positions], dtype=torch.int64)
    want = store_rule(got if store == "flashmla" else got.to(torch.bfloat16), store)
    have = stored_rows(raw, page, loc, store)
    problems = []
    if store == "flashmla":
        nd = int((have["data"] != want["data"]).sum())
    else:
        nd, _ = indexer_payload_mismatches(have["data"], got.to(torch.bfloat16), want["scale"])
    if nd:
        problems.append(f"{nd} stored data byte(s) differ from the store rule applied to the emitted rows")
    if store == "flashmla":
        ns = int((have["scale"] != want["scale"]).sum())
        if ns:
            problems.append(f"{ns} stored E8M0 scale byte(s) differ from the store rule")
    else:
        ulp = (have["scale"].view(torch.int32).to(torch.int64) - want["scale"].view(torch.int32).to(torch.int64)).abs()
        if int(ulp.max()) > 1:
            problems.append(f"a stored FP32 scale is {int(ulp.max())} ULP from the store rule (one allowed)")
    rec["checks"]["cache_bytes"] = {"policy": "store", "ok": not problems, "problems": problems, "rows": len(positions)}
    rec["emitting_positions"] = positions
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", default="/app")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--example", choices=("bf16", "packed", "c4-core", "c4-indexer", "c128"))
    sel.add_argument("--workload", choices=("prefill", "decode"))
    sel.add_argument("--profile-map", metavar="OUT", help="write the tree's component map (execution attribution)")
    ap.add_argument("--mode", choices=("correctness", "measure"), default="correctness")
    ap.add_argument("--seed", type=int, default=20260929, help="the public workload seed (--workload only)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    if args.workload and args.mode != "measure":
        ap.error("--workload measures; use --mode measure")
    if args.profile_map:
        arm = load_arm(Path(args.tree))
        profiles = [profile_family(arm, fam, phase, args.seed + i) for phase in ("prefill", "decode")
                    for i, fam in enumerate(workload_families(phase))]
        cmap = {"identity": arm.identity, "roles": component_map(profiles)}
        Path(args.profile_map).write_text(json.dumps(cmap, indent=1) + "\n")
        for role, r in cmap["roles"].items():
            print(f"{role}: {', '.join(r['base_names']) or '(no operator kernel observed)'}")
        return 0
    t0 = time.monotonic()
    arm = load_arm(Path(args.tree))
    rec: dict = {"identity": arm.identity}
    try:
        if args.example:
            rec.update(run_example(arm, args.example, args.mode))
        else:
            fams = []
            for i, fam in enumerate(workload_families(args.workload)):
                fams.append(measure_family(arm, fam, args.workload, args.seed + i))
                print(f"{fam['family']}: wall {fams[-1]['wall_s'] * 1e3:.3f} ms, gpu {fams[-1]['gpu_s'] * 1e3:.3f} ms "
                      f"(median of {len(fams[-1]['draws'])} draws)", flush=True)
            rec.update({"workload": args.workload, "families": fams})
        rec["structural"] = "valid"
    except NotImplementedError as exc:
        rec.update({"structural": "unavailable", "error": f"NotImplementedError: {exc}"})
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        rec.update({"structural": "failed", "error": f"{type(exc).__name__}: {exc}"[:4000]})
    rec["elapsed_s"] = round(time.monotonic() - t0, 3)
    text = json.dumps(rec, indent=1, default=str)
    if args.output:
        Path(args.output).write_text(text + "\n")
    if rec.get("structural") != "valid":
        print(f"not measured: {rec.get('error')}")
        return 1
    if args.example and args.mode == "correctness":
        print(f"example {args.example}: {'conforms' if rec.get('conforms') else 'does NOT conform'} to its expected values "
              f"under the published tolerances (this does not establish acceptance on held-out cases)")
        for c in _iter_checks(rec):
            if not c.get("ok"):
                print("  " + "; ".join(c.get("problems", [])))
        return 0 if rec.get("conforms") else 2
    print("measured; no pass threshold is applied here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
