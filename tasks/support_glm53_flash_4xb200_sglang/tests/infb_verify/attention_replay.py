#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""R-ATTENTION's author replay: the released model's complete attention sublayer computed from captured inputs."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
BLOCK = 128
REPRESENTATION = os.environ.get("INFB_GLM53_REPLAY_REPRESENTATION", "deployment_ue8m0")
REPRESENTATIONS = ("deployment_ue8m0", "release_float_scales")
READINGS = ("deployment", "exact")
ATTN_NAMES = ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj",
              "indexer.wq_b", "indexer.wk", "indexer.k_norm", "indexer.weights_proj", "indexer.index_kpool_compress_ape",
              "indexer.index_kpool_compress_gate")


def load_layer(model_dir: str, layer: int, device: str = "cuda") -> dict:
    """The checkpoint's own tensors for one attention layer (`model.language_model.layers.<L>.self_attn.*`)."""
    from safetensors import safe_open
    index = json.loads(Path(model_dir, "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.self_attn."
    keys = sorted(k for k in index if k.startswith(prefix))
    out: dict = {}
    by_file: dict = {}
    for k in keys:
        by_file.setdefault(index[k], []).append(k)
    for fname, ks in by_file.items():
        with safe_open(os.path.join(model_dir, fname), framework="pt", device="cpu") as f:
            for k in ks:
                out[k[len(prefix):]] = f.get_tensor(k).to(device)
    return out


def _dequant_weight(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """e4m3 weight in 128 x 128 blocks times its per-block float32 scale, as float32."""
    out_f, in_f = w.shape
    s = scale_inv.float().repeat_interleave(BLOCK, 0)[:out_f].repeat_interleave(BLOCK, 1)[:, :in_f]
    return w.float() * s


def _ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """The smallest power of two at or above |x| (a UE8M0 scale), elementwise, as float32."""
    bits = x.abs().float().contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = (exp + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254)
    return (exp << 23).view(torch.float32)


def _weight(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """The projection weight the replay multiplies by, as float32, in the frozen representation."""
    wd = _dequant_weight(w, scale_inv)
    if REPRESENTATION == "release_float_scales":
        return wd
    wb = wd.to(torch.bfloat16).float()
    n, k = wb.shape
    pad = torch.zeros(-(-n // BLOCK) * BLOCK, -(-k // BLOCK) * BLOCK, device=wb.device)
    pad[:n, :k] = wb
    v = pad.view(-1, BLOCK, pad.shape[1] // BLOCK, BLOCK)
    sf = _ceil_pow2(v.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4) / FP8_MAX)
    q = (v * (1.0 / sf)).to(torch.float8_e4m3fn).float() * sf
    return q.view(pad.shape)[:n, :k]


def _act_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token groups of 128 channels quantized to e4m3 and returned dequantized: scale = group max |x| / 448, or
    its power-of-two ceiling in the deployment representation."""
    shape = x.shape
    g = x.float().reshape(-1, shape[-1] // BLOCK, BLOCK)
    if REPRESENTATION == "release_float_scales":
        s = (g.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    else:
        s = _ceil_pow2(g.abs().amax(-1, keepdim=True).clamp(min=1e-10) / FP8_MAX)
    q = (g / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(shape)


def _fp8_linear(x: torch.Tensor, w: dict, name: str, reading: str = "deployment") -> torch.Tensor:
    if reading == "exact":
        key = f"{name}.__w__exact"
        if key not in w:
            w[key] = _dequant_weight(w[f"{name}.weight"], w[f"{name}.weight_scale_inv"])
        return (x.float() @ w[key].T).to(torch.bfloat16)
    key = f"{name}.__w__{REPRESENTATION}"
    if key not in w:
        w[key] = _weight(w[f"{name}.weight"], w[f"{name}.weight_scale_inv"])
    return (_act_quant(x) @ w[key].T).to(torch.bfloat16)


def _bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.bfloat16).float() @ weight.to(torch.bfloat16).float().T).to(torch.bfloat16)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * h.to(torch.bfloat16).float()).to(torch.bfloat16)


def _layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(torch.bfloat16)


def _fp8_kv(c: torch.Tensor) -> torch.Tensor:
    return c.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.bfloat16)


def _indexer_topk(w: dict, cfg: dict, x: torch.Tensor, q_resid: torch.Tensor, qpos: torch.Tensor) -> list[torch.Tensor]:
    """`Glm5NextTextIndexer.forward` for one sequence (no padding, all S keys valid): the selected token indices of
    each query position, pools expanded to tokens plus the always-selected tail."""
    S = x.shape[0]
    H, D, kp, topk = cfg["index_n_heads"], cfg["index_head_dim"], cfg["index_kpool"], cfg["index_topk"]
    q = _bf16_linear(q_resid[qpos], w["indexer.wq_b.weight"]).view(-1, H, D)
    k = _layernorm(_bf16_linear(x, w["indexer.wk.weight"]), w["indexer.k_norm.weight"], w["indexer.k_norm.bias"])
    gate = _bf16_linear(x, w["indexer.index_kpool_compress_gate"])
    n_pools = S // kp
    grouped_k = k[: n_pools * kp].view(n_pools, kp, D)
    logits = gate[: n_pools * kp].view(n_pools, kp, D).float() + w["indexer.index_kpool_compress_ape"].float()[None]
    prob = torch.nan_to_num(logits.softmax(dim=1)).to(grouped_k.dtype)
    pool_keys = (prob * grouped_k).sum(dim=1)
    scores = F.relu(torch.einsum("qhd,pd->qhp", q.float(), pool_keys.float()) * D ** -0.5)
    weights = _bf16_linear(x[qpos], w["indexer.weights_proj.weight"]).float() * H ** -0.5
    index_scores = torch.einsum("qh,qhp->qp", weights, scores)
    pool_end = torch.arange(n_pools, device=x.device) * kp + (kp - 1)
    visible = pool_end[None, :] <= qpos[:, None]
    index_scores = index_scores.masked_fill(~visible, torch.finfo(index_scores.dtype).min)
    select_k = min(topk // kp, n_pools)
    out = []
    for i in range(qpos.shape[0]):
        p = int(qpos[i])
        n_vis = int(visible[i].sum())
        if select_k > 0 and n_vis > 0:
            sel = index_scores[i].topk(select_k).indices
            sel = sel[visible[i][sel]]
            toks = (sel[:, None] * kp + torch.arange(kp, device=x.device)[None]).flatten()
        else:
            toks = torch.empty(0, dtype=torch.long, device=x.device)
        visible_count = p + 1
        tail_count = visible_count % kp
        tail = torch.arange(visible_count - tail_count, visible_count, device=x.device)
        out.append(torch.unique(torch.cat([toks, tail])))
    return out


def replay(w: dict, cfg: dict, x_history: torch.Tensor, query_positions: list[int], fp8_kv: bool = True,
           reading: str = "deployment") -> torch.Tensor:
    """The author sublayer output (float32, [Q, hidden]) at `query_positions` of a sequence whose attention-sublayer
    inputs for positions 0..S-1 are `x_history` ([S, hidden], BF16). `fp8_kv` is the recipe's cache precision and stays
    on in every graded use of the deployment reading (I5); the exact reading keeps a BF16 latent."""
    assert REPRESENTATION in REPRESENTATIONS, REPRESENTATION
    assert reading in READINGS, reading
    if reading == "exact":
        fp8_kv = False
    x = x_history.to(torch.bfloat16)
    S = x.shape[0]
    H, Dn, Dv, R = cfg["num_attention_heads"], cfg["qk_nope_head_dim"], cfg["v_head_dim"], cfg["kv_lora_rank"]
    eps = cfg["rms_norm_eps"]
    qpos = torch.tensor(query_positions, device=x.device, dtype=torch.long)
    q_resid = _rmsnorm(_fp8_linear(x, w, "q_a_proj", reading), w["q_a_layernorm.weight"], eps)
    q = _fp8_linear(q_resid[qpos], w, "q_b_proj", reading).view(-1, H, Dn)
    ckv = _fp8_linear(x, w, "kv_a_proj_with_mqa", reading)[:, :R]
    c = _rmsnorm(ckv, w["kv_a_layernorm.weight"], eps)
    if fp8_kv:
        c = _fp8_kv(c)
    kv = _bf16_linear(c, w["kv_b_proj.weight"]).view(S, H, Dn + Dv)
    k, v = kv[..., :Dn], kv[..., Dn:]
    selected = _indexer_topk(w, cfg, x, q_resid, qpos)
    scale = (Dn + cfg.get("qk_rope_head_dim", 0)) ** -0.5
    outs = []
    for i, toks in enumerate(selected):
        toks = toks[toks <= int(qpos[i])]
        att = torch.einsum("hd,thd->ht", q[i].float(), k[toks].float()) * scale
        p = att.softmax(dim=-1)
        o = torch.einsum("ht,thd->hd", p, v[toks].float()).to(torch.bfloat16)
        outs.append(o.reshape(-1))
    attn = torch.stack(outs)
    y = _fp8_linear(attn, w, "o_proj", reading)
    return y.float()


def metric(candidate: torch.Tensor, author: torch.Tensor) -> dict:
    """The public policy's E and maximum absolute difference (`glm53_protocols/attention.py`, the verifier's own copy),
    after the record's own validity checks: the expected shape and finite values."""
    from glm53_protocols.attention import max_abs_difference, relative_rms_error
    c, a = candidate.double(), author.double()
    if c.shape != a.shape:
        return {"valid": False, "problem": f"shape {tuple(c.shape)} against {tuple(a.shape)}"}
    if not torch.isfinite(c).all():
        return {"valid": False, "problem": "the candidate tensor holds a value that is not finite"}
    return {"valid": True, "E": relative_rms_error(c, a), "max_abs": max_abs_difference(c, a),
            "rms_author": float(a.pow(2).mean().sqrt())}


def tau_rule(errors: list[float]) -> float:
    """The public allowance from the calibration errors: ceil(1e6 * max(0.005, 3 * max(E))) / 1e6."""
    return math.ceil(1e6 * max(0.005, 3 * max(errors))) / 1e6
