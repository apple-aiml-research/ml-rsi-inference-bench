#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private enclosing-region comparator: released weights, trusted arithmetic, independent mathematics.

The expected output of a captured native region is computed from the checkpoint's own tensors
(`/model` safetensors, identified by name), PyTorch's own linear arithmetic, and the independent
QSA/PLE mathematics of the trusted public copy (`qwen38.reference.qsa`, `qwen38.reference.ple`)
over each request's *logical* history. It never calls a candidate module, reads a candidate cache
or imports a candidate metadata type. History is the committed rows of earlier steps (rejected
speculative rows are dropped); rows of the current step are visible causally to later rows of the
same step.

QSA selection starts from the mathematics' own top-k over the independent index scores; when an
output disagrees at the public allowance, the admissible selections within the score allowance are
searched (`QSA.admissible`). All QSA and PLE projections of this checkpoint are BF16 (only MoE
experts are FP8); the n-gram table is FP8 with a per-tensor scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen38.reference import ple as rp
from qwen38.reference import qsa as rq

QSA_ALLOWANCE = rq.OUTPUT_ALLOWANCE
PLE_ALLOWANCE = rp.OUTPUT_ALLOWANCE
SELECTION_ALLOWANCE = rq.SELECTION_ALLOWANCE
MAX_SWAPS = 32


class Checkpoint:
    """Tensors of the released checkpoint by name, loaded on demand to one device."""

    def __init__(self, root: str | Path = "/model", device: str | torch.device = "cuda"):
        self.root = Path(root)
        self.device = device
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = json.loads((self.root / "config.json").read_text())["text_config"]
        self._cache = {}

    def tensor(self, name: str, device=None) -> torch.Tensor:
        from safetensors import safe_open
        key = (name, str(device or self.device))
        if key not in self._cache:
            with safe_open(str(self.root / self.index[name]), framework="pt", device="cpu") as f:
                self._cache[key] = f.get_tensor(name).to(device or self.device)
        return self._cache[key]

    def rows(self, name: str, index: torch.Tensor) -> torch.Tensor:
        """Selected rows of a large tensor without loading it whole."""
        from safetensors import safe_open
        index = index.cpu()
        with safe_open(str(self.root / self.index[name]), framework="pt", device="cpu") as f:
            sl = f.get_slice(name)
            out = [sl[int(i):int(i) + 1] for i in index.tolist()]
        return torch.cat(out).to(self.device) if out else torch.empty(0)

    def shape(self, name: str) -> list[int]:
        from safetensors import safe_open
        with safe_open(str(self.root / self.index[name]), framework="pt", device="cpu") as f:
            return list(f.get_slice(name).get_shape())


def rope_tables(config, positions: torch.Tensor):
    params = config["rope_parameters"]
    return rq.rope_tables(positions, head_dim=config["head_dim"],
                          partial_rotary_factor=params["partial_rotary_factor"], rope_theta=params["rope_theta"])


def compare(actual: torch.Tensor, expected: torch.Tensor, allowance: dict, name: str) -> dict:
    """Every element within atol + rtol·|expected|; returns the error summary, raising on violation."""
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} differs from {tuple(expected.shape)}")
    a, e = actual.float(), expected.float()
    if not torch.isfinite(a).all():
        raise AssertionError(f"{name}: nonfinite native output")
    diff = (a - e).abs()
    bound = allowance["atol"] + allowance["rtol"] * e.abs()
    bad = int((diff > bound).sum())
    summary = {"elements": diff.numel(), "max_abs": float(diff.max()) if diff.numel() else 0.0,
               "mean_abs": float(diff.mean()) if diff.numel() else 0.0, "violations": bad}
    if bad:
        worst = int(torch.argmax(diff - bound))
        raise AssertionError(f"{name}: {bad} of {diff.numel()} elements exceed the allowance; worst "
                             f"flat index {worst}: native {float(a.flatten()[worst])} expected {float(e.flatten()[worst])}")
    return summary


class QSA:
    """One QSA layer: logical per-request history and outputs for query rows."""

    def __init__(self, ckpt: Checkpoint, layer: int):
        c = ckpt.config
        p = f"model.language_model.layers.{layer}.self_attn."
        self.c, self.layer = c, layer
        self.wq, self.wk, self.wv, self.wo = (ckpt.tensor(p + n + ".weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.qn, self.kn = ckpt.tensor(p + "q_norm.weight"), ckpt.tensor(p + "k_norm.weight")
        self.wi = ckpt.tensor(p + "indexer.index_qk_proj.weight")
        self.iqn, self.ikn = ckpt.tensor(p + "indexer.q_layernorm.weight"), ckpt.tensor(p + "indexer.k_layernorm.weight")
        self.eps = c["rms_norm_eps"]
        self.hd, self.nh, self.nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        self.ih, self.ihd = c["indexer_n_heads"], c["indexer_head_dim"]
        self.ratio, self.budget = c["indexer_compress_ratio"], c["indexer_budget"]

    def rows(self, hidden, positions):
        """Per-row logical K, V and raw index keys (the state a row contributes to history)."""
        k = F.linear(hidden, self.wk).view(-1, self.nkv, self.hd)
        v = F.linear(hidden, self.wv).view(-1, self.nkv, self.hd)
        cos, sin = rope_tables(self.c, positions)
        k = rq.rope(rq.gemma_norm(k, self.kn, self.eps), cos, sin)
        ik = F.linear(hidden, self.wi)[:, self.ih * self.ihd:].reshape(-1, self.ihd)
        return {"k": k, "v": v, "ik": ik}

    def _queries(self, hidden, positions):
        q_all = F.linear(hidden, self.wq).view(-1, self.nh, 2 * self.hd)
        q, gate = q_all[..., :self.hd], q_all[..., self.hd:]
        cos, sin = rope_tables(self.c, positions)
        q = rq.rope(rq.gemma_norm(q, self.qn, self.eps), cos, sin)
        iq = F.linear(hidden, self.wi)[:, :self.ih * self.ihd].reshape(-1, self.ih, self.ihd)
        return q, gate, rq.rope(rq.gemma_norm(iq, self.iqn, self.eps), cos, sin)

    def _compressed(self, IK):
        return rq.compressed_keys(IK, self.ikn, ratio=self.ratio, eps=self.eps,
                                  tables=lambda positions: rope_tables(self.c, positions))

    def _scores(self, iq_row, ck, visible):
        return rq.index_scores(iq_row, ck, visible)

    def _project(self, attended, gate_row):
        return F.linear(attended.to(torch.bfloat16).flatten().unsqueeze(0) * torch.sigmoid(gate_row.flatten()),
                        self.wo)[0]

    def outputs(self, hidden, positions, context: dict) -> tuple[torch.Tensor, list]:
        """Outputs for query rows (`hidden`, `positions`), each seeing `context` rows 0..its position,
        with the mathematics' own top-k selection."""
        q, gate, iq = self._queries(hidden, positions)
        K, V = context["k"], context["v"]
        ck = self._compressed(context["ik"])
        out = torch.zeros(hidden.shape[0], self.nh, self.hd, dtype=torch.float32, device=hidden.device)
        margins = []
        for i, p in enumerate(positions.tolist()):
            visible = rq.visible_groups(p, self.ratio)
            if visible:
                logits = self._scores(iq[i], ck, visible)
                groups = rq.select(logits, budget_groups=self.budget // self.ratio)
                if groups.numel() < visible:
                    rest = logits.clone(); rest[groups] = -float("inf")
                    margins.append(float(logits[groups].min() - rest.max()))
            else:
                groups = torch.empty(0, dtype=torch.long, device=hidden.device)
            out[i] = rq.attend(q[i], K, V, rq.attended_tokens(groups, p, self.ratio), scale=self.hd ** -0.5)
        return rq.gated_output(out, gate, self.wo), margins

    def admissible(self, hidden_row, position: int, context: dict, native_row, allowance=QSA_ALLOWANCE):
        """Search the admissible selections of one row for one whose output agrees with `native_row`.

        Admissible: the weakest selected group scores at most the selection allowance below the strongest
        omitted one. Starting from this comparator's top-k, groups inside the boundary window are swapped
        (each swap the one that most reduces the worst allowance excess) until the output agrees, no swap
        improves it, or MAX_SWAPS swaps were made. Returns (output or None, evidence)."""
        q, gate, iq = self._queries(hidden_row, torch.tensor([position], device=hidden_row.device))
        K, V = context["k"], context["v"]
        ck = self._compressed(context["ik"])
        visible = (position + 1) // self.ratio
        want = min(self.budget // self.ratio, visible)
        evidence = {"position": position, "visible_groups": visible, "selected_groups": want, "swaps": []}
        if not visible or want >= visible:
            return None, dict(evidence, reason="no selection freedom")
        s = self._scores(iq[0], ck, visible)
        order = torch.argsort(s, descending=True)
        kth = float(s[order[want - 1]])
        slack = SELECTION_ALLOWANCE["atol"] + SELECTION_ALLOWANCE["rtol"] * abs(kth)
        window = ((s >= kth - slack) & (s <= kth + slack)).nonzero().flatten()
        certain = (s > kth + slack).nonzero().flatten()
        selected = set(order[:want].tolist())
        chosen = set(g for g in window.tolist() if g in selected)
        evidence.update(window=int(window.numel()), certain=int(certain.numel()), boundary_score=kth, slack=slack)
        rep, scale = self.nh // self.nkv, self.hd ** -0.5
        tail = torch.arange(visible * self.ratio, position + 1, device=K.device)
        span = torch.arange(self.ratio, device=K.device)
        pool = torch.cat((certain, window))
        tokens = (pool[:, None] * self.ratio + span).flatten()
        every = torch.cat((tokens, tail))
        logits = torch.einsum("hd,shd->hs", q[0].float(), K[every].float().repeat_interleave(rep, dim=1)) * scale
        shift = logits.max(-1, keepdim=True).values
        e = torch.exp(logits - shift)                                     # (h, tokens)
        vv = V[every].float().repeat_interleave(rep, dim=1)               # (tokens, h, d)
        num = torch.einsum("hs,shd->shd", e, vv)
        groups = pool.numel()
        g_num = num[:groups * self.ratio].view(groups, self.ratio, self.nh, self.hd).sum(1)
        g_den = e[:, :groups * self.ratio].view(self.nh, groups, self.ratio).sum(-1).T   # (groups, h)
        t_num, t_den = num[groups * self.ratio:].sum(0), e[:, groups * self.ratio:].sum(-1)
        position_of = {int(g): i for i, g in enumerate(pool.tolist())}
        base = [position_of[g] for g in certain.tolist()]
        native = native_row.float()
        bound = allowance["atol"] + allowance["rtol"] * native.abs()

        def output(members):
            idx = torch.tensor(members, device=K.device, dtype=torch.long)
            n = g_num[idx].sum(0) + t_num
            d = g_den[idx].sum(0) + t_den
            return self._project(n / d[:, None], gate[0])

        def excess(y):
            return float(((y.float() - native).abs() - allowance["atol"] - allowance["rtol"] * y.float().abs()).max())

        members = base + [position_of[g] for g in chosen]
        y = output(members)
        worst = excess(y)
        inside = sorted(position_of[g] for g in chosen)
        outside = sorted(position_of[g] for g in window.tolist() if g not in chosen)
        while worst > 0 and len(evidence["swaps"]) < MAX_SWAPS and inside and outside:
            n0 = g_num[members].sum(0) + t_num
            d0 = g_den[members].sum(0) + t_den
            out_i = torch.tensor(inside, device=K.device)
            in_j = torch.tensor(outside, device=K.device)
            best = (worst, None)
            for chunk in in_j.split(max(1, 4096 // max(1, len(inside)))):
                n1 = n0[None, None] - g_num[out_i][:, None] + g_num[chunk][None, :]   # (i, j, h, d)
                d1 = d0[None, None] - g_den[out_i][:, None] + g_den[chunk][None, :]   # (i, j, h)
                att = (n1 / d1[..., None]).to(torch.bfloat16).flatten(2)
                ys = F.linear(att * torch.sigmoid(gate[0].flatten()), self.wo).float()
                ex = ((ys - native).abs() - allowance["atol"] - allowance["rtol"] * ys.abs()).amax(-1)
                value, where = torch.min(ex.flatten(), 0)
                if float(value) < best[0]:
                    a, b = divmod(int(where), chunk.numel())
                    best = (float(value), (inside[a], int(chunk[b])))
            if best[1] is None:
                break
            gone, came = best[1]
            inside.remove(gone); outside.remove(came); inside.append(came); outside.append(gone)
            members = base + inside
            evidence["swaps"].append({"out": int(pool[gone]), "in": int(pool[came]), "excess": best[0]})
            worst = best[0]
        if worst > 0:
            return None, dict(evidence, reason="no admissible selection within the search reproduces the output",
                              excess=worst)
        # Admissibility of the final selection over every visible group (the public definition).
        final = torch.tensor(sorted(set(int(pool[i]) for i in members)), device=s.device)
        ok, margin = rq.admissible(s, final)
        if not ok or final.numel() != want:
            return None, dict(evidence, reason="the agreeing selection is not admissible", margin=margin)
        return output(members), dict(evidence, margin=margin, excess=worst)


class PLE:
    """The PLE layer: n-gram lookup from the released table, gating and dilated causal convolution."""

    def __init__(self, ckpt: Checkpoint, layer: int):
        c = ckpt.config
        p = f"model.language_model.layers.{layer}.ple."
        self.ckpt, self.c, self.prefix = ckpt, c, p
        self.hidden, self.hc = c["hidden_size"], c["hc_count"]
        self.C = self.hidden * self.hc
        self.kernel, self.dilation = c["ple_conv_kernel_size"], c["ngram_size"]
        self.width = (self.kernel - 1) * self.dilation
        self.context_len = c["ngram_size"] - 1
        self.heads_per_ngram = c["heads_per_ngram"]
        eos = c["eos_token_id"]
        self.eos = eos[0] if isinstance(eos, list) else eos
        self.eps = c["rms_norm_eps"]
        self.multipliers = ckpt.tensor(p + "ple_embedding.layer_multipliers", "cpu").long()
        self.sizes = ckpt.tensor(p + "ple_embedding.ngram_heads_vocab_sizes", "cpu").long()
        self.offsets = ckpt.tensor(p + "ple_embedding.ngram_heads_offsets", "cpu").long()
        self.scale = ckpt.tensor(p + "ple_embedding.ngram_embedding.weight_scale")
        self.shard_rows = ckpt.shape(p + "ple_embedding.ngram_embedding.shard_0.weight")[0]
        self.wk = ckpt.tensor(p + "key_proj.weight")
        self.wv = ckpt.tensor(p + "value_proj.weight")
        self.nk, self.nq, self.nc = (ckpt.tensor(p + n + ".weight") for n in ("norm_key", "norm_query", "norm_conv"))
        self.conv = ckpt.tensor(p + "conv1d.weight").reshape(self.C, self.kernel)

    def ngram_ids(self, context: list[int], tokens: list[int]) -> torch.Tensor:
        return rp.ngram_rows(context, tokens, multipliers=self.multipliers, sizes=self.sizes, offsets=self.offsets,
                             heads_per_ngram=self.heads_per_ngram, context_len=self.context_len, eos=self.eos)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.flatten()
        unique, inverse = torch.unique(flat, return_inverse=True)
        table = []
        for shard in torch.unique(unique // self.shard_rows).tolist():
            members = unique[(unique // self.shard_rows) == shard]
            rows = self.ckpt.rows(self.prefix + f"ple_embedding.ngram_embedding.shard_{shard}.weight",
                                  members - shard * self.shard_rows)
            table.append((members, rows))
        dim = table[0][1].shape[-1]
        values = torch.empty(unique.numel(), dim, dtype=torch.bfloat16, device=self.scale.device)
        lookup = {int(u): i for i, u in enumerate(unique.tolist())}
        for members, rows in table:
            where = torch.tensor([lookup[int(m)] for m in members.tolist()], device=values.device)
            values[where] = rp.dequantize(rows, self.scale)
        return values[inverse.to(values.device)].view(*ids.shape, dim).flatten(-2)

    def rows(self, hidden, context_tokens, tokens):
        """Per-row gated value (residual) and convolution input."""
        emb = self.embed(self.ngram_ids(context_tokens, tokens))
        return rp.gate(F.linear(emb, self.wk), F.linear(emb, self.wv), hidden, norm_key=self.nk,
                       norm_query=self.nq, norm_conv=self.nc, streams=self.hc, eps=self.eps)

    def outputs(self, gated, conv_in, history_conv):
        """Outputs for a step's rows given the committed conv inputs before them (zeros when fresh)."""
        return rp.convolution_output(gated, conv_in, history_conv, self.conv, dilation=self.dilation)
