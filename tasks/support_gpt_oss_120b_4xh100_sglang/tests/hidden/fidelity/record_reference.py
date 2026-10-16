#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Record the model-author reference log-probabilities for the fidelity sequences.

Authoring step, run ONCE under torchrun; never at grading, never in the image.

    torchrun --nproc-per-node=4 record_reference.py --seqs <dir> --checkpoint <original/>

The reference is `openai/gpt-oss` v0.0.1 (`gpt_oss/torch/model.py`), vendored in this task under
`environment/public/gpt_oss/` and read from there rather than downloaded, so the code that produced
these numbers ships beside them. It is independent of the engine in the way the bar needs: `Checkpoint`
expands the MXFP4 blocks through the `FP4_VALUES` table with `scales - 127` exponents into bf16 and
multiplies in a general matrix routine, so no Triton kernel is shared with SGLang; `AttentionBlock`
computes the sinks inside `sdpa` and applies the 128-token window on the sliding layers; `MLPBlock`
shards the intermediate dimension over `torch.distributed` ranks with an all-reduce.

`Transformer.forward(tokens)` returns logits at every position with no KV cache, so the teacher-forced
log-probabilities of a whole sequence are one forward call. Positions 1..T-1 are recorded (each is the
log-probability the reference assigns to the token that actually follows), which is what the verifier
compares against `meta_info.input_token_logprobs` from `POST /generate`.

The log-softmax is chunked over positions on purpose: a 4,608-token sequence has a [4608, 201088]
logits tensor, and materialising its float32 softmax at once costs about 3.7 GB on top of a rank that
already holds a quarter of a 234 GB bf16 model.

The long band is recorded at world 8 with `--q-chunk 1024` (see `chunk_sdpa`), over the files
`build_long_band.py` writes:

    torchrun --nproc-per-node=8 record_reference.py --seqs <dir> --checkpoint <original/> \
        --reference-root /public --mlp-chunk 64 --q-chunk 1024
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

CHUNK = 512


def chunk_mlp(model, chunk: int) -> int:
    """Make every MLPBlock evaluate its experts in token chunks. Arithmetic-preserving, by construction.

    Why this is needed, measured on 4xH100: `MLPBlock.forward` gathers the selected
    experts' weights per token -- `self.mlp1_weight[expert_indices, ...]` is
    [T, experts_per_token, 2*intermediate/world, hidden] -- which at world=4 costs **33.2 MB per
    token** for mlp1 and 16.6 MB for mlp2. The weights already occupy 72.9 GiB of each 79.2 GiB card,
    so a 501-token forward asks for 7.74 GiB against 4.27 GiB free and the run dies. The reference was
    written to generate one token at a time; teacher-forcing a 4,608-token sequence is not what it was
    built for.

    Why chunking changes no number: the MLP is **token-wise**. Every operation between the gather and
    the weighted expert sum -- the two einsums, swiglu, the bias adds, the final `bec,be->bc` -- indexes
    the batch axis independently, and nothing accumulates across tokens. Splitting the batch axis
    therefore feeds each token exactly the same operands in the same order. The one operation that is
    not token-local is `dist.all_reduce`, which sums each token's partial result across ranks; called
    per chunk it sums the same partials for the same tokens. `--verify-chunking` proves this on a real
    sequence instead of arguing it: the same sequence is run chunked and unchunked and the
    log-probabilities are compared.
    """
    from gpt_oss.torch.model import MLPBlock

    if chunk <= 0:                      # --mlp-chunk 0: no patch at all, for the determinism check
        return 0
    if getattr(MLPBlock, "_infb_chunked", False):
        return chunk
    original = MLPBlock.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= chunk:
            return original(self, x)
        return torch.cat([original(self, x[s : s + chunk]) for s in range(0, x.shape[0], chunk)], dim=0)

    MLPBlock.forward = forward
    MLPBlock._infb_chunked = True
    return chunk


def chunk_sdpa(chunk: int) -> int:
    """Make the reference's `sdpa` evaluate its query rows in chunks. Arithmetic-preserving per row; measured.

    Why this is needed: the reference replicates attention on every rank, and `sdpa`
    materialises [heads, q_mult, T, T + 1] scores for all 64 query heads -- 32 GiB in bf16 at T = 16,384, twice
    over for the sink concatenation, on any world size. The long band's 16,384-token sequence cannot be recorded
    otherwise.

    Why it changes no number: every query row's masked softmax is independent of the other rows -- the causal
    and sliding masks and the sink are per row -- so evaluating Q in chunks of rows feeds each row the same
    operands, and only the GEMM tiling can differ. This was measured instead of assumed: on both 8,192-token
    sequences at world 8, the chunked recording is bitwise identical to the unchunked one.
    """
    import gpt_oss.torch.model as reference

    if chunk <= 0:                      # --q-chunk 0: the reference's own sdpa, unchanged
        return 0
    if getattr(reference, "_infb_q_chunked", False):
        return chunk
    original = reference.sdpa

    def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
        n_tokens, n_heads, q_mult, d_head = Q.shape
        if n_tokens <= chunk:
            return original(Q, K, V, S, sm_scale, sliding_window)
        Kx = K[:, :, None, :].expand(-1, -1, q_mult, -1)
        Vx = V[:, :, None, :].expand(-1, -1, q_mult, -1)
        out = []
        for q0 in range(0, n_tokens, chunk):
            q1 = min(q0 + chunk, n_tokens)
            # the reference's own operations, on query rows q0..q1-1 against every key: row r is position q0 + r
            mask = torch.triu(Q.new_full((q1 - q0, n_tokens), -float("inf")), diagonal=q0 + 1)
            if sliding_window > 0:
                mask += torch.tril(mask.new_full((q1 - q0, n_tokens), -float("inf")), diagonal=q0 - sliding_window)
            QK = torch.einsum("qhmd,khmd->hmqk", Q[q0:q1], Kx)
            QK *= sm_scale
            QK += mask[None, None, :, :]
            QK = torch.cat([QK, S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, q1 - q0, -1)], dim=-1)
            W = torch.softmax(QK, dim=-1)[..., :-1]
            out.append(torch.einsum("hmqk,khmd->qhmd", W, Vx).reshape(q1 - q0, -1))
            del QK, W, mask
        return torch.cat(out, 0)

    reference.sdpa = sdpa
    reference._infb_q_chunked = True
    return chunk


def logprobs(model, ids: list[int], device) -> list[float]:
    """Teacher-forced log-probabilities for positions 1..T-1, chunked over positions in float32."""
    x = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)                                      # [T, vocab], bf16
        lp = torch.empty(len(ids) - 1, dtype=torch.float32, device=device)
        for s in range(0, len(ids) - 1, CHUNK):
            e = min(s + CHUNK, len(ids) - 1)
            block = torch.log_softmax(logits[s:e].float(), dim=-1)
            lp[s:e] = block.gather(-1, x[s + 1 : e + 1, None]).squeeze(-1)
            del block
        values = lp.cpu().tolist()
        del logits, lp
    torch.cuda.empty_cache()
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", required=True, help="directory of <id>.json sequence files")
    ap.add_argument("--checkpoint", required=True, help="the original/ checkpoint directory")
    ap.add_argument("--reference-root", required=True, help="directory holding the gpt_oss package")
    ap.add_argument("--reference-commit", default="49316946", help="openai/gpt-oss v0.0.1")
    ap.add_argument("--mlp-chunk", type=int, default=64, help="tokens per expert gather; see chunk_mlp")
    ap.add_argument("--verify-chunking", default="", help="a sequence id to run both ways first")
    ap.add_argument("--verify-tokens", type=int, default=128,
                    help="prefix length for the chunking check; the unchunked arm caps it")
    ap.add_argument("--only", default="", help="comma-separated ids to record (default: all)")
    ap.add_argument("--q-chunk", type=int, default=0, help="query rows per sdpa evaluation; see chunk_sdpa")
    args = ap.parse_args()

    sys.path.insert(0, args.reference_root)
    from gpt_oss.torch.model import Transformer
    from gpt_oss.torch.utils import init_distributed

    chunk_sdpa(args.q_chunk)

    device = init_distributed()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    seqs = sorted(p for p in pathlib.Path(args.seqs).glob("*.json") if p.name != "index.json")
    if args.only:
        want = set(args.only.split(","))
        seqs = [p for p in seqs if p.stem in want]

    t0 = time.time()
    model = Transformer.from_checkpoint(args.checkpoint, device=device)
    load_s = time.time() - t0
    if rank == 0:
        free, total = torch.cuda.mem_get_info(device)
        print(f"[ref] loaded in {load_s:.0f} s; rank 0 holds {(total-free)/2**30:.1f} GiB of "
              f"{total/2**30:.1f} GiB; world_size={world}; {len(seqs)} sequences; sdpa query chunk "
              f"{args.q_chunk or 'none'}", flush=True)

    if args.verify_chunking:
        # On a SHORT PREFIX, because the unchunked arm is the arm that cannot run: at world=4 the
        # mlp1 gather costs 33.2 MB per token against about 4.7 GiB free, so roughly 140 tokens is the
        # unchunked ceiling. 128 tokens with a 64-token chunk still splits the batch axis twice, which
        # is what the equivalence claim needs; a longer prefix would only re-measure the OOM.
        p = pathlib.Path(args.seqs) / f"{args.verify_chunking}.json"
        ids = json.loads(p.read_text())["token_ids"][: args.verify_tokens]
        assert args.mlp_chunk <= 0 or len(ids) > args.mlp_chunk, \
            "the prefix must exceed the chunk or nothing is split"
        unchunked = logprobs(model, ids, device)
        chunk_mlp(model, args.mlp_chunk)
        chunked = logprobs(model, ids, device)
        worst = max(abs(a - b) for a, b in zip(unchunked, chunked))
        identical = unchunked == chunked
        if rank == 0:
            print(f"[ref] chunking check on {args.verify_chunking}[:{len(ids)}] with chunk="
                  f"{args.mlp_chunk}: bitwise identical={identical}  max |delta|={worst:.3e}  over "
                  f"{len(chunked)} values", flush=True)
        if worst > 1e-5:
            if rank == 0:
                print("[ref] FAILURE: chunking changed the numbers by more than 1e-5", flush=True)
            return 1
    else:
        chunk_mlp(model, args.mlp_chunk)

    failures = []
    for p in seqs:
        rec = json.loads(p.read_text())
        ids = rec["token_ids"]
        t1 = time.time()
        try:
            values = logprobs(model, ids, device)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            failures.append((rec["id"], len(ids), str(exc).split(".")[0]))
            if rank == 0:
                print(f"[ref] {rec['id']:11} T={len(ids):5} OOM: {str(exc)[:90]}", flush=True)
            continue
        if rank == 0:
            rec["reference"] = {
                "name": "openai/gpt-oss v0.0.1 gpt_oss.torch.model.Transformer",
                "commit": args.reference_commit,
                "checkpoint": "original/ layout of openai/gpt-oss-120b at revision "
                              "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
                "world_size": world,
                "mlp_token_chunk": args.mlp_chunk,
                "dtype": "bf16 weights, float32 log_softmax",
                "positions": "1..T-1, i.e. the log-probability of each token given its prefix",
                "n_values": len(values),
                "logprobs": values,
            }
            p.write_text(json.dumps(rec) + "\n")
            print(f"[ref] {rec['id']:11} T={len(ids):5} recorded {len(values)} values in "
                  f"{time.time() - t1:.1f} s  mean lp {sum(values)/len(values):+.4f}", flush=True)

    if rank == 0:
        print(f"[ref] done: {len(seqs) - len(failures)}/{len(seqs)} sequences in "
              f"{time.time() - t0:.0f} s", flush=True)
        for i, t, why in failures:
            print(f"[ref] NOT RECORDED {i} (T={t}): {why}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
