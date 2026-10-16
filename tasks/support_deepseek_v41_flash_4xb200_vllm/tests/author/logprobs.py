#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Teacher-forced per-token log-probabilities from the model-author implementation, at model parallelism 4.

    torchrun --nproc-per-node 4 /opt/model-author/logprobs.py \
        --input sequences.json --output logprobs.json \
        [--ckpt-path /model-author/deepseek-v41-mp4] [--config /opt/model-author/inference/config.json]

`sequences.json` is a JSON list of objects with an `id` and `token_ids` (a list of token ids; the first is usually BOS,
id 0), or an object whose `examples` or `sequences` key holds such a list (the fidelity examples' own format). For each
sequence the author's `Transformer` runs one prefill at position 0, and the output head is called with its existing
`full_logits=True` option so that every position's logits are returned instead of only the last; the log-probability of
each next token is `log_softmax` in float32, rounded to 6 decimals. The output is a JSON list of
`{"id", "positions", "logprobs"}` with one value per position after the first.

This is a thin command around the released `inference/` code, which it imports unmodified from beside this file: it
builds the model and loads the converted shards exactly as `inference/generate.py` does, and changes no model math.
It uses the four GPUs the vLLM server uses, so run it while no server holds them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inference"))
sys.path.insert(0, str(HERE / "encoding"))


def _load_sequences(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("examples") or data.get("sequences") or []
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path}: no sequences")
    for s in data:
        ids = s.get("token_ids") if isinstance(s, dict) else None
        if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
            raise SystemExit(f"{path}: a sequence without a list of at least two integer token ids")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt-path", default="/model-author/deepseek-v41-mp4")
    ap.add_argument("--config", default=str(HERE / "inference" / "config.json"))
    a = ap.parse_args(argv)

    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from safetensors.torch import load_model  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from model import ModelArgs, Transformer  # noqa: PLC0415  (the author's, unmodified)

    seqs = _load_sequences(a.input)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    with open(a.config) as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = 1
    args.max_seq_len = max(len(s["token_ids"]) for s in seqs)
    tokenizer = AutoTokenizer.from_pretrained(a.ckpt_path)
    with torch.device("cuda"):
        model = Transformer(args, tokenizer)
    load_model(model, os.path.join(a.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    torch.set_default_device("cuda")
    head_forward = model.head.forward
    model.head.forward = lambda x, full_logits=False: head_forward(x, full_logits=True)  # the author's own option

    out = []
    with torch.inference_mode():
        for s in seqs:
            ids = torch.tensor([s["token_ids"]], dtype=torch.long, device="cuda")
            _, logits, _ = model.forward(ids, 0)
            logp = torch.log_softmax(logits[0].float(), dim=-1)
            nxt = ids[0, 1:]
            values = logp[:-1].gather(-1, nxt[:, None])[:, 0]
            out.append({"id": s.get("id"), "positions": int(nxt.numel()),
                        "logprobs": [round(float(v), 6) for v in values.tolist()]})
            if rank == 0:
                print(f"[author] {s.get('id')}: {nxt.numel()} positions, mean log-prob {float(values.mean()):.6f}",
                      flush=True)
    if rank == 0:
        Path(a.output).write_text(json.dumps(out))
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
