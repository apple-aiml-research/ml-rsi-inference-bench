#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Throughput harness for this task."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SPEC_TOKENS = 3


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link.resolve())
    raise SystemExit("no model: set INFB_MODEL_PATH or stage <workspace>/model")


def _drafter() -> str:
    """The EAGLE3 drafter directory — a SECOND checkpoint, and the engine will not start without it."""
    direct = os.environ.get("INFB_SPEC_DRAFTER_PATH")
    if direct and Path(direct, "config.json").exists():
        return direct
    for entry in (os.environ.get("INFB_AUX_MODEL_PATHS") or "").split(os.pathsep):
        entry = entry.strip()
        if entry and "eagle3" in Path(entry).name.lower() and Path(entry, "config.json").exists():
            return entry
    baked = "/opt/infb/aux-models/nvidia/gpt-oss-120b-Eagle3"
    if Path(baked, "config.json").exists():
        return baked
    raise SystemExit(
        "no speculative drafter: set INFB_SPEC_DRAFTER_PATH to a directory holding the "
        "nvidia/gpt-oss-120b-Eagle3 checkpoint (config.json + model.safetensors). "
        "environment/Dockerfile bakes it there at build time; if this fires, that build step was "
        "skipped or the image is not this task's."
    )


def _build_llm(model: str, drafter: str, *, max_num_seqs: int, max_model_len: int,
               enforce_eager: bool):
    from vllm import LLM

    extra = {}
    if not enforce_eager:
        extra["compilation_config"] = {
            "max_cudagraph_capture_size": (1 + SPEC_TOKENS) * max_num_seqs,
        }
    return LLM(
        model=model,
        # The drafter publishes no tokenizer of its own, so the target's is used for both.
        tokenizer=model,
        async_scheduling=True,
        # Set on BOTH arms, so enabling speculation is the verifier's action and never the agent's
        # lever -- the agent can only make already-on speculation faster, and cannot raise its score
        # by turning speculation off.
        speculative_config={
            "method": "eagle3",
            "model": drafter,
            "num_speculative_tokens": SPEC_TOKENS,
        },
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
        seed=12345,
        **extra,
    )


def _sampling(out_len: int, temperature: float, top_p: float, top_k: int,
              *, n: int, jitter: float):
    """Identical to the scored harness; see public/workloads.json for the request shape."""
    from vllm import SamplingParams

    rng = random.Random(20260913)
    lo, hi = int(out_len * (1.0 - jitter)), int(out_len * (1.0 + jitter))
    lo = max(8, lo)
    hi = max(lo, hi)
    return [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ignore_eos=True,                # the length is the draw above, not the model's choice
            max_tokens=(out_len if jitter <= 0 else rng.randint(lo, hi)),
            detokenize=False,
        )
        for _ in range(n)
    ]


def _prompts(n: int, in_len: int, vocab: int):
    """Random token-id prompts: no tokenizer round-trip, so the shape is exact."""
    from vllm import TokensPrompt

    rng = random.Random(12345)
    return [
        TokensPrompt(prompt_token_ids=[rng.randrange(16, max(32, vocab - 16)) for _ in range(in_len)])
        for _ in range(n)
    ]


def _vocab_size(model: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return int(getattr(cfg, "vocab_size", 32000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path(os.environ.get("INFB_WORKSPACE", "/app")))
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--output-len-jitter", type=float, default=0.0,
                    help="per-request max_tokens ~ U[L(1-j), L(1+j)]. The scored run uses 0. Only "
                         "the reduced-scale probe needs a non-zero value.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    ws = args.workspace
    model = _model(ws)
    drafter = _drafter()


    llm = _build_llm(
        model,
        drafter,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    prompts = _prompts(args.num_prompts, args.input_len, _vocab_size(model))
    sampling = _sampling(args.output_len, args.temperature, args.top_p, args.top_k,
                         n=args.num_prompts, jitter=args.output_len_jitter)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    produced = sum(len(c.token_ids) for o in outs for c in o.outputs)
    total = args.num_prompts * args.input_len + produced
    result = {
        "elapsed_time": elapsed,
        "num_requests": args.num_prompts,
        "total_num_tokens": total,
        "total_output_tokens": produced,
        "requests_per_second": args.num_prompts / elapsed,
        "tokens_per_second": total / elapsed,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
