#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Greedy-decode a fixed prompt set and record the tokens, so the two arms can be compared."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "In one sentence, a transformer model is",
    "List the first five prime numbers:",
    "The opposite of 'increase' is",
    "A byte consists of",
    "The chemical symbol for gold is",
    "The largest planet in the Solar System is",
]
MAX_TOKENS = 48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--speculative-config", default="")
    args = ap.parse_args()

    os.environ["PYTHONPATH"] = f"{args.repo}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    sys.path.insert(0, args.repo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {"prompts": len(PROMPTS), "max_tokens": MAX_TOKENS, "repo": args.repo}
    try:
        from vllm import LLM, SamplingParams

        kwargs: dict[str, object] = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": 0.90,
            "trust_remote_code": True,
            "seed": 12345,
            "enforce_eager": True,
        }
        if args.speculative_config:
            kwargs["speculative_config"] = json.loads(args.speculative_config)
        llm = LLM(**kwargs)
        outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=12345))
        record["token_ids"] = [list(o.outputs[0].token_ids) for o in outs]
        record["texts"] = [o.outputs[0].text for o in outs]
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- the arm's failure must be recorded, not raised
        import traceback
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-4000:]

    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[equivalence] wrote {out} ok={record.get('ok')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
