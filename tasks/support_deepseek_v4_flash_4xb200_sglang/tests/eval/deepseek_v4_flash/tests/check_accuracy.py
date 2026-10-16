#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""GSM8K on a running server, through upstream's own evaluator (`python/sglang/test/run_eval.py`): 5-shot, the completion
API, greedy, 512 max tokens, last-number scoring, 128 client threads. `gsm8k_args` is the invocation the grader
makes, shared byte for byte with it; the grader runs it on its own held-out selection of questions."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

MODEL = "/model/deepseek-ai/DeepSeek-V4-Flash"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gsm8k_test.jsonl"


def gsm8k_args(base_url: str, model: str, data_path: str, num_examples: int, num_shots: int) -> SimpleNamespace:
    """upstream's run_eval arguments for GSM8K at this task's protocol."""
    return SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        api="completion",
        max_tokens=512,
        num_examples=num_examples,
        num_threads=128,
        num_shots=num_shots,
        gsm8k_data_path=data_path,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    from sglang.test.run_eval import run_eval  # noqa: PLC0415 - the engine's own evaluator

    ap = argparse.ArgumentParser(description="GSM8K accuracy on a running server (measured, no pass mark)")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=str(FIXTURE))
    ap.add_argument("--num-examples", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    args = ap.parse_args(argv)
    metrics = run_eval(gsm8k_args(args.base_url, args.model, args.data, args.num_examples, args.num_shots))
    print(json.dumps({k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}, sort_keys=True,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
