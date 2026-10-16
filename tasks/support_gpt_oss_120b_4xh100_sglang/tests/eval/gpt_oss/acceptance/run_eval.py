#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Measure GPQA accuracy against an already running server.

The small local examples exercise the same scoring and request implementation as
the private full-set check. This command reports measurements without a floor.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .simple_eval_common import ChatCompletionSampler, make_report, set_ulimit
from .simple_eval_gpqa import GPQAEval


def evaluate(*, filename, base_url, model="openai/gpt-oss-120b", num_examples=None,
             num_threads=198, max_tokens=4096, temperature=0.1,
             reasoning_effort="low", seed=0, output=None):
    set_ulimit()
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    evaluator = GPQAEval(str(filename), num_examples, num_threads, seed=seed)
    sampler = ChatCompletionSampler(model=model, base_url=base_url.rstrip("/")+"/v1",
                                    max_tokens=max_tokens, temperature=temperature,
                                    reasoning_effort=reasoning_effort)
    start = time.monotonic()
    result = evaluator(sampler)
    metrics = result.metrics | {"score": result.score, "questions": len(evaluator.examples),
                                "seconds": time.monotonic()-start}
    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
        (output / "report.html").write_text(make_report(result))
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=Path(__file__).with_name("examples.csv"))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--num-examples", type=int)
    p.add_argument("--num-threads", type=int, default=198)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = vars(p.parse_args())
    a["filename"] = a.pop("inputs")
    evaluate(**a)


if __name__ == "__main__":
    main()
