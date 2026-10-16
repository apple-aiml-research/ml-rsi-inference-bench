#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The quality protocol: GSM8K through vLLM's own evaluator, against a running server.

    python quality.py --base-url http://127.0.0.1:8000 [--limit 200 | --all] [--app /app]

The evaluator is the engine's own `tests/evals/gsm8k/gsm8k_eval.py`, called as its own test calls it for a
configuration with the harness defaults: 5-shot completion prompts, greedy decoding (temperature 0), at most 256
generated tokens, seed 42, every question in flight at once, answers scored by its own parser; its per-request
timeout is the public limit for one HTTP correctness workload (900 s, the README's operation limits) rather than the
evaluator's 600 s default, so no inner timeout is shorter than the operation it bounds. Its data loader
would fetch the public train and test files from the network; here it reads the offline copies in
`../datasets/gsm8k/` instead. This script selects which test questions run (the first `--limit`, or all 1,319) and
prints the evaluator's result, the number correct and DSpark's acceptance length from the server's own counters.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent / "datasets" / "gsm8k"
NUM_SHOTS = 5
MAX_TOKENS = 256
TEMPERATURE = 0.0
SEED = 42
#: The public operation limit for one HTTP correctness workload below 64,000 input tokens (README): the batch is one
#: explicitly submitted concurrent batch, every request of it bounded by this.
REQUEST_TIMEOUT_S = 900


def evaluator(app: Path):
    """The engine's GSM8K evaluator module, from the tree at `app`."""
    path = app / "tests" / "evals" / "gsm8k" / "gsm8k_eval.py"
    spec = importlib.util.spec_from_file_location("infb_gsm8k_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_gsm8k(ge, host: str, port: int, train_path: Path, test_path: Path, indices: list[int] | None) -> tuple[dict, int]:
    """Run the evaluator on the test questions at `indices` (all of them when None), with the offline data.
    Returns the evaluator's result and the number of questions it ran."""
    train = list(ge.read_jsonl(str(train_path)))
    full = list(ge.read_jsonl(str(test_path)))
    test = full if indices is None else [full[i] for i in indices]
    ge.load_gsm8k_data = lambda: (train, test)
    result = ge.evaluate_gsm8k(
        num_questions=len(test), num_shots=NUM_SHOTS, max_tokens=MAX_TOKENS, host=host, port=int(port),
        temperature=TEMPERATURE, seed=SEED, request_timeout_seconds=REQUEST_TIMEOUT_S,
    )
    return result, len(test)


_METRIC = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)\s*$")


def counter_total(metrics_text: str, name: str) -> float:
    """Sum a Prometheus counter over every label set (the bare name or its `_total` spelling)."""
    wanted = {name, f"{name}_total"}
    total = 0.0
    seen = False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC.match(line)
        if m and m.group("name") in wanted:
            total += float(m.group("value"))
            seen = True
    return total if seen else float("nan")


def acceptance_length(metrics_text: str) -> float:
    """DSpark's mean acceptance length, 1 + accepted / drafts, from the server's speculative-decoding counters."""
    drafts = counter_total(metrics_text, "vllm:spec_decode_num_drafts")
    accepted = counter_total(metrics_text, "vllm:spec_decode_num_accepted_tokens")
    return (1.0 + accepted / drafts) if drafts and drafts == drafts and drafts > 0 else float("nan")


def result_record(result: dict, n: int) -> dict:
    """The evaluator's result as printed: its fields, and the number correct."""
    record = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in result.items()}
    record["correct"] = round(float(result["accuracy"]) * n)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--limit", type=int, default=200, help="the first N test questions (default 200)")
    g.add_argument("--all", action="store_true", help="all 1,319 test questions")
    ap.add_argument("--app", default="/app", help="the engine tree whose evaluator runs")
    ap.add_argument("--datasets", default=str(DATASETS))
    a = ap.parse_args(argv)
    ge = evaluator(Path(a.app))
    host, port = a.base_url.rstrip("/").rsplit(":", 1)
    datasets = Path(a.datasets)
    n_test = sum(1 for line in (datasets / "test.jsonl").read_text().splitlines() if line.strip())
    indices = None if a.all else list(range(min(a.limit, n_test)))
    result, n = run_gsm8k(ge, host, int(port), datasets / "train.jsonl", datasets / "test.jsonl", indices)
    record = result_record(result, n)
    try:
        with urllib.request.urlopen(a.base_url.rstrip("/") + "/metrics", timeout=60) as r:
            record["dspark_acceptance_length"] = acceptance_length(r.read().decode())
    except OSError as exc:
        record["dspark_acceptance_length"] = float("nan")
        print(f"(no metrics: {exc})")
    print(json.dumps(record, indent=1))
    print(f"{record['correct']} of {n} correct ({record['accuracy']:.4f}); invalid rate {record.get('invalid_rate', math.nan):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
