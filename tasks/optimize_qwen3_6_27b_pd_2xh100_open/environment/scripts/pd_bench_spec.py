#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The PD multi-turn bench INPUT: corpus, conversation spec, client argv. One source for both sides."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

PESSIMISTIC_TOKENS_PER_SEC = 500
REQUEST_TIMEOUT_FLOOR_SEC = 900
WORDS_PER_LINE = 18
CORPUS_LINE_FLOOR = 24000
CORPUS_WORDS = (
    "system", "request", "response", "latency", "prefill", "decode", "token", "cache",
    "kernel", "tensor", "router", "engine", "memory", "throughput", "attention",
    "sequence", "batch", "schedule", "transfer", "gradient", "context", "window",
)


def request_timeout_sec(pd: dict, concurrency: int) -> int:
    """Seconds to allow one conversation turn before calling it hung."""
    override = os.environ.get("INFB_PD_REQUEST_TIMEOUT_SEC", "").strip()
    if override:
        return int(override)
    worst_turn_tokens = (int(pd.get("first_turn_prefix_tokens", 8000))
                         + int(pd["new_tokens_per_turn"])
                         + int(pd["reply_tokens"])) * max(1, concurrency)
    return max(REQUEST_TIMEOUT_FLOOR_SEC, worst_turn_tokens // PESSIMISTIC_TOKENS_PER_SEC)


def corpus_line_count(pd: dict) -> int:
    """Lines the generated corpus needs: `conversations x first_turn_prefix_tokens`, because
    `bench_dataset.generate_conversations` advances `base_offset` after EVERY conversation. +1 token
    because the client's assertion is strict."""
    needed_tokens = int(pd.get("conversations", 16)) * int(pd.get("first_turn_prefix_tokens", 8000)) + 1
    return max(CORPUS_LINE_FLOOR, -(-(needed_tokens * 3) // WORDS_PER_LINE))


def write_corpus(pd: dict, out: Path, *, seed: int | None = None) -> Path:
    """Generate the deterministic pseudo-corpus, or leave an existing one alone."""
    lines = corpus_line_count(pd)
    if out.is_file():
        return out
    if seed is None:
        seed = int(os.environ.get("INFB_BENCH_SEED", "1024"))
    rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for _ in range(lines):
            fh.write(" ".join(rng.choice(CORPUS_WORDS) for _ in range(WORDS_PER_LINE)) + "." + chr(10))
    needed_tokens = int(pd.get("conversations", 16)) * int(pd.get("first_turn_prefix_tokens", 8000)) + 1
    if lines * WORDS_PER_LINE < needed_tokens:
        raise SystemExit(
            f"FATAL: the generated PD corpus cannot cover the spec: {pd.get('conversations', 16)} "
            f"conversations x {pd.get('first_turn_prefix_tokens', 8000)} prefix tokens = {needed_tokens} "
            f"tokens needed, corpus word floor is {lines * WORDS_PER_LINE}."
        )
    return out


def conversation_spec(pd: dict, corpus: Path) -> dict:
    """The `generate_conversations` input the client reads. Every value comes from the pd table."""
    return {
        "filetype": "generate_conversations",
        "num_conversations": int(pd.get("conversations", 16)),
        "text_files": [str(corpus)],
        "print_stats": False,
        "prompt_input": {
            "num_turns": {"distribution": "constant",
                          "value": int(pd.get("conversation_turns", 8))},
            "common_prefix_num_tokens": {"distribution": "constant", "value": 0},
            "prefix_num_tokens": {"distribution": "constant",
                                  "value": int(pd.get("first_turn_prefix_tokens", 8000))},
            "num_tokens": {"distribution": "constant", "value": int(pd["new_tokens_per_turn"])},
        },
        "prompt_output": {
            "num_tokens": {"distribution": "constant", "value": int(pd["reply_tokens"])},
        },
    }


def write_spec(pd: dict, workdir: Path, *, seed: int | None = None) -> Path:
    """Generate the corpus and write the conversation spec beside it. Returns the spec path."""
    workdir.mkdir(parents=True, exist_ok=True)
    corpus = write_corpus(pd, workdir / "corpus_generated.txt", seed=seed)
    spec = workdir / "conversations.json"
    spec.write_text(json.dumps(conversation_spec(pd, corpus), indent=2) + chr(10), encoding="utf-8")
    return spec


def build_bench_argv(*, client: Path, venv: Path, pd: dict, model_path: str, router_port: int,
                     stats_out: Path, served_model_name: str, concurrency: int,
                     spec: Path) -> list[str]:
    """The client argv. Four of these flags are not optional, each for a measured reason:

    * `--served-model-name`: the pools are started with it, so without it the client sends a filesystem
      path as the model id, every request returns `prefill HTTP 404 ... does not exist`, and the run
      finishes almost instantly having completed zero conversations.
    * `--no-early-stop`: without it the client stops the remaining clients once enough conversations
      finish, so the stats file holds fewer requests than the spec asked for while still reporting
      `num_failures=0` on both arms -- a silent subset rather than an error.
    * `--request-timeout-sec`: the client's own default is 120 s, while turn 1 alone is
      16 x (8000 + 12000) = 320,000 prompt tokens to prefill, so the default gives up on every
      conversation while both pools are still answering HTTP 200. It is a HANG detector, not a
      performance bound, which is why `request_timeout_sec` divides by a pessimistic token rate.
    * `--input-file`: it must be the GENERATED spec, not "any .txt beside the client" -- vLLM ships a
      sample .txt in `benchmarks/multi_turn/` and an SGLang tree has no multi-turn client at all, so
      taking whatever is there benches two different corpora in the two arms.
    """
    return [
        str(venv / "bin" / "python"), str(client),
        "--model", model_path,
        "--served-model-name", served_model_name,
        "--url", "http://127.0.0.1:" + str(router_port),
        "--input-file", str(spec),
        "--num-clients", str(concurrency),
        "--max-active-conversations", str(concurrency),
        "--no-early-stop",
        "--request-timeout-sec", str(request_timeout_sec(pd, concurrency)),
        "--stats-json-output", str(stats_out),
    ]


def bench_env() -> dict:
    """The client's environment: the router is on loopback, so loopback must not be proxied."""
    env = dict(os.environ)
    loop = "127.0.0.1,localhost"
    for key in ("no_proxy", "NO_PROXY"):
        env[key] = ",".join([x for x in [env.get(key, ""), loop] if x])
    return env
