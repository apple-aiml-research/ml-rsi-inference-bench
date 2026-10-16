#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The grader's final-channel check, shipped as its own client.

The grader asks sixteen fixed questions of its own at low reasoning effort, greedily, through
`/v1/chat/completions`, and requires every answer's assistant `content` to be the model's final-channel
text alone: no Harmony channel token, and none of the channel names a tokenizer leaves behind when it
skips special tokens. This file is the check it runs, byte for byte; the questions beside it are public
ones, not the grader's.

    python3 /app/eval/gpt_oss/final_channel/probe.py --base-url http://127.0.0.1:8000

It prints each answer's verdict and a JSON summary, and exits 1 if any answer is not clean. Standard
library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOKENS = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>", "<|constrain|>")
NAMES = ("assistantanalysis", "assistantcommentary", "assistantfinal")
MAX_TOKENS = 2048


def reasons(content: str, reasoning: str) -> list[str]:
    """Why an answer's content is not the final channel alone; empty when it is clean."""
    found = [f"channel token {t}" for t in TOKENS if t in content]
    found += [f"channel name {n}" for n in NAMES if n in content]
    if content.lstrip().startswith("analysis"):
        found.append("content opens with the analysis channel's name")
    if len(reasoning.strip()) >= 40 and reasoning.strip()[:40] in content:
        found.append("the analysis text is inside content")
    return found


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def served_model(base_url: str, timeout: float = 300.0) -> str:
    """The first model the server lists."""
    return _get(f"{base_url}/v1/models", timeout)["data"][0]["id"]


def ask(base_url: str, model: str, question: str, max_tokens: int = MAX_TOKENS, timeout: float = 900.0) -> dict:
    """One question at low effort, greedily; the answer's content and why it is not clean, if it is not."""
    body = {"model": model, "messages": [{"role": "user", "content": question}], "reasoning_effort": "low",
            "temperature": 0.0, "max_tokens": max_tokens}
    choice = _post(f"{base_url}/v1/chat/completions", body, timeout)["choices"][0]
    message = choice.get("message") or {}
    content, reasoning = message.get("content") or "", message.get("reasoning_content") or ""
    return {"question": question, "finish_reason": choice.get("finish_reason"), "content": content,
            "reasoning_content_chars": len(reasoning), "reasons": reasons(content, reasoning)}


def probe(base_url: str, questions: list[str], model: str | None = None, max_tokens: int = MAX_TOKENS,
          workers: int = 16) -> list[dict]:
    """Every question at once, sixteen at a time; one row per question, in order."""
    base_url = base_url.rstrip("/")
    model = model or served_model(base_url)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda q: ask(base_url, model, q, max_tokens), questions))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    ap.add_argument("--model", default=None, help="default: the first model the server lists")
    a = ap.parse_args()
    rows = probe(a.base_url, json.loads(Path(a.questions).read_text()), a.model)
    for i, r in enumerate(rows):
        verdict = "clean" if not r["reasons"] else "NOT CLEAN: " + "; ".join(r["reasons"])
        print(f"{i:2d} {verdict} | {r['content'][:80]!r}")
    bad = sum(1 for r in rows if r["reasons"])
    print(json.dumps({"questions": len(rows), "clean": len(rows) - bad, "not_clean": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
