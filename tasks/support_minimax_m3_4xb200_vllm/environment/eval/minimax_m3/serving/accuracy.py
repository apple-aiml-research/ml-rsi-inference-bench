#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""GSM8K through chat completions with thinking (adapted from SGLang's MiniMax-M3 accuracy test).

A last integer of more than MAX_DIGITS digits reads as no answer (wrong): every GSM8K test answer has at most seven
digits, so 64 leaves a wide margin above any answer while a run of digits that long can only be a degenerate output;
converting it would not change the verdict and, past Python's 4,300-digit limit, would raise instead of scoring.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .fidelity import MODEL
from .http import request

DATASET = Path("/public/datasets/gsm8k/main_test.jsonl")
SUFFIX = "\n\nPlease reason step by step, and give the final answer as a single integer on the last line."
INT = re.compile(r"-?\d+")
MAX_DIGITS = 64


def last_integer(text: str):
    found = INT.findall(text.replace(",", ""))
    if not found or len(found[-1].lstrip("-")) > MAX_DIGITS:
        return None
    return int(found[-1])


def gold(answer: str) -> int:
    return int(answer.split("####")[-1].strip().replace(",", ""))


def ask(base, question, *, model=MODEL, timeout=3600):
    body = {"model": model, "messages": [{"role": "user", "content": question + SUFFIX}], "temperature": 0.0,
            "max_tokens": 4096, "chat_template_kwargs": {"thinking_mode": "enabled"}}
    resp = request(base, "/v1/chat/completions", body, timeout=timeout)
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("reasoning") or "") + "\n" + (msg.get("content") or ""), (resp.get("choices") or [{}])[0].get(
        "finish_reason")


def evaluate(base, *, limit=None, model=MODEL, concurrency=64, dataset=DATASET) -> dict:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outs = list(pool.map(lambda r: ask(base, r["question"], model=model), rows))
    records = []
    for i, (row, (text, finish)) in enumerate(zip(rows, outs)):
        pred = last_integer(text)
        records.append({"index": i, "gold": gold(row["answer"]), "pred": pred, "finish_reason": finish,
                        "correct": pred == gold(row["answer"])})
    score = sum(r["correct"] for r in records) / len(records)
    return {"questions": len(records), "accuracy": score, "wall_s": round(time.monotonic() - t0, 1),
            "records": records}
