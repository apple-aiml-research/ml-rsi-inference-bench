#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""GSM8K, public: the upstream engine's GSM8K protocol for this model."""
from __future__ import annotations

import json
import re
import time

from .common import content, gather
from .recipes import SERVED_NAME

DATA = "/public/datasets/gsm8k"


def get_answer_value(s: str):
    s = s.replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def gsm8k(srv, data_dir: str = DATA, concurrency: int = 16, limit: int | None = None) -> dict:
    train = [json.loads(l) for l in open(f"{data_dir}/main_train.jsonl")]
    test = [json.loads(l) for l in open(f"{data_dir}/main_test.jsonl")][:limit]
    shots = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(5))

    def one(q):
        r = srv.post("/v1/chat/completions", {"model": SERVED_NAME, "temperature": 1.0, "max_tokens": 256,
                                              "stop": ["Question", "Assistant:", "<|separator|>"],
                                              "messages": [{"role": "user", "content": shots + f"Question: {q['question']}\nAnswer:"}]})
        want = get_answer_value(q["answer"].split("####")[-1])
        return want is not None and get_answer_value(content(r)) == want, r.get("_status")

    t0 = time.monotonic()
    res = gather(one, test, concurrency=concurrency)
    return {"n": len(res), "accuracy": sum(x[0] for x in res) / max(len(res), 1), "wall_s": round(time.monotonic() - t0, 1),
            "statuses": sorted({x[1] for x in res}, key=str)}
