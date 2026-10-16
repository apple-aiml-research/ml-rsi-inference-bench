#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Helpers the serving protocols share (public): /model's tokenizer, bounded concurrency, and a
response's raw generated text.
"""
from __future__ import annotations

import concurrent.futures as cf
import json

MODEL = "/model"
_TOKENIZER = None
_EOS: set[int] | None = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    return _TOKENIZER


def gather(fn, items, concurrency: int = 4) -> list:
    """fn over items with at most `concurrency` in flight, results in order."""
    with cf.ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(fn, items))


def eos_ids() -> set[int]:
    """The checkpoint's end-of-sequence ids (generation_config.json)."""
    global _EOS
    if _EOS is None:
        with open(f"{MODEL}/generation_config.json") as fh:
            eos = json.load(fh).get("eos_token_id")
        _EOS = set(eos if isinstance(eos, list) else [eos])
    return _EOS


def raw_text(ids: list[int]) -> str:
    """A response's own generated ids decoded with skip_special_tokens=False, its terminating EOS dropped."""
    ids = list(ids)
    if ids and ids[-1] in eos_ids():
        ids = ids[:-1]
    return tokenizer().decode(ids, skip_special_tokens=False)


def choice(res: dict) -> dict:
    return (res.get("choices") or [{}])[0]


def content(res: dict) -> str:
    return (choice(res).get("message") or {}).get("content") or ""


def completion_tokens(res: dict):
    return (res.get("usage") or {}).get("completion_tokens")
