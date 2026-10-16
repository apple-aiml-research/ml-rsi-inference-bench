#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Chat reasoning, public."""
from __future__ import annotations

from .common import choice, raw_text
from .recipes import SERVED_NAME

THOUGHT_HEAD = "<|channel>thought\n"
MARKS = ("<|channel>", "<channel|>")


def expected_split(raw: str, thinking: bool) -> tuple[str, str]:  # noqa: ARG001 -- the text decides
    if not raw.startswith(THOUGHT_HEAD):
        return "", raw
    reasoning, _, rest = raw[len(THOUGHT_HEAD):].partition("<channel|>")
    return reasoning, rest


def stream_parts(chunks: list[dict]) -> tuple[str, str, list[int]]:
    reasoning, content, ids = [], [], []
    for ch in chunks:
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            reasoning.append(d.get("reasoning_content") or d.get("reasoning") or "")
            content.append(d.get("content") or "")
            ids.extend(c.get("token_ids") or [])
    return "".join(reasoning), "".join(content), ids


def chat_reasoning(srv, cases: list[dict]) -> dict:
    """cases: [{"prompt", "thinking"}]."""
    out, ok = [], True
    for c in cases:
        body = {"model": SERVED_NAME, "max_tokens": 1024, "temperature": 0.0, "return_token_ids": True,
                "messages": [{"role": "user", "content": c["prompt"]}],
                "chat_template_kwargs": {"enable_thinking": bool(c["thinking"])}}
        r = srv.post("/v1/chat/completions", body)
        msg = choice(r).get("message") or {}
        forms = {"plain": (r.get("_status"), msg.get("reasoning_content") or msg.get("reasoning") or "",
                           msg.get("content") or "", raw_text(choice(r).get("token_ids") or []))}
        st, chunks, _ = srv.stream("/v1/chat/completions", body)
        sr, sc, sids = stream_parts(chunks)
        forms["stream"] = (st, sr, sc, raw_text(sids))
        for form, (status, got_r, got_c, raw) in forms.items():
            exp_r, exp_c = expected_split(raw, bool(c["thinking"]))
            good = (status == 200 and got_c.strip() == exp_c.strip() and got_r.strip() == exp_r.strip()
                    and not any(m in got_c + got_r for m in MARKS) and (not c["thinking"] or bool(got_r.strip())))
            out.append({"prompt": c["prompt"][:60], "thinking": c["thinking"], "form": form, "status": status, "pass": good,
                        "opens_thought_block": raw.startswith(THOUGHT_HEAD),
                        "raw": raw[:240], "reasoning": got_r[:160], "content": got_c[:160]})
            ok &= good
    off = [x for x in out if not x["thinking"]]
    rate = sum(x["opens_thought_block"] for x in off) / len(off) if off else None
    return {"cases": out, "thought_block_rate_thinking_disabled": rate, "pass": bool(ok)}
