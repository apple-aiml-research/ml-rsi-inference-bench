#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""test_tool_calls: the hidden cases through the public protocol's parse
(eval/diffusion_gemma/serving/tools.py), both forms greedy (temperature 0): with no draw the stream
check is judged once. Each stream check's content and calls are recorded. A row grades the draw modes
its case lists in `graded` (the frozen selection; both when absent); a draw left out is still sent and
recorded. Expected on the reference: pass. The selection is read over the reference's own repeated
runs; a reference draw failing one of the two disclosed shapes inside a run is a reference failure
(disposition.py), never a candidate failure.
"""
from __future__ import annotations

from eval.diffusion_gemma.serving.common import raw_text
from eval.diffusion_gemma.serving.recipes import SERVED_NAME
from eval.diffusion_gemma.serving.tools import _served_calls, _stream_calls, _validate, parse_calls


def _stream_check(srv, body: dict, expected_calls) -> dict:
    st, chunks, _ = srv.stream("/v1/chat/completions", body)
    calls, text = _stream_calls(chunks)
    n = expected_calls if expected_calls is not None else len(calls)
    return {"status": st, "calls": calls, "content": text[:600],
            "pass": st == 200 and "<|tool_call>" not in text and len(calls) == n}


def run(srv, cases: list[dict]) -> dict:
    rows, ok = [], True
    for c in cases:
        body = {"model": SERVED_NAME, "max_tokens": 1024, "temperature": 0.0, "messages": c["messages"],
                "tools": c["tools"], "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": True}}
        if c.get("tool_choice"):
            body["tool_choice"] = c["tool_choice"]
        r = srv.post("/v1/chat/completions", body)
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        expected = parse_calls(raw_text(ch.get("token_ids") or []))
        got = _served_calls(msg)
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in c["tools"]}
        valid = all(n in schemas and _validate(a, schemas[n]) for n, a in got)
        good = (r.get("_status") == 200 and got == expected and valid and "<|tool_call>" not in (msg.get("content") or "")
                and len(got) == c.get("expected_calls", len(got)))
        draws = [_stream_check(srv, body, c.get("expected_calls"))]
        s_good = draws[0]["pass"]
        graded = c.get("graded", ["nonstream", "stream"])
        row_pass = (good or "nonstream" not in graded) and (s_good or "stream" not in graded)
        rows.append({"id": c["id"], "status": r.get("_status"), "expected": expected, "served": got, "valid": valid,
                     "nonstream_pass": good, "stream_draws": draws, "graded": graded, "pass": row_pass})
        ok &= row_pass
    return {"rows": rows, "pass": bool(ok)}
