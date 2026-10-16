#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm.
# Modified by Apple Inc. in 2026.
"""Parsers: one F2P node, test_gemma4_tool_parser_whole_canvas_streaming.

It is the conjunction of the landing's two added streaming tests of tests/tool_parsers/
test_gemma4_tool_parser.py -- test_streaming_single_chunk_complete_tool_call and
test_streaming_multi_chunk_batched_tool_calls -- adapted to obtain the parser through
ToolParserManager.get_tool_parser("gemma4") with /model's tokenizer (real token ids for every
delta) and to assert only through extract_tool_calls_streaming. Each function's own result is
recorded in the node's JUnit properties.

Expected: pass on the oracle; fail on the prepared base (a whole call in one delta is swallowed).
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from infb_verify import common

MODEL = "/model"


def _parser_and_tokenizer():
    from transformers import AutoTokenizer
    from vllm.tool_parsers import ToolParserManager

    tok = AutoTokenizer.from_pretrained(MODEL)
    return ToolParserManager.get_tool_parser("gemma4")(tok), tok


def _request():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


def _simulate_streaming(parser, tok, request, chunks: list[str]) -> list[tuple[Any, str]]:
    results, previous_text, previous_ids = [], "", []
    for chunk in chunks:
        current_text = previous_text + chunk
        delta_ids = tok.encode(chunk, add_special_tokens=False)
        current_ids = previous_ids + delta_ids
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text, current_text=current_text, delta_text=chunk,
            previous_token_ids=tuple(previous_ids), current_token_ids=tuple(current_ids),
            delta_token_ids=tuple(delta_ids), request=request)
        results.append((delta, current_text))
        previous_text, previous_ids = current_text, current_ids
    return results


def _collect_tool_calls_by_index(results) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for delta, _ in results:
        if not (delta and delta.tool_calls):
            continue
        for tc in delta.tool_calls:
            entry = by_index.setdefault(tc.index, {"name": None, "arguments": ""})
            func = tc.function
            if isinstance(func, dict):
                name, arg = func.get("name"), func.get("arguments", "")
            else:
                name, arg = getattr(func, "name", None), getattr(func, "arguments", "") or ""
            if name:
                entry["name"] = name
            if arg:
                entry["arguments"] += arg
    return by_index


def single_chunk_complete_tool_call():
    """The landing's test_streaming_single_chunk_complete_tool_call."""
    parser, tok = _parser_and_tokenizer()
    chunks = ['<|tool_call>call:name_a_color{color_hex:<|"|>00ff11<|"|>}<tool_call|>']
    results = _simulate_streaming(parser, tok, _request(), chunks)
    tool_call_deltas = [d for d, _ in results if d is not None and d.tool_calls]
    assert len(tool_call_deltas) == 1, "Expected exactly one delta carrying the batched tool call"
    assert all(d.content is None for d, _ in results if d is not None), "Complete tool call must not leak as content"
    by_index = _collect_tool_calls_by_index(results)
    assert set(by_index) == {0}
    assert by_index[0]["name"] == "name_a_color"
    assert json.loads(by_index[0]["arguments"]) == {"color_hex": "00ff11"}


def multi_chunk_batched_tool_calls():
    """The landing's test_streaming_multi_chunk_batched_tool_calls."""
    parser, tok = _parser_and_tokenizer()
    chunks = ['<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
              '<|tool_call>call:get_time{timezone:<|"|>GMT<|"|>}<tool_call|>']
    results = _simulate_streaming(parser, tok, _request(), chunks)
    by_index = _collect_tool_calls_by_index(results)
    assert set(by_index) == {0, 1}, f"Expected two tool calls (indices 0 and 1), got {sorted(by_index)}"
    assert by_index[0]["name"] == "get_weather"
    assert json.loads(by_index[0]["arguments"]) == {"location": "London"}
    assert by_index[1]["name"] == "get_time"
    assert json.loads(by_index[1]["arguments"]) == {"timezone": "GMT"}


def test_gemma4_tool_parser_whole_canvas_streaming(record_property):
    outcomes = {}
    for name, fn in (("test_streaming_single_chunk_complete_tool_call", single_chunk_complete_tool_call),
                     ("test_streaming_multi_chunk_batched_tool_calls", multi_chunk_batched_tool_calls)):
        try:
            fn()
            outcomes[name] = "passed"
        except AssertionError as exc:
            outcomes[name] = f"failed: {exc}"[:500]
        except Exception as exc:  # noqa: BLE001 -- recorded as the function's failure
            outcomes[name] = f"error: {type(exc).__name__}: {exc}"[:500]
        record_property(name, outcomes[name])
    common.record("test_gemma4_tool_parser_whole_canvas_streaming", {"functions": outcomes})
    assert all(v == "passed" for v in outcomes.values()), outcomes
