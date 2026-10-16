#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm.
# Modified by Apple Inc. in 2026.
"""Adapted upstream parser tests: four fail-to-pass nodes, one per parser and mode.

The functions are the reference's own unit tests for its reasoning parser (fifteen) and tool-call parser (six), plus,
when the reference admits the streaming-by-text-markers change, the seven reasoning functions that change adds
(`config.json` `infb.parsers.admit_split_marker_functions`). They are rewritten to obtain the parsers the way the
serving layer does -- through `ReasoningParserManager` / `ToolParserManager` under the recipe's name `minimax_m3`,
constructed with `/model`'s tokenizer (`vllm.tokenizers.get_tokenizer`) -- and to assert only through the base-class
interfaces. Every node is the conjunction of its functions; each function's own result is a JUnit property of the
node. The adaptations, function by function, are in `tests/_upstream/PROVENANCE.md`.
"""

from __future__ import annotations

import functools
import json
import os
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
MODEL = os.environ.get("INFB_MODEL_DIR", "/model")
NAME = "minimax_m3"
NS = "]<]minimax[>["


def admit_split_marker_functions() -> bool:
    return bool(json.loads((TESTS_DIR / "config.json").read_text())["infb"]["parsers"]["admit_split_marker_functions"])


@functools.cache
def model_tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(MODEL)


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode_per_character(tokenizer, text: str) -> list[int]:
    """Marker text as a model can emit it: one ordinary token per character (no added-token match)."""
    return [token_id for ch in text for token_id in tokenizer.encode(ch, add_special_tokens=False)]


class SplitMarkerTokenizer:
    """`/model`'s tokenizer whose encodes split everything into characters while its vocabulary still has the marker
    entries (the upstream `SplitMiniMaxM3Tokenizer`, over the real vocabulary)."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __len__(self) -> int:
        return len(self._base)

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        return encode_per_character(self._base, text)

    def tokenize(self, text: str, *args, **kwargs) -> list[str]:
        return [token for ch in text for token in self._base.tokenize(ch)]


# ---------------------------------------------------------------------------------------------------------------------
# Reasoning parser


def reasoning_parser_cls():
    from vllm.reasoning import ReasoningParserManager

    return ReasoningParserManager.get_reasoning_parser(NAME)


def make_parser(chat_template_kwargs: dict[str, str] | None = None, tokenizer=None):
    tokenizer = model_tokenizer() if tokenizer is None else tokenizer
    return reasoning_parser_cls()(tokenizer, chat_template_kwargs=chat_template_kwargs), tokenizer


def chat_request():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    return ChatCompletionRequest(messages=[], model="test-model")


def run_streaming(parser, encode_delta: Callable[[str], list[int]], chunks: list[str]):
    previous_text = ""
    previous_token_ids: list[int] = []
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_end_states: list[bool] = []

    for chunk in chunks:
        delta_token_ids = encode_delta(chunk)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        reasoning_end_states.append(parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids))

        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return "".join(reasoning_parts) or None, "".join(content_parts) or None, reasoning_end_states


def streaming(parser, tokenizer, chunks):
    return run_streaming(parser, lambda text: encode(tokenizer, text), chunks)


def streaming_runtime_split(parser, tokenizer, chunks):
    return run_streaming(parser, lambda text: encode_per_character(tokenizer, text), chunks)


def parser_registration():
    from vllm.reasoning import ReasoningParser

    parser_cls = reasoning_parser_cls()

    assert isinstance(parser_cls, type) and issubclass(parser_cls, ReasoningParser)


def nonstreaming_extracts_explicit_reasoning_block():
    parser, _ = make_parser()

    reasoning, content = parser.extract_reasoning("<mm:think>plan</mm:think>answer", chat_request())

    assert reasoning == "plan"
    assert content == "answer"


def nonstreaming_without_start_tag_is_content():
    parser, _ = make_parser()

    reasoning, content = parser.extract_reasoning("plain answer", chat_request())

    assert reasoning is None
    assert content == "plain answer"


def nonstreaming_drops_leading_end_tag():
    parser, _ = make_parser()

    reasoning, content = parser.extract_reasoning("</mm:think>answer", chat_request())

    assert reasoning is None
    assert content == "answer"


def nonstreaming_non_leading_end_tag_is_content():
    parser, _ = make_parser()

    reasoning, content = parser.extract_reasoning("XXX</mm:think>YYY", chat_request())

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"


def nonstreaming_enabled_mode_starts_in_reasoning():
    parser, _ = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})

    reasoning, content = parser.extract_reasoning("plan</mm:think>answer", chat_request())

    assert reasoning == "plan"
    assert content == "answer"


def nonstreaming_open_reasoning_block():
    parser, _ = make_parser()

    reasoning, content = parser.extract_reasoning("<mm:think>still thinking", chat_request())

    assert reasoning == "still thinking"
    assert content is None


def streaming_reasoning_tags_are_not_returned():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = streaming(parser, tokenizer, ["<mm:think>", "plan", "</mm:think>", "answer"])

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states[2]  # the step the end tag arrives on; whether the state then stays set is not graded


def streaming_boundary_can_emit_reasoning_and_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = streaming(parser, tokenizer, ["<mm:think>plan</mm:think>answer"])

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [True]


def streaming_drops_leading_end_tag():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = streaming(parser, tokenizer, ["</mm:think>", "answer"])

    assert reasoning is None
    assert content == "answer"
    assert end_states[0]  # the step the end tag arrives on; whether the state then stays set is not graded


def streaming_non_leading_end_tag_is_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = streaming(parser, tokenizer, ["XXX</mm:think>YYY"])

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"
    assert end_states == [True]


def streaming_enabled_mode_starts_in_reasoning():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})

    reasoning, content, end_states = streaming(parser, tokenizer, ["plan", "</mm:think>", "answer"])

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states[1]  # the step the end tag arrives on; whether the state then stays set is not graded


def streaming_plain_content_ends_reasoning_phase():
    parser, tokenizer = make_parser()

    reasoning, content, _ = streaming(parser, tokenizer, ["plain ", "answer"])

    assert reasoning is None
    assert content == "plain answer"
    # no end tag arrives, so no reasoning-end state is graded here


def token_id_helpers():
    parser, tokenizer = make_parser()
    output_ids = encode(tokenizer, "<mm:think>abc</mm:think>def")
    open_reasoning_ids = encode(tokenizer, "<mm:think>abc")
    content_ids = encode(tokenizer, "plain")

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert not parser.is_reasoning_end(content_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.extract_content_ids(content_ids) == content_ids
    assert parser.count_reasoning_tokens(output_ids) == len(encode(tokenizer, "abc"))


def token_id_helpers_enabled_mode():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    output_ids = encode(tokenizer, "abc</mm:think>def")
    open_reasoning_ids = encode(tokenizer, "abc")

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.count_reasoning_tokens(output_ids) == len(encode(tokenizer, "abc"))
    assert parser.count_reasoning_tokens(open_reasoning_ids) == len(encode(tokenizer, "abc"))


# The seven functions of the streaming-by-text-markers change: markers arrive as ordinary per-character tokens.


def streaming_split_marker_tokens_are_not_returned():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = streaming_runtime_split(
        parser, tokenizer, ["<mm:think>", "Reasoning", " content", "</mm:think>", "content"]
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, False, True, True]


def streaming_split_marker_text_drives_end_state():
    parser, tokenizer = make_parser()
    previous_text = ""
    previous_token_ids: list[int] = []

    for chunk in ["<mm:think>", "Reasoning", " content", "</mm:think>"]:
        delta_token_ids = encode_per_character(tokenizer, chunk)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids
        parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        previous_text = current_text
        previous_token_ids = current_token_ids

    assert parser.is_reasoning_end_streaming(previous_token_ids, []) is True


def streaming_split_end_marker_content_ids_are_stripped():
    parser, tokenizer = make_parser()
    previous_text = "<mm:think>Reasoning"
    previous_token_ids = encode_per_character(tokenizer, previous_text)
    delta_text = "</mm:think>content"
    delta_token_ids = encode_per_character(tokenizer, delta_text)
    current_token_ids = previous_token_ids + delta_token_ids

    parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=previous_text + delta_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids,
    )

    assert parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids)
    assert tokenizer.decode(parser.extract_content_ids(delta_token_ids)) == "content"


def streaming_split_marker_tokens_enabled_mode():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})

    reasoning, content, end_states = streaming_runtime_split(
        parser, tokenizer, ["Reasoning", " content", "</mm:think>", "content"]
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, True, True]


def streaming_split_marker_text_across_deltas():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = streaming_runtime_split(
        parser, tokenizer, ["<mm:", "think>", "Reasoning", " content", "</mm:", "think>", "content"]
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, False, False, False, True, True]


def streaming_split_leading_end_marker_text_across_deltas():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = streaming_runtime_split(parser, tokenizer, ["</mm:", "think>", "content"])

    assert reasoning is None
    assert content == "content"
    assert end_states == [False, True, True]


def token_id_helpers_with_split_marker_tokens():
    tokenizer = SplitMarkerTokenizer(model_tokenizer())
    parser, _ = make_parser(tokenizer=tokenizer)
    output_ids = tokenizer.encode("<mm:think>abc</mm:think>def", add_special_tokens=False)
    open_reasoning_ids = tokenizer.encode("<mm:think>abc", add_special_tokens=False)
    content_ids = tokenizer.encode("plain", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert not parser.is_reasoning_end(content_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.extract_content_ids(content_ids) == content_ids
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))


# ---------------------------------------------------------------------------------------------------------------------
# Tool-call parser


def sample_tools():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam, FunctionDefinition

    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="create_order",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "urgent": {"type": "boolean"},
                        "note": {"type": "string"},
                        "shipping": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}, "zip": {"type": "integer"}},
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
                            },
                        },
                        "metadata": {"type": "object", "additionalProperties": {"type": "string"}},
                        "duplicate_demo": {"type": "object"},
                    },
                },
            ),
        )
    ]


def tool_parser_cls():
    from vllm.tool_parsers import ToolParserManager

    return ToolParserManager.get_tool_parser(NAME)


def tool_parser():
    return tool_parser_cls()(model_tokenizer(), sample_tools())


def build_order_call() -> str:
    return (
        f"{NS}<tool_call>\n"
        f'{NS}<invoke name="create_order">'
        f"{NS}<user_id>42{NS}</user_id>"
        f"{NS}<urgent>true{NS}</urgent>"
        f"{NS}<note>Please leave at front desk.{NS}</note>"
        f"{NS}<shipping>"
        f"{NS}<city>Singapore{NS}</city>"
        f"{NS}<zip>018956{NS}</zip>"
        f"{NS}</shipping>"
        f"{NS}<items>"
        f"{NS}<item>{NS}<sku>book-001{NS}</sku>{NS}<qty>2{NS}</qty>{NS}</item>"
        f"{NS}<item>{NS}<sku>pen-007{NS}</sku>{NS}<qty>5{NS}</qty>{NS}</item>"
        f"{NS}</items>"
        f"{NS}<metadata>"
        f"{NS}<source>mobile{NS}</source>"
        f"{NS}<campaign>may-launch{NS}</campaign>"
        f"{NS}</metadata>"
        # upstream's `duplicate_demo` (a key repeated at one level, read as an array) leaves the input
        f"{NS}</invoke>\n"
        f"{NS}</tool_call>"
    )


def build_order_invocation(user_id: int) -> str:
    return f'{NS}<invoke name="create_order">{NS}<user_id>{user_id}{NS}</user_id>{NS}</invoke>'


def build_multiple_order_call() -> str:
    return f"{NS}<tool_call>\n{build_order_invocation(1)}\n{build_order_invocation(2)}\n{NS}</tool_call>"


def feed(parser, chunks: list[str | tuple[str, list[int]]]) -> list:
    previous = ""
    results = []
    for chunk in chunks:
        if isinstance(chunk, tuple):
            delta, delta_ids = chunk
        else:
            delta = chunk
            delta_ids = []

        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=delta_ids,
            request=None,
        )
        if result is not None:
            results.append(result)
        previous = current
    return results


def collect_content(results: list) -> str:
    return "".join(result.content for result in results if result.content)


def collect_tool_calls(results: list) -> dict[int, dict[str, Any]]:
    tool_calls: dict[int, dict[str, Any]] = {}
    for result in results:
        for tool_call in result.tool_calls or []:
            tool_calls.setdefault(tool_call.index, {"id": None, "name": "", "arguments": ""})
            if tool_call.id:
                tool_calls[tool_call.index]["id"] = tool_call.id
            if tool_call.function:
                if tool_call.function.name:
                    tool_calls[tool_call.index]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls[tool_call.index]["arguments"] += tool_call.function.arguments
    return tool_calls


def minimax_m3_parser_registered():
    from vllm.tool_parsers import ToolParser

    parser_cls = tool_parser_cls()

    assert isinstance(parser_cls, type) and issubclass(parser_cls, ToolParser)


def non_streaming_nested_tool_call():
    parser = tool_parser()
    result = parser.extract_tool_calls("I will create it.\n" + build_order_call(), request=None)

    assert result.tools_called
    assert result.content == "I will create it.\n"
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "create_order"
    assert json.loads(tool_call.function.arguments) == {
        "user_id": 42,
        "urgent": True,
        "note": "Please leave at front desk.",
        "shipping": {"city": "Singapore", "zip": 18956},
        "items": [{"sku": "book-001", "qty": 2}, {"sku": "pen-007", "qty": 5}],
        "metadata": {"source": "mobile", "campaign": "may-launch"},
    }


def non_streaming_without_tool_call_keeps_content():
    parser = tool_parser()
    result = parser.extract_tool_calls("plain response", request=None)

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == "plain response"


def non_streaming_multiple_tool_calls():
    parser = tool_parser()
    result = parser.extract_tool_calls(build_multiple_order_call(), request=None)

    assert result.tools_called
    assert result.content is None
    assert [tool_call.function.name for tool_call in result.tool_calls] == ["create_order", "create_order"]
    assert [json.loads(tool_call.function.arguments)["user_id"] for tool_call in result.tool_calls] == [1, 2]


def streaming_without_tool_call_emits_text():
    parser = tool_parser()
    results = feed(parser, ["plain ", "response"])

    assert collect_content(results) == "plain response"
    assert collect_tool_calls(results) == {}


def streaming_nested_tool_call():
    parser = tool_parser()
    eos_id = model_tokenizer().eos_token_id
    tool_call_text = build_order_call()
    results = feed(
        parser,
        [
            "I will create it.\n",
            tool_call_text[:5],
            tool_call_text[5:17],
            tool_call_text[17:120],
            tool_call_text[120:],
            ("", [eos_id]),
        ],
    )

    assert collect_content(results) == "I will create it.\n"
    tool_calls = collect_tool_calls(results)
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "create_order"
    assert tool_calls[0]["id"] is not None
    assert json.loads(tool_calls[0]["arguments"]) == json.loads(parser.streamed_args_for_tool[0])
    assert json.loads(parser.prev_tool_call_arr[0]["arguments"])["items"][1]["qty"] == 5
    assert results[-1].content is None


# ---------------------------------------------------------------------------------------------------------------------
# The four scored nodes

REASONING = [
    parser_registration,
    nonstreaming_extracts_explicit_reasoning_block,
    nonstreaming_without_start_tag_is_content,
    nonstreaming_drops_leading_end_tag,
    nonstreaming_non_leading_end_tag_is_content,
    nonstreaming_enabled_mode_starts_in_reasoning,
    nonstreaming_open_reasoning_block,
    token_id_helpers,
    token_id_helpers_enabled_mode,
]
REASONING_SPLIT = [token_id_helpers_with_split_marker_tokens]
REASONING_STREAMING = [
    streaming_reasoning_tags_are_not_returned,
    streaming_boundary_can_emit_reasoning_and_content,
    streaming_drops_leading_end_tag,
    streaming_non_leading_end_tag_is_content,
    streaming_enabled_mode_starts_in_reasoning,
    streaming_plain_content_ends_reasoning_phase,
]
REASONING_STREAMING_SPLIT = [
    streaming_split_marker_tokens_are_not_returned,
    streaming_split_marker_text_drives_end_state,
    streaming_split_end_marker_content_ids_are_stripped,
    streaming_split_marker_tokens_enabled_mode,
    streaming_split_marker_text_across_deltas,
    streaming_split_leading_end_marker_text_across_deltas,
]
TOOL = [
    minimax_m3_parser_registered,
    non_streaming_nested_tool_call,
    non_streaming_without_tool_call_keeps_content,
    non_streaming_multiple_tool_calls,
]
TOOL_STREAMING = [streaming_without_tool_call_emits_text, streaming_nested_tool_call]


def conjunction(request, functions: Sequence[Callable[[], None]]) -> None:
    """Run every function; record each one's result as a property of this node (written to the JUnit report as
    `<property name="<function>" value="pass|fail: ...">`); the node passes iff all of them pass."""
    failed = []
    for function in functions:
        try:
            function()
        except Exception as error:  # an assertion or any other failure of the candidate's parser
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            request.node.user_properties.append((function.__name__, "fail: " + detail[:1000]))
            failed.append(function.__name__)
        else:
            request.node.user_properties.append((function.__name__, "pass"))
    assert not failed, f"{len(failed)} of {len(functions)} adapted functions failed: {', '.join(failed)}"


def test_reasoning_parser(request):
    conjunction(request, REASONING + (REASONING_SPLIT if admit_split_marker_functions() else []))


def test_reasoning_parser_streaming(request):
    conjunction(request, REASONING_STREAMING + (REASONING_STREAMING_SPLIT if admit_split_marker_functions() else []))


def test_tool_parser(request):
    conjunction(request, TOOL)


def test_tool_parser_streaming(request):
    conjunction(request, TOOL_STREAMING)
