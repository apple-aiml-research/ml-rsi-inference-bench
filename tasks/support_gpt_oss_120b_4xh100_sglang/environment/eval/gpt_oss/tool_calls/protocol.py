#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""OpenAI chat protocol checks used by the public probe and private verifier."""
from __future__ import annotations

import codecs
import json
import math
from collections.abc import Iterable
from typing import Any


class ProtocolError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ProtocolError(f"non-finite JSON value: {value}")


def parse_json(text: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), "non-finite JSON number")
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def equal_json(actual: Any, expected: Any) -> bool:
    """Recursive JSON equality, including booleans distinct from numbers."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return all(not isinstance(x, float) or math.isfinite(x) for x in (actual, expected)) and actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equal_json(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(equal_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def events(chunks: Iterable[bytes]) -> Iterable[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer = ""
    data = []
    for chunk in chunks:
        require(isinstance(chunk, bytes), "SSE transport chunks must be bytes")
        try:
            buffer += decoder.decode(chunk)
        except UnicodeError as exc:
            raise ProtocolError("invalid UTF-8 in SSE") from exc
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
                continue
            if line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value if sep else "")
    try:
        buffer += decoder.decode(b"", final=True)
    except UnicodeError as exc:
        raise ProtocolError("truncated UTF-8 in SSE") from exc
    require(not buffer.strip() and not data, "unterminated SSE event")


def _identity(response: dict) -> tuple:
    require(isinstance(response, dict), "response must be an object")
    require(isinstance(response.get("id"), str) and bool(response["id"]), "missing response ID")
    require(isinstance(response.get("model"), str) and bool(response["model"]), "missing model identity")
    require(type(response.get("created")) is int, "missing creation timestamp")
    return response["id"], response["model"], response["created"]


def assemble(chunks: Iterable[bytes]) -> dict:
    identity = None
    done = False
    finished = False
    finish_reason = None
    role = None
    content = []
    reasoning = []
    calls: dict[int, dict] = {}
    usage = None
    usage_only_seen = False
    count = 0
    for data in events(chunks):
        require(not done, "data after [DONE]")
        if data == "[DONE]":
            require(finished, "[DONE] before finish_reason")
            done = True
            continue
        event = parse_json(data)
        current_identity = _identity(event)
        require(event.get("object") == "chat.completion.chunk", "unexpected SSE object")
        if identity is None:
            identity = current_identity
        require(identity == current_identity, "response identity changed within stream")
        count += 1
        choices = event.get("choices")
        require(isinstance(choices, list), "missing choices")
        if not choices:
            require(finished and not usage_only_seen and event.get("usage") is not None,
                    "only one terminal usage event is allowed after finish")
            usage_only_seen = True
            usage = event["usage"]
            continue
        require(not finished, "choice data after terminal choice")
        require(len(choices) == 1 and type(choices[0].get("index")) is int and choices[0]["index"] == 0,
                "expected one choice at index 0")
        choice = choices[0]
        delta = choice.get("delta")
        require(isinstance(delta, dict), "missing delta")
        if delta.get("role") is not None:
            require(delta["role"] == "assistant" and role in (None, "assistant"), "invalid assistant role")
            role = delta["role"]
        for key, destination in (("content", content), ("reasoning_content", reasoning)):
            piece = delta.get(key)
            if piece is not None:
                require(isinstance(piece, str), f"{key} delta is not text")
                destination.append(piece)
        for part in delta.get("tool_calls") or []:
            require(type(part.get("index")) is int and part["index"] == 0, "expected one tool call at index 0")
            call = calls.setdefault(0, {"id": None, "type": None, "function": {"name": "", "arguments": ""}})
            for key in ("id", "type"):
                value = part.get(key)
                if value is not None:
                    require(isinstance(value, str) and bool(value), f"empty tool {key}")
                    require(call[key] in (None, value), f"tool {key} changed within stream")
                    call[key] = value
            fn = part.get("function") or {}
            require(isinstance(fn, dict), "tool function delta is not an object")
            for key in ("name", "arguments"):
                piece = fn.get(key)
                if piece is not None:
                    require(isinstance(piece, str), f"tool {key} delta is not text")
                    call["function"][key] += piece
        if event.get("usage") is not None:
            usage = event["usage"]
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
            require(finish_reason in ("stop", "tool_calls"), f"incomplete/invalid finish: {finish_reason!r}")
            finished = True
    require(done and finished and count > 0, "missing terminal choice or [DONE]")
    require(role == "assistant", "missing assistant role")
    result = {"id": identity[0], "model": identity[1], "created": identity[2], "object": "chat.completion",
              "choices": [{"index": 0, "message": {"role": role, "content": "".join(content),
                           "reasoning_content": "".join(reasoning), "tool_calls": list(calls.values())},
                           "finish_reason": finish_reason}]}
    if usage is not None:
        result["usage"] = usage
    validate(result)
    return result


def validate(response: dict, *, tool: str | None = None, arguments: dict | None = None,
             content: str | None = None) -> dict:
    _identity(response)
    require(response.get("object") == "chat.completion", "unexpected response object")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "expected exactly one choice")
    choice = choices[0]
    require(type(choice.get("index")) is int and choice["index"] == 0, "expected choice index 0")
    message = choice.get("message")
    require(isinstance(message, dict) and message.get("role") == "assistant", "missing assistant message")
    calls = message.get("tool_calls") or []
    require(isinstance(calls, list) and len(calls) <= 1, "at most one call per turn is in scope")
    if calls:
        require(choice.get("finish_reason") == "tool_calls", "call did not finish as tool_calls")
        call = calls[0]
        require(isinstance(call.get("id"), str) and bool(call["id"]), "missing tool call ID")
        require(call.get("type") == "function", "invalid tool call type")
        fn = call.get("function")
        require(isinstance(fn, dict) and isinstance(fn.get("name"), str) and bool(fn["name"]), "missing function name")
        require(isinstance(fn.get("arguments"), str), "arguments must be JSON text")
        value = parse_json(fn["arguments"])
        require(isinstance(value, dict), "arguments must encode an object")
        if tool is not None:
            require(fn["name"] == tool, f"wrong tool: {fn['name']!r}")
        if arguments is not None:
            require(equal_json(value, arguments), "argument values or JSON types differ")
    else:
        require(tool is None and arguments is None, "missing expected tool call")
        require(choice.get("finish_reason") == "stop", "content did not finish normally")
        require(isinstance(message.get("content"), str), "missing no-call content")
    if content is not None:
        require((message.get("content") or "") == content, "content differs")
    return message


def history_call(message: dict) -> dict:
    """Preserve the server's call ID and JSON text in the next client turn."""
    return {k: message[k] for k in ("role", "content", "tool_calls") if k in message}
