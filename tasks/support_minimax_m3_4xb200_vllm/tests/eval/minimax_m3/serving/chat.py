#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Chat completions through the ``minimax_m3`` reasoning and tool-call parsers, checked against each response's own
generated token IDs.

Every request sets ``return_token_ids``. The response's generated IDs are decoded with ``/model``'s tokenizer
(special tokens kept, a final end-of-sequence token dropped) and the raw text is parsed independently of the
engine's parsers, so no check depends on two generations agreeing:

* reasoning: with ``thinking_mode`` ``enabled`` the prompt ends in ``<mm:think>``, so the reasoning is the text before
  ``</mm:think>`` and the content the text after it; ``disabled`` ends the prompt in ``</mm:think>`` and everything is
  content; ``adaptive`` splits iff the output contains the markers. After ``strip()`` the response's ``reasoning``
  and ``content`` (streaming: concatenated deltas) must equal those parts, neither may contain a marker, and
  ``enabled`` must yield non-empty reasoning.
* tool calls: the raw text's ``]<]minimax[>[<tool_call>`` block is parsed by the chat template's format
  (``<invoke name="...">``, one element per argument, ``<item>`` for array elements, scalars typed by the tool's JSON
  schema) into (name, arguments); the response's ``tool_calls`` (streaming: assembled deltas) must equal them
  exactly, the arguments must validate against the schema, and ``content`` must carry no namespace markup.

  python /app/eval/minimax_m3/run.py chat --base-url http://127.0.0.1:8000 --output /tmp/c.json
  python /app/eval/minimax_m3/run.py tools --base-url http://127.0.0.1:8000 --output /tmp/t.json
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from .fidelity import MODEL
from .http import ObservationError, request, stream

THINK_BEGIN, THINK_END = "<mm:think>", "</mm:think>"
NS = "]<]minimax[>["
TOOL_BEGIN, TOOL_END = NS + "<tool_call>", NS + "</tool_call>"
EOS = "[e~["
TOKENIZER = "/model/tokenizer.json"


@lru_cache(maxsize=1)
def tokenizer(path: str = TOKENIZER):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def raw_text(token_ids: list[int], path: str = TOKENIZER) -> str:
    text = tokenizer(path).decode(token_ids, skip_special_tokens=False)
    return text[: -len(EOS)] if text.endswith(EOS) else text


def expected_split(raw: str, mode: str) -> tuple[str, str]:
    if mode == "disabled":
        return "", raw
    if mode == "enabled":
        if THINK_END in raw:
            reasoning, content = raw.split(THINK_END, 1)
            return reasoning, content
        return raw, ""
    if mode == "adaptive":
        if THINK_END in raw:
            before, content = raw.split(THINK_END, 1)
            return before.split(THINK_BEGIN)[-1], content
        return "", raw
    raise ValueError(mode)


# ------------------------------------------------------------------ tool-call text

_TAG = re.compile(re.escape(NS) + r"<(/?)([^<>]+)>")


def _parse_elements(text: str):
    """Parse ``NS<k>...NS</k>`` elements into [(key, value)] where value is a string or a nested list."""
    stack: list[tuple[str, list, int]] = [("", [], 0)]
    pos = 0
    for m in _TAG.finditer(text):
        closing, tag = m.group(1) == "/", m.group(2)
        if not closing:
            stack.append((tag, [], m.end()))
        else:
            if len(stack) < 2 or stack[-1][0] != tag:
                raise ObservationError(f"unbalanced tool-call element </{tag}>")
            name, children, start = stack.pop()
            value = children if children else text[start:m.start()]
            stack[-1][1].append((name, value))
        pos = m.end()
    if len(stack) != 1:
        raise ObservationError(f"unclosed tool-call element <{stack[-1][0]}>")
    return stack[0][1]


def _typed(value, schema: dict):
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "array":
        if isinstance(value, str):
            return [] if value == "" else [value]
        return [_typed(v, (schema or {}).get("items", {})) for k, v in value if k == "item"]
    if t == "object":
        props = (schema or {}).get("properties", {})
        if isinstance(value, str):
            return {} if value == "" else value
        return {k: _typed(v, props.get(k, {})) for k, v in value}
    if not isinstance(value, str):  # nested structure where a scalar was declared: keep it as parsed
        return {k: _typed(v, {}) for k, v in value}
    if t == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if t == "number":
        try:
            f = float(value)
            return int(f) if f.is_integer() and "." not in value and "e" not in value.lower() else f
        except ValueError:
            return value
    if t == "boolean":
        return {"true": True, "false": False}.get(value, value)
    return value


def parse_tool_calls(raw: str, tools: list[dict]) -> list[dict]:
    """[(name, arguments)] of the raw text's tool-call block (empty when there is none)."""
    if TOOL_BEGIN not in raw:
        return []
    body = raw.split(TOOL_BEGIN, 1)[1]
    if TOOL_END not in body:
        raise ObservationError("unterminated tool-call block")
    body = body.split(TOOL_END, 1)[0]
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    calls = []
    for m in re.finditer(re.escape(NS) + r'<invoke name="([^"]*)">(.*?)' + re.escape(NS) + r"</invoke>", body,
                         flags=re.S):
        name, inner = m.group(1), m.group(2)
        props = schemas.get(name, {}).get("properties", {})
        args = {k: _typed(v, props.get(k, {})) for k, v in _parse_elements(inner)}
        calls.append({"name": name, "arguments": args})
    return calls


# ------------------------------------------------------------------ requests

def chat(base: str, messages: list[dict], *, mode: str | None, streaming: bool, tools=None, max_tokens=2048,
         model: str = MODEL, timeout: float = 900) -> dict:
    """One greedy chat request; returns reasoning, content, tool calls and the generated token IDs."""
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens,
            "return_token_ids": True}
    if mode is not None:
        body["chat_template_kwargs"] = {"thinking_mode": mode}
    if tools:
        body.update({"tools": tools, "tool_choice": "auto"})
    if not streaming:
        resp = request(base, "/v1/chat/completions", body, timeout=timeout)
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if not isinstance(choices, list) or len(choices) != 1:
            raise ObservationError("expected exactly one chat choice")
        msg = choices[0].get("message") or {}
        ids = choices[0].get("token_ids")
        calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments")}
                 for c in (msg.get("tool_calls") or [])]
        return {"reasoning": msg.get("reasoning") or "", "content": msg.get("content") or "",
                "tool_calls": calls, "token_ids": ids, "finish_reason": choices[0].get("finish_reason")}
    events = stream(base, "/v1/chat/completions", body, timeout=timeout)
    reasoning, content, ids, finish = [], [], [], None
    calls: dict[int, dict] = {}
    for ev in events:
        for ch in ev.get("choices") or []:
            delta = ch.get("delta") or {}
            reasoning.append(delta.get("reasoning") or "")
            content.append(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if isinstance(ch.get("token_ids"), list):
                ids += ch["token_ids"]
            finish = ch.get("finish_reason") or finish
    return {"reasoning": "".join(reasoning), "content": "".join(content),
            "tool_calls": [calls[i] for i in sorted(calls)], "token_ids": ids or None, "finish_reason": finish}


def _ids(resp: dict) -> list[int]:
    ids = resp.get("token_ids")
    if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ObservationError("the response carries no generated token IDs")
    return ids


def check_reasoning(resp: dict, mode: str) -> dict:
    raw = raw_text(_ids(resp))
    want_r, want_c = expected_split(raw, mode)
    got_r, got_c = resp["reasoning"], resp["content"]
    problems = []
    if got_r.strip() != want_r.strip():
        problems.append("reasoning differs from the text before the end-of-thinking marker")
    if got_c.strip() != want_c.strip():
        problems.append("content differs from the text after the end-of-thinking marker")
    if any(m in got_r or m in got_c for m in (THINK_BEGIN, THINK_END)):
        problems.append("a thinking marker leaked into reasoning or content")
    if mode == "enabled" and not got_r.strip():
        problems.append("thinking enabled but no reasoning returned")
    return {"mode": mode, "raw_tokens": len(resp["token_ids"]), "problems": problems, "pass": not problems,
            "expected": {"reasoning": want_r, "content": want_c}, "served": {"reasoning": got_r, "content": got_c}}


def check_tools(resp: dict, tools: list[dict], expected_calls: int | None = None) -> dict:
    import jsonschema

    raw = raw_text(_ids(resp))
    want = parse_tool_calls(raw.split(THINK_END, 1)[-1] if THINK_END in raw else raw, tools)
    problems = []
    got = []
    for c in resp["tool_calls"]:
        try:
            got.append({"name": c["name"], "arguments": json.loads(c["arguments"] or "{}")})
        except (TypeError, json.JSONDecodeError):
            problems.append(f"tool call {c.get('name')!r}: arguments are not JSON")
    if not problems and got != want:
        problems.append("tool calls differ from the calls parsed from the generated text")
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    for c in got:
        try:
            jsonschema.validate(c["arguments"], schemas.get(c["name"], {}))
        except jsonschema.ValidationError as exc:
            problems.append(f"{c['name']}: arguments do not validate: {exc.message}")
    if NS in resp["content"]:
        problems.append("namespace markup in content")
    if expected_calls is not None and len(want) != expected_calls:
        problems.append(f"{len(want)} calls generated, the case expects {expected_calls}")
    return {"problems": problems, "pass": not problems, "parsed": want, "served": got}
