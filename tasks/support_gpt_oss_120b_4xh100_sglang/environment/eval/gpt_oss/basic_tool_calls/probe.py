#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The grader's tool-call check, shipped as its own client.

The grader asks twelve fixed questions of its own through `/v1/chat/completions`, each of which needs one of the
three tools below, with those tools and `tool_choice: "auto"`, at low reasoning effort, greedily, and asks each twice:
once as a plain response and once streamed (`stream: true`), whose deltas it assembles by `tool_calls[].index`. It
requires every answer, in both forms, to come back as an OpenAI tool call:
- `finish_reason` is "tool_calls";
- a `tool_calls` entry names the tool the question needs;
- its `arguments` parse as a JSON object carrying that tool's required parameters;
- `content` carries no Harmony channel text.

The server is the recipe's launch, unchanged: no tool-call parser flag is passed. This file is the check the grader
runs, byte for byte; the questions beside it are public ones, not the grader's.

    python3 /app/eval/gpt_oss/basic_tool_calls/probe.py --base-url http://127.0.0.1:8000

It prints each answer's verdict and a JSON summary, and exits 1 if any answer is not a tool call. Standard
library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "The city name, e.g. Paris"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount of money from one currency to another at today's rate.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
            "to_currency": {"type": "string", "description": "ISO 4217 code, e.g. EUR"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "lookup_flight",
        "description": "Look up the status of a flight by its flight number.",
        "parameters": {"type": "object", "properties": {
            "flight_number": {"type": "string", "description": "e.g. BA117"}},
            "required": ["flight_number"]}}},
]
REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}
RESIDUE = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>", "<|constrain|>",
           "assistantanalysis", "assistantcommentary", "assistantfinal", "to=functions.")
MAX_TOKENS = 1024


def reasons(status: int, choice: dict, expected: str) -> list[str]:
    """Why an answer is not the tool call its question needs; empty when it is."""
    if status != 200:
        return [f"HTTP {status}"]
    found = []
    if choice.get("finish_reason") != "tool_calls":
        found.append(f"finish_reason is {choice.get('finish_reason')!r}, not 'tool_calls'")
    message = choice.get("message") or {}
    calls = [c.get("function") or {} for c in message.get("tool_calls") or []]
    named = [c for c in calls if c.get("name") == expected]
    if not named:
        found.append(f"no tool call names {expected} (named: {[c.get('name') for c in calls]})")
    else:
        try:
            args = json.loads(named[0].get("arguments") or "")
        except ValueError:
            args = None
        if not isinstance(args, dict):
            found.append(f"{expected}'s arguments are not a JSON object: {named[0].get('arguments')!r}"[:200])
        else:
            missing = [k for k in REQUIRED[expected] if k not in args]
            if missing:
                found.append(f"{expected}'s arguments lack {missing}")
    content = message.get("content") or ""
    found += [f"Harmony text {r!r} in content" for r in RESIDUE if r in content]
    return found


def assemble(lines) -> dict:
    """A streamed completion's server-sent events, assembled into a plain response's shape: the content, the tool
    calls (each call's name and arguments joined across its deltas, by index) and the finish reason."""
    content, calls, finish = [], {}, None
    for raw in lines:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        for choice in json.loads(data).get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                fn = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn["name"] += (tc.get("function") or {}).get("name") or ""
                fn["arguments"] += (tc.get("function") or {}).get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    message = {"content": "".join(content), "tool_calls": [{"function": calls[i]} for i in sorted(calls)]}
    return {"choices": [{"message": message, "finish_reason": finish}]}


def _post(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, (assemble(r.readlines()) if body.get("stream") else json.loads(r.read()))
    except urllib.error.HTTPError as e:
        return e.code, {}


def served_model(base_url: str, timeout: float = 300.0) -> str:
    """The first model the server lists."""
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as r:
        return json.loads(r.read())["data"][0]["id"]


def ask(base_url: str, model: str, question: str, expected: str, stream: bool = False, max_tokens: int = MAX_TOKENS,
        timeout: float = 900.0) -> dict:
    """One question with the three tools, at low effort, greedily, plain or streamed; the verdict's reasons, if it
    is not a tool call."""
    body = {"model": model, "messages": [{"role": "user", "content": question}], "tools": TOOLS,
            "tool_choice": "auto", "reasoning_effort": "low", "temperature": 0.0, "max_tokens": max_tokens,
            "stream": stream}
    status, resp = _post(f"{base_url}/v1/chat/completions", body, timeout)
    choice = (resp.get("choices") or [{}])[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    return {"question": question, "expected": expected, "stream": stream, "status": status,
            "finish_reason": choice.get("finish_reason"),
            "calls": [{"name": (c.get("function") or {}).get("name"),
                       "arguments": (c.get("function") or {}).get("arguments")} for c in calls],
            "reasons": reasons(status, choice, expected)}


def probe(base_url: str, questions: list[dict], model: str | None = None, max_tokens: int = MAX_TOKENS,
          workers: int = 12) -> list[dict]:
    """Every question plain and then streamed, each form's questions at once, twelve at a time; one row per question
    and form, the plain rows first, in question order."""
    base_url = base_url.rstrip("/")
    model = model or served_model(base_url)
    rows = []
    for stream in (False, True):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows += list(ex.map(lambda q: ask(base_url, model, q["question"], q["tool"], stream, max_tokens), questions))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    ap.add_argument("--model", default=None, help="default: the first model the server lists")
    a = ap.parse_args()
    rows = probe(a.base_url, json.loads(Path(a.questions).read_text()), a.model)
    for i, r in enumerate(rows):
        verdict = "tool call" if not r["reasons"] else "NOT A TOOL CALL: " + "; ".join(r["reasons"])
        print(f"{i:2d} {'streamed' if r['stream'] else 'plain':8} {verdict} | {r['calls']}")
    bad = sum(1 for r in rows if r["reasons"])
    print(json.dumps({"answers": len(rows), "tool_calls": len(rows) - bad, "not_tool_calls": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
