#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Exercise two sequential tool cycles and no-call answers on a running server.

python /app/eval/gpt_oss/tool_calls/probe.py --base-url http://127.0.0.1:8000 --output /tmp/chat
Raw bodies and streams are retained. The same client accepts private examples.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import secrets
import sys
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import ProtocolError, assemble, history_call, parse_json, require, validate

_receipt_lock = threading.Lock()
_receipts = set()


def fresh_receipt():
    with _receipt_lock:
        require(len(_receipts) < 900, "fresh receipt space exhausted")
        while True:
            value = str(100 + secrets.randbelow(900))
            if value not in _receipts:
                _receipts.add(value)
                return value


def contains_receipt(content: str, receipt: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(receipt) + r"(?!\w)", content) is not None


def post(base_url: str, body: dict, record: Path, timeout: float = 900) -> dict:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.with_suffix(".request.json").write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        with gzip.open(record.with_suffix(".response.gz"), "wb") as out:
            out.write(exc.read())
        record.with_suffix(".http.json").write_text(json.dumps({"status": exc.code,
                                      "headers": dict(exc.headers)}, indent=2) + "\n")
        raise
    with response:
        record.with_suffix(".http.json").write_text(json.dumps({"status": response.status,
                                      "headers": dict(response.headers)}, indent=2) + "\n")
        require(response.status == 200, f"HTTP {response.status}")
        mime = response.headers.get_content_type()
        require(mime == ("text/event-stream" if body.get("stream") else "application/json"),
                f"unexpected Content-Type {mime}")
        with gzip.open(record.with_suffix(".response.gz"), "wb") as out:
            def chunks():
                while data := response.read(4096):
                    out.write(data)
                    yield data
            result = assemble(chunks()) if body.get("stream") else parse_json(b"".join(chunks()).decode("utf-8"))
    validate(result)
    record.with_suffix(".assembled.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def tools_for(example: dict) -> list[dict]:
    return [{"type": "function", "function": {
        "name": "lookup_record", "description": "Look up a named record. The receipt field of the tool result is its current code.",
        "parameters": example["schema"],
    }}]


def conversation(base_url: str, model: str, example: dict, stream: bool, output: Path) -> dict:
    tools = tools_for(example)
    messages = [{"role": "system", "content": (
        "Follow each user instruction exactly. When asked to look up a record, call lookup_record once. "
        "After receiving a tool receipt, report the value of its receipt field. "
        "Use no tool when asked to repeat a word.") }]
    body = {"model": model, "tools": tools, "tool_choice": "auto", "n": 1,
            "reasoning_effort": "low", "temperature": 0, "max_tokens": 1024, "stream": stream}
    call_ids = []
    response_ids = []
    for cycle, payload in enumerate(example["payloads"]):
        messages.append({"role": "user", "content": (
            "Look up this record using lookup_record, then report the code from its receipt: " + json.dumps(payload, ensure_ascii=False))})
        response = post(base_url, {**body, "messages": messages}, output / f"cycle-{cycle}-call")
        response_ids.append(response["id"])
        message = validate(response, tool="lookup_record", arguments=payload)
        call = message["tool_calls"][0]
        call_ids.append(call["id"])
        messages.append(history_call(message))
        receipt = fresh_receipt()
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": json.dumps({"receipt": receipt})})
        messages.append({"role": "user", "content": "Report the receipt value from the tool result above. Do not call the tool again."})
        response = post(base_url, {**body, "messages": messages}, output / f"cycle-{cycle}-result")
        response_ids.append(response["id"])
        message = validate(response)
        require(not message.get("tool_calls"), "unexpected tool call after result")
        content = message["content"]
        require(not any(marker in content for marker in
                        ("<|start|>", "<|end|>", "<|channel|>", "<|message|>", "<|return|>")),
                "structural channel text leaked into receipt response")
        require(contains_receipt(content, receipt),
                "continuation did not report the fresh tool result")
        messages.append(history_call(message))
    messages.append({"role": "user", "content": "Repeat only the word cobalt. No tools are needed."})
    response = post(base_url, {**body, "messages": messages}, output / "no-call")
    response_ids.append(response["id"])
    message = validate(response)
    require(not message.get("tool_calls"), "tools=auto made an unnecessary call")
    require("cobalt" in message["content"].casefold(), "no-call response did not answer the request")
    require(not any(marker in message["content"] for marker in
                    ("<|start|>", "<|end|>", "<|channel|>", "<|message|>", "<|return|>")),
            "structural channel text leaked into no-call response")
    require(len(set(response_ids)) == len(response_ids), "response ID reused across requests")
    return {"stream": stream, "call_ids": call_ids, "response_ids": response_ids, "ok": True}


def probe(base_url: str, model: str, examples: list[dict], output: Path, workers: int = 4) -> list[dict]:
    jobs = [(i, example, stream) for i, example in enumerate(examples) for stream in (False, True)]
    def run(job):
        i, example, stream = job
        directory = output / f"example-{i}-{'stream' if stream else 'plain'}"
        try:
            return {"example": i, **conversation(base_url, model, example, stream, directory)}
        except Exception as exc:
            return {"example": i, "stream": stream, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--examples", type=Path, default=Path(__file__).with_name("examples.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rows = probe(args.base_url, args.model, parse_json(args.examples.read_text()), args.output, args.workers)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    sys.exit(0 if rows and all(row["ok"] for row in rows) else 1)
