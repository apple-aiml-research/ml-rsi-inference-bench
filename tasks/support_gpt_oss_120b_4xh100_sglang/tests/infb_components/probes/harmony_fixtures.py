#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private deterministic fixtures through the submitted production chat handler.

`python -m infb_components.probes.harmony_fixtures --recipes <recipes.json> --fixtures <file> --output <file>`.
Only the tokenizer manager, the model generation boundary, is replaced, by a stand-in
that carries the server arguments. Those arguments are built as the production launch
builds them before any subprocess starts: the declared recipe through the submitted
tree's `prepare_server_args`, then its launch-time validation `check_server_args()`,
which the launch path (`sglang.srt.entrypoints.engine._launch_subprocesses`) runs
first. A default the submission completes there is present here as in production; one
it completes later, inside its own tokenizer manager, is not. Request validation,
checkpoint templating, reasoning/function parsing and plain/SSE response construction
run in the submitted SGLang tree. Chunk boundaries are valid tokenizer prefixes;
arbitrary transport fragmentation is checked separately by the protocol client, which
is the trusted public `gpt_oss.tool_calls.protocol`.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

from gpt_oss.tool_calls.protocol import assemble, parse_json, require, validate


def prefixes(tokenizer, text: str, stride: int) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    result = []
    for stop in range(stride, len(ids), stride):
        piece = tokenizer.decode(ids[:stop], skip_special_tokens=False)
        if piece and text.startswith(piece) and "\ufffd" not in piece:
            if not result or len(piece) > len(result[-1]):
                result.append(piece)
    require(tokenizer.decode(ids, skip_special_tokens=False) == text, "fixture is not a tokenizer round trip")
    result.append(text)
    return result


class GenerationBoundary:
    def __init__(self, tokenizer, server_args, config, text, stride, delay):
        self.tokenizer = tokenizer
        self.server_args = server_args
        self.model_config = config
        self.text = text
        self.stride = stride
        self.delay = delay
        self.seen = None

    def create_abort_task(self, request):
        return None

    async def generate_request(self, request, raw_request):
        self.seen = request
        parts = prefixes(self.tokenizer, self.text, self.stride) if request.stream else [self.text]
        for index, part in enumerate(parts):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield {"text": part, "index": 0, "meta_info": {
                "id": "fixture-response", "prompt_tokens": len(request.input_ids),
                "completion_tokens": len(self.tokenizer.encode(part, add_special_tokens=False)),
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": 200002} if index == len(parts) - 1 else None,
            }}


def canonical_messages(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for message in result:
        if message.get("content") is None:
            message["content"] = ""
        if not message.get("tool_calls"):
            message.pop("tool_calls", None)
        for call in message.get("tool_calls") or []:
            args = call["function"]["arguments"]
            if isinstance(args, str):
                call["function"]["arguments"] = parse_json(args)
    return result


async def run_one(tokenizer, server_args, config, body: dict, text: str, stride: int, delay: float,
                  observation: dict | None = None):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.managers.template_manager import TemplateManager
    manager = GenerationBoundary(tokenizer, server_args, config, text, stride, delay)
    templates = TemplateManager()
    templates.initialize_templates(manager, server_args.model_path)
    serving = OpenAIServingChat(manager, templates)
    if observation is None:
        observation = {}
    source = Path(inspect.getfile(type(serving))).resolve()
    observation['handler'] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    observation['request'] = body
    request = ChatCompletionRequest(**copy.deepcopy(body))
    response = await serving.handle_request(request, None)
    if body.get("stream"):
        raw = b"".join([part.encode() if isinstance(part, str) else part async for part in response.body_iterator])
        observation['raw_sse_hex'] = raw.hex()
        # Every UTF-8 and CRLF boundary is fair game at the transport layer.
        result = assemble(raw[i:i + 1] for i in range(len(raw)))
    else:
        require(hasattr(response, "model_dump"), f"invalid plain response: {response}")
        result = response.model_dump()
    observation['response'] = result
    require(manager.seen is not None, f"request did not reach generation: {response}")
    expected_ids = tokenizer.apply_chat_template(
        canonical_messages(body["messages"]), tokenize=True, add_generation_prompt=True,
        tools=copy.deepcopy(body["tools"]), reasoning_effort="low",
    )
    observation['input_ids'] = manager.seen.input_ids
    observation['expected_input_ids'] = expected_ids
    require(manager.seen.input_ids == expected_ids, "history differs from checkpoint template; check JSON arguments and tool results")
    validate(result)
    return result


def launch_server_args(recipes: Path, serving_class: str = "S0"):
    """The declared recipe's server arguments, completed as the production launch completes them."""
    from sglang.srt.server_args import prepare_server_args
    recipe = json.loads(recipes.read_text())
    server_args = prepare_server_args(["--model-path", recipe["model"], *recipe[serving_class]])
    server_args.check_server_args()
    return server_args


async def run(model: str, rows: list[dict], delay: float = 0, recipes: Path | None = None) -> list[dict]:
    from transformers import AutoTokenizer
    from sglang.srt.configs.model_config import ModelConfig
    require(recipes is not None, "the serving recipe is required: --recipes <recipes.json>")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    server_args = launch_server_args(recipes)
    config = ModelConfig.from_server_args(server_args)
    output = []
    for index, row in enumerate(rows):
        for stream in (False, True):
            for stride in ([1, 7, 100000] if stream else [100000]):
                body = {"model": model, "messages": row["messages"], "tools": row["tools"],
                        "tool_choice": "auto", "reasoning_effort": "low", "temperature": 0,
                        "n": 1, "max_tokens": 1024, "stream": stream,
                        "stream_options": {"include_usage": True} if stream else None}
                observation = {}
                try:
                    response = await run_one(tokenizer, server_args, config, body, row["generated"], stride, delay,
                                             observation)
                    validate(response, **row["expected"])
                    output.append({"fixture": index, "stream": stream, "stride": stride, "ok": True,
                                   "observation": observation})
                except Exception as exc:
                    output.append({"fixture": index, "stream": stream, "stride": stride, "ok": False,
                                   "error": f"{type(exc).__name__}: {exc}", "observation": observation})
    return output


def examples() -> list[dict]:
    tools = [{"type": "function", "function": {"name": "record_payload", "description": "Record an object.",
              "parameters": {"type": "object", "properties": {"payload": {"type": "object"}}, "required": ["payload"]}}}]
    arguments = {"payload": {"enabled": True, "n": 1, "string": "1", "nested": {"nil": None, "empty": {}},
                             "values": [False, 0, "", "雪", "quote\"\\\n"],
                             "literal": "<|channel|>", "terminals": "<|end|><|call|><|return|>"}}
    call = {"id": "arbitrary-client-echo", "type": "function",
            "function": {"name": "record_payload", "arguments": json.dumps(arguments, ensure_ascii=False)}}
    history = [{"role": "user", "content": "Record this object."}]
    final_history = history + [{"role": "assistant", "content": None, "tool_calls": [call]},
                               {"role": "tool", "tool_call_id": call["id"], "content": '{"receipt":"fresh_é_42"}'}]
    return [
        {"tools": tools, "messages": history, "generated": (
            "<|channel|>analysis<|message|>Use the tool.<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.record_payload<|constrain|>json<|message|>"
            + json.dumps(arguments, ensure_ascii=False) + "<|call|>"),
         "expected": {"tool": "record_payload", "arguments": arguments, "content": ""}},
        {"tools": tools, "messages": final_history,
         "generated": "<|channel|>analysis<|message|>Read the result.<|end|><|start|>assistant<|channel|>final<|message|>fresh_é_42<|return|>",
         "expected": {"content": "fresh_é_42"}},
        {"tools": tools, "messages": [{"role": "user", "content": "Repeat literally: <|channel|>"}],
         "generated": "<|channel|>final<|message|>Literal <|channel|> stays.<|return|>",
         "expected": {"content": "Literal <|channel|> stays."}},
        {"tools": tools, "messages": final_history + [
            {"role": "assistant", "content": "fresh_é_42", "tool_calls": []},
            {"role": "user", "content": "Repeat cobalt without a tool."}],
         "generated": "<|channel|>final<|message|>cobalt<|return|>",
         "expected": {"content": "cobalt"}},
        {"tools": tools, "messages": final_history,
         "generated": "<|channel|>final<|message|>First. <|end|>"
                      "<|start|>assistant<|channel|>analysis<|message|>Check the result.<|end|>"
                      "<|start|>assistant<|channel|>final<|message|>Second.<|return|>",
         "expected": {"content": "First. Second."}},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/model")
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0)
    args = parser.parse_args()
    rows = parse_json(args.fixtures.read_text()) if args.fixtures else examples()
    result = asyncio.run(run(args.model, rows, args.delay, args.recipes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result and all(row["ok"] for row in result) else 1)
