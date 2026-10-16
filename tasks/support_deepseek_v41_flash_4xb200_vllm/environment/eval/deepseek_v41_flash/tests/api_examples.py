#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""API development examples against a running server: completions, streaming, log-probabilities, chat, a second
conversation turn, determinism and concurrent requests.

    python api_examples.py --base-url http://127.0.0.1:8000

Each example sends a small request, prints what came back and says whether the response is well formed: HTTP and
JSON framing, the object type, one choice, the finish reason, token usage, requested log-probabilities (one finite
number of at most zero per generated token), and for streams the server-sent-event framing (`data:` lines, one
terminal `[DONE]`, one finish reason, the role first, usage when requested) and that the streamed pieces assemble to
the non-streamed reply of the same deterministic request. It exits 1 when an example's response is malformed or
inconsistent. The protocol functions below are importable. These are development examples for the interfaces the
task names, not a pass verdict: they do not establish model quality, request isolation or task acceptance.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import urllib.error
import urllib.request

MODEL = "/model/deepseek-ai/DeepSeek-V4.1-Flash"
#: The public operation limit for one HTTP correctness request below 64,000 input tokens (README).
REQUEST_TIMEOUT_S = 900
FINISH_REASONS = ("stop", "length")


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def post(base: str, path: str, body: dict, timeout: float = REQUEST_TIMEOUT_S) -> tuple[int | None, object, str | None]:
    """POST JSON; (HTTP status or None, the parsed body or None, a transport error or None). Never raises."""
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    except (urllib.error.URLError, OSError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"[:400]
    try:
        return status, json.loads(raw), None
    except ValueError:
        return status, None, f"the body is not JSON ({raw[:160]!r})"


def stream(base: str, path: str, body: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    """POST with `stream: true` and read the server-sent events; {status, lines, chunks, done, errors}. Never raises."""
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    out: dict = {"status": None, "lines": 0, "chunks": [], "done": False, "errors": []}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out["status"] = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                out["lines"] += 1
                if out["done"]:
                    out["errors"].append("data after the terminal [DONE]")
                    break
                if not line.startswith("data: "):
                    out["errors"].append(f"a line that is not a data event: {line[:80]!r}")
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    out["done"] = True
                    continue
                try:
                    out["chunks"].append(json.loads(payload))
                except ValueError:
                    out["errors"].append(f"an event that is not JSON: {payload[:80]!r}")
    except urllib.error.HTTPError as exc:
        out["status"] = exc.code
        out["errors"].append(f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError) as exc:
        out["errors"].append(f"{type(exc).__name__}: {exc}"[:400])
    if not out["done"] and not out["errors"]:
        out["errors"].append("the stream ended without the terminal [DONE]")
    return out


def logprob_errors(values: list, count: int, what: str) -> list[str]:
    """Requested log-probabilities: exactly `count` finite numbers, each at most zero."""
    errors = []
    if len(values) != count:
        errors.append(f"{what}: {len(values)} log-probabilities for {count} generated tokens")
    bad = [i for i, v in enumerate(values) if not (is_number(v) and math.isfinite(v) and v <= 1e-6)]
    if bad:
        errors.append(f"{what}: {len(bad)} log-probabilities that are not finite numbers at most 0, first at {bad[:3]}")
    return errors


def usage_errors(usage, what: str) -> list[str]:
    if not isinstance(usage, dict) or not all(is_number(usage.get(k)) for k in ("prompt_tokens", "completion_tokens")):
        return [f"{what}: no usage counts"]
    if is_number(usage.get("total_tokens")) and usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        return [f"{what}: usage total {usage['total_tokens']} is not prompt + completion"]
    return []


def check_completion(status, body, *, logprobs: bool = False, token_ids: bool = False) -> tuple[list[str], dict]:
    """A non-streamed /v1/completions reply: framing, one choice, text, finish reason, usage; with `logprobs`, one
    finite log-probability per generated token; with `token_ids`, the generated token ids. Returns (errors, facts)."""
    facts: dict = {}
    if status != 200 or not isinstance(body, dict):
        return [f"HTTP {status}" if status != 200 else "the reply is not a JSON object"], facts
    errors = []
    if body.get("object") != "text_completion":
        errors.append(f"object is {body.get('object')!r}, not 'text_completion'")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return errors + ["not exactly one choice"], facts
    c = choices[0]
    text = c.get("text")
    if not isinstance(text, str):
        errors.append("the choice has no text")
    if c.get("finish_reason") not in FINISH_REASONS:
        errors.append(f"finish_reason {c.get('finish_reason')!r}")
    errors += usage_errors(body.get("usage"), "completion")
    n = (body.get("usage") or {}).get("completion_tokens")
    facts.update(text=text, finish_reason=c.get("finish_reason"), completion_tokens=n)
    if token_ids:
        ids = c.get("token_ids")
        if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
            errors.append("no generated token ids")
        elif is_number(n) and len(ids) != n:
            errors.append(f"{len(ids)} generated token ids for {n} completion tokens")
        facts["token_ids"] = ids
    if logprobs:
        lp = c.get("logprobs")
        values = lp.get("token_logprobs") if isinstance(lp, dict) else None
        if not isinstance(values, list):
            errors.append("no generated-token log-probabilities")
        elif is_number(n):
            errors += logprob_errors(values, n, "completion")
        facts["token_logprobs"] = values
    return errors, facts


def check_chat(status, body, *, logprobs: bool = False) -> tuple[list[str], dict]:
    """A non-streamed /v1/chat/completions reply: framing, one assistant message with text content, finish reason,
    usage; with `logprobs`, one finite log-probability per generated token. Returns (errors, facts)."""
    facts: dict = {}
    if status != 200 or not isinstance(body, dict):
        return [f"HTTP {status}" if status != 200 else "the reply is not a JSON object"], facts
    errors = []
    if body.get("object") != "chat.completion":
        errors.append(f"object is {body.get('object')!r}, not 'chat.completion'")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return errors + ["not exactly one choice"], facts
    c = choices[0]
    msg = c.get("message") if isinstance(c.get("message"), dict) else {}
    if msg.get("role") != "assistant":
        errors.append(f"message role {msg.get('role')!r}")
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        errors.append("the message has no text content")
    if msg.get("tool_calls"):
        errors.append("a tool call in a tool-free conversation")
    if c.get("finish_reason") not in FINISH_REASONS:
        errors.append(f"finish_reason {c.get('finish_reason')!r}")
    errors += usage_errors(body.get("usage"), "chat")
    n = (body.get("usage") or {}).get("completion_tokens")
    facts.update(content=content, finish_reason=c.get("finish_reason"), usage=body.get("usage"))
    if logprobs:
        lp = c.get("logprobs")
        items = lp.get("content") if isinstance(lp, dict) else None
        if not isinstance(items, list) or not all(isinstance(i, dict) and isinstance(i.get("token"), str) for i in items):
            errors.append("no generated-token log-probabilities")
        elif is_number(n):
            errors += logprob_errors([i.get("logprob") for i in items], n, "chat")
            facts["tokens"] = [i["token"] for i in items]
            facts["token_logprobs"] = [i.get("logprob") for i in items]
    return errors, facts


def check_chat_stream(s: dict, *, logprobs: bool = False, usage: bool = True) -> tuple[list[str], dict]:
    """A streamed chat reply read by `stream`: SSE framing, every event a chat.completion.chunk with one choice, the
    assistant role first, exactly one finish reason, content pieces, per-token log-probabilities when requested and the
    final usage chunk when requested, consistent with the pieces streamed. Returns (errors, facts with the assembly)."""
    errors = list(s["errors"])
    if s["status"] != 200:
        errors.append(f"HTTP {s['status']}")
    pieces, finishes, roles, tokens, lps, usage_seen = [], [], [], [], [], None
    for k, chunk in enumerate(s["chunks"]):
        if not isinstance(chunk, dict) or chunk.get("object") != "chat.completion.chunk":
            errors.append(f"event {k} is not a chat.completion.chunk")
            continue
        if chunk.get("usage") is not None and not chunk.get("choices"):
            usage_seen = chunk["usage"]
            continue
        choices = chunk.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            errors.append(f"event {k} has not exactly one choice")
            continue
        c = choices[0]
        delta = c.get("delta") if isinstance(c.get("delta"), dict) else {}
        if delta.get("role") is not None:
            roles.append((k, delta["role"]))
        if isinstance(delta.get("content"), str):
            pieces.append(delta["content"])
        if delta.get("tool_calls"):
            errors.append(f"event {k} carries a tool call in a tool-free conversation")
        if c.get("finish_reason") is not None:
            finishes.append((k, c["finish_reason"]))
        if logprobs and isinstance(c.get("logprobs"), dict) and isinstance(c["logprobs"].get("content"), list):
            for item in c["logprobs"]["content"]:
                tokens.append(item.get("token") if isinstance(item, dict) else None)
                lps.append(item.get("logprob") if isinstance(item, dict) else None)
    if not roles or roles[0][1] != "assistant" or roles[0][0] != 0:
        errors.append("the stream does not open with the assistant role")
    if len(finishes) != 1 or finishes[0][1] not in FINISH_REASONS:
        errors.append(f"finish reasons in the stream: {finishes}")
    if usage:
        errors += usage_errors(usage_seen, "stream")
    n = (usage_seen or {}).get("completion_tokens")
    if logprobs and is_number(n):
        errors += logprob_errors(lps, n, "stream")
    assembled = "".join(pieces)
    if not assembled:
        errors.append("the stream carries no content")
    return errors, {"content": assembled, "events": len(s["chunks"]), "finish": finishes, "usage": usage_seen,
                    "tokens": tokens, "token_logprobs": lps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)
    base, model = a.base_url, a.model
    problems: list[str] = []

    def check(name: str, errors: list[str], detail: str) -> None:
        ok = not errors
        print(f"[{'ok' if ok else 'PROBLEM'}] {name}: {detail}" + ("" if ok else f" -- {'; '.join(errors)}"), flush=True)
        if not ok:
            problems.append(name)

    greedy = {"model": model, "temperature": 0.0, "max_tokens": 32}
    prompt = "The three primary colors of light are"

    errors, first = check_completion(*post(base, "/v1/completions", dict(greedy, prompt=prompt))[:2])
    check("completion", errors, repr((first.get("text") or "")[:120]))

    errors, again = check_completion(*post(base, "/v1/completions", dict(greedy, prompt=prompt))[:2])
    same = again.get("text") == first.get("text")
    check("greedy determinism (the same request twice)", errors + ([] if same else ["the two replies differ"]),
          "identical" if same else repr((again.get("text") or "")[:80]))

    s = stream(base, "/v1/completions", dict(greedy, prompt=prompt))
    streamed = "".join(c["choices"][0].get("text", "") for c in s["chunks"] if isinstance(c, dict) and c.get("choices"))
    check("streaming completion", s["errors"] + ([] if streamed == first.get("text") else ["the pieces do not assemble to the non-streamed text"]),
          f"{len(s['chunks'])} events")

    errors, facts = check_completion(*post(base, "/v1/completions", dict(greedy, prompt=prompt, logprobs=0))[:2], logprobs=True)
    check("log-probabilities of generated tokens", errors, f"first {(facts.get('token_logprobs') or [])[:3]}")

    status, body, _ = post(base, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
                                                     "prompt_logprobs": 0})
    plp = ((body or {}).get("choices") or [{}])[0].get("prompt_logprobs") if isinstance(body, dict) else None
    check("prompt log-probabilities (fidelity/check.py parses them)",
          [] if status == 200 and isinstance(plp, list) and len(plp) > 1 and plp[0] is None else ["no prompt_logprobs list"],
          f"{len(plp) if isinstance(plp, list) else plp} entries")

    chat = {"model": model, "temperature": 0.0, "max_tokens": 32, "logprobs": True,
            "messages": [{"role": "user", "content": "Name the three primary colors of light."}]}
    errors, reply = check_chat(*post(base, "/v1/chat/completions", chat)[:2], logprobs=True)
    check("chat completion", errors, repr((reply.get("content") or "")[:120]))

    errors, streamed = check_chat_stream(stream(base, "/v1/chat/completions", dict(chat, stream_options={"include_usage": True})),
                                         logprobs=True)
    if streamed.get("content") != reply.get("content"):
        errors.append("the streamed pieces do not assemble to the non-streamed reply")
    check("streaming chat completion", errors, f"{streamed.get('events')} events")

    turn2 = dict(chat, messages=chat["messages"] + [{"role": "assistant", "content": reply.get("content") or ""},
                                                     {"role": "user", "content": "Which of them has the longest wavelength?"}])
    errors, second = check_chat(*post(base, "/v1/chat/completions", turn2)[:2], logprobs=True)
    check("a second conversation turn", errors, repr((second.get("content") or "")[:120]))

    prompts = [f"Write the number {i} in words:" for i in range(8)]
    alone = [check_completion(*post(base, "/v1/completions", dict(greedy, prompt=p))[:2])[1].get("text") for p in prompts]
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as ex:
        together = list(ex.map(lambda p: check_completion(*post(base, "/v1/completions", dict(greedy, prompt=p))[:2]), prompts))
    errors = [e for errs, _ in together for e in errs]
    agree = sum(1 for (_, f), t in zip(together, alone) if f.get("text") == t)
    check("concurrent completions", errors, f"{len(together)} replies; {agree} identical to their solo replies")
    print("The concurrent example reports well-formed replies and solo/concurrent agreement; "
          "it does not establish complete request isolation or semantic correctness.")

    print(f"{len(problems)} problem(s)" + (": " + ", ".join(problems) if problems else ""))
    print("These examples do not establish model quality, full API coverage or task acceptance.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
