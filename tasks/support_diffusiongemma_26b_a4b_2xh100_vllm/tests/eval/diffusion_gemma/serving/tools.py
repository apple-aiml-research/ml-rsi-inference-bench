#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tool calls, public: the served tool-call parse against an independent parse of
the response's own generated text."""
from __future__ import annotations

import json
import re

from .common import raw_text
from .recipes import SERVED_NAME

Q = '<|"|>'
CALL_RE = re.compile(r"<\|tool_call>call:([A-Za-z_][\w.-]*)(\{.*?\})<tool_call\|>", re.S)


class _P:
    def __init__(self, s: str):
        self.s, self.i = s, 0

    def value(self):
        s = self.s
        if s.startswith(Q, self.i):
            j = s.index(Q, self.i + len(Q))
            v = s[self.i + len(Q):j]
            self.i = j + len(Q)
            return v
        if s.startswith("{", self.i):
            return self.obj()
        if s.startswith("[", self.i):
            self.i += 1
            out = []
            while not s.startswith("]", self.i):
                out.append(self.value())
                if s.startswith(",", self.i):
                    self.i += 1
            self.i += 1
            return out
        m = re.compile(r"true|false|null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?").match(s, self.i)
        if not m:
            raise ValueError(f"unparsable value at {self.i}: {s[self.i:self.i + 20]!r}")
        self.i = m.end()
        t = m.group(0)
        return {"true": True, "false": False, "null": None}.get(t, float(t) if any(c in t for c in ".eE") else int(t))

    def obj(self):
        s = self.s
        assert s[self.i] == "{"
        self.i += 1
        out = {}
        while not s.startswith("}", self.i):
            m = re.compile(r"[A-Za-z_][\w.-]*").match(s, self.i)
            if not m:
                raise ValueError(f"key expected at {self.i}")
            key = m.group(0)
            self.i = m.end()
            assert s[self.i] == ":"
            self.i += 1
            out[key] = self.value()
            if s.startswith(",", self.i):
                self.i += 1
        self.i += 1
        return out


def parse_calls(raw: str) -> list[tuple[str, dict]]:
    out = []
    for m in CALL_RE.finditer(raw):
        out.append((m.group(1), _P(m.group(2)).obj()))
    return out


def _validate(args: dict, schema: dict) -> bool:
    try:
        import jsonschema
        jsonschema.validate(args, schema)
        return True
    except ImportError:
        return all(k in args for k in schema.get("required", []))
    except Exception:  # noqa: BLE001
        return False


def _served_calls(msg: dict) -> list[tuple[str, dict]]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            out.append((fn.get("name"), json.loads(fn.get("arguments") or "{}")))
        except Exception:  # noqa: BLE001
            out.append((fn.get("name"), {"_unparsable": fn.get("arguments")}))
    return out


def _stream_calls(chunks: list[dict]) -> tuple[list[tuple[str, dict]], str]:
    calls: dict[int, dict] = {}
    content = ""
    for c in chunks:
        d = ((c.get("choices") or [{}])[0].get("delta") or {})
        content += d.get("content") or ""
        for tc in d.get("tool_calls") or []:
            slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            slot["name"] += fn.get("name") or ""
            slot["arguments"] += fn.get("arguments") or ""
    out = []
    for _, v in sorted(calls.items()):
        try:
            out.append((v["name"], json.loads(v["arguments"] or "{}")))
        except Exception:  # noqa: BLE001
            out.append((v["name"], {"_unparsable": v["arguments"]}))
    return out, content


def tool_calls(srv, cases: list[dict]) -> dict:
    rows, ok = [], True
    for c in cases:
        body = {"model": SERVED_NAME, "max_tokens": 1024, "temperature": 0.0, "messages": c["messages"],
                "tools": c["tools"], "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": True}}
        if c.get("tool_choice"):
            body["tool_choice"] = c["tool_choice"]
        r = srv.post("/v1/chat/completions", body)
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        raw = raw_text(ch.get("token_ids") or [])
        expected = parse_calls(raw)
        got = _served_calls(msg)
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in c["tools"]}
        valid = all(n in schemas and _validate(a, schemas[n]) for n, a in got)
        content = msg.get("content") or ""
        good = (r.get("_status") == 200 and got == expected and valid and "<|tool_call>" not in content
                and len(got) == c.get("expected_calls", len(got)))
        st, chunks, _ = srv.stream("/v1/chat/completions", body)
        s_calls, s_content = _stream_calls(chunks)
        s_good = st == 200 and "<|tool_call>" not in s_content and len(s_calls) == c.get("expected_calls", len(s_calls))
        rows.append({"id": c["id"], "status": r.get("_status"), "expected": expected, "served": got, "valid": valid,
                     "stream_status": st, "stream_calls": s_calls, "pass": good and s_good})
        ok &= good and s_good
    return {"rows": rows, "pass": bool(ok)}
