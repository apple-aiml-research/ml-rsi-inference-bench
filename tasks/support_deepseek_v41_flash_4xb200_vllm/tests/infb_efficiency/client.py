#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The efficiency measurement client for DeepSeek-V4.1-Flash on vLLM: per-user decode speed and output throughput.

This file is the evaluation's own instrument: the grader measures with these same bytes.

What it measures, against one running server:

  * S1, per-user decode speed at concurrency 1: for each of k = 8 requests, (n - 1) / (t_last - t_first)
    over its n = 512 output tokens, where t_first is the arrival of the first chunk that adds a token
    and t_last the arrival of the chunk that completes the 512th; the axis value is the median. Two
    discarded requests come first.
  * S_C, output throughput at concurrency C: N = 4C requests kept C in flight; the sum of their
    output tokens over (last completion - first request). One discarded wave of C concurrent requests
    comes first. This task's C is 16.
  * Every prompt is a distinct document of natural text, cut at its own offset to exactly 4,096 tokens
    and sent as token ids (a token-list ``prompt`` to vLLM's ``/v1/completions``), so no chat template
    runs and the input length is exact. No two prompts share a leading page, and every request carries
    its own ``cache_salt``, so no request can read another's prefix cache.
  * Greedy (temperature 0, sent explicitly), ``ignore_eos``, 512 new tokens, ``skip_special_tokens``
    false. A timed request counts only if it answers HTTP 200, finishes for length and returns exactly
    512 output tokens, counted from the streamed token ids (``return_token_ids``) where the server
    returns them and from the reported count otherwise.

The clock and the counting are the client's own (``time.perf_counter`` at arrival, ids counted here). What
the server reports -- its token count, its cached tokens -- is recorded beside the client's numbers and
flagged when the two disagree.

Standard library only, plus ``tokenizers`` for the model's ``tokenizer.json``. Run it with a clean
environment, in its own process, outside the engine tree.

Usage, against your own server:

    /opt/vllm/bin/python /app/eval/deepseek_v41_flash/serving/client.py measure --engine vllm \\
        --base-url http://127.0.0.1:8000 \\
        --tokenizer /model/deepseek-ai/DeepSeek-V4.1-Flash/tokenizer.json \\
        --texts /app/eval/deepseek_v41_flash/serving/sample.jsonl.xz --concurrency 16

It prints each axis's value, and with ``--record FILE`` writes the full record, every request's timings
included. The README next to this file describes the output.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import fcntl
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable

KIT = "infb-day0-efficiency-kit/1"
RECORD_SCHEMA = "infb-efficiency/1"

# The workload. Only the concurrency C (16 for this task) varies with the recipe.
ISL = 4096
OSL = 512
K_C1 = 8
WARMUPS_C1 = 2
WARMUP_WAVES = 1
N_PER_C = 4
LEAD_GUARD = 64      # no two prompts of one draw share their first 64 token ids
MAX_OFFSET = 512     # each document is cut at its own offset in [0, 512] tokens
IDLE_TIMEOUT_S = 120.0

AXES = ("decode_c1", "throughput")
PHASES = ("c1_warmup", "c1_timed", "cC_warmup", "cC_timed")

# Exit codes of `measure` (the verifier's nodes and the reference step read them).
EXIT_OK = 0
EXIT_BELOW_RHO = 1
EXIT_MEASUREMENT_FAILED = 2
EXIT_DEADLINE = 3
EXIT_USAGE = 4
EXIT_WORKLOAD_MISMATCH = 5


class WorkloadError(Exception):
    pass


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class HFTokenizer:
    """The model's own `tokenizer.json` through `tokenizers` (AutoTokenizer fails on some checkpoints
    under their pinned Transformers, e.g. DeepSeek-V4-Flash under 5.6.0; the file itself always loads)."""

    kind = "tokenizers"

    def __init__(self, path: str):
        from tokenizers import Tokenizer  # noqa: PLC0415 - the one non-stdlib import, loaded on demand

        self.path = str(path)
        self.sha256 = sha256_file(path)
        self._t = Tokenizer.from_file(self.path)

    def encode(self, text: str) -> list[int]:
        return list(self._t.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self._t.decode(list(ids), skip_special_tokens=False)

    def special_prefix(self) -> list[int]:
        """What the tokenizer's own post-processor puts before content (a BOS where it adds one)."""
        probe = "Hello"
        bare = list(self._t.encode(probe, add_special_tokens=False).ids)
        full = list(self._t.encode(probe, add_special_tokens=True).ids)
        for i in range(len(full) - len(bare) + 1):
            if full[i:i + len(bare)] == bare:
                return full[:i]
        return []


class ByteTokenizer:
    """A stand-in for tests and stub servers only: one id per UTF-8 byte, offset by 3."""

    kind = "bytes"
    path = "bytes"
    sha256 = "bytes"

    def encode(self, text: str) -> list[int]:
        return [b + 3 for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(max(0, min(255, i - 3)) for i in ids).decode("utf-8", errors="replace")

    def special_prefix(self) -> list[int]:
        return []


def load_tokenizer(spec: str):
    """`bytes` (tests), the model's `tokenizer.json`, or a checkpoint directory that holds one."""
    if spec == "bytes":
        return ByteTokenizer()
    if Path(spec).is_dir():
        if (Path(spec) / "tokenizer.json").is_file():
            return HFTokenizer(str(Path(spec) / "tokenizer.json"))
        raise WorkloadError(f"no tokenizer.json in {spec}")
    return HFTokenizer(spec)


def _open_text(path: str | os.PathLike):
    name = str(path)
    if name.endswith(".xz"):
        try:
            import lzma  # noqa: PLC0415 - stdlib, but optional in some Python builds
        except ImportError as exc:
            raise WorkloadError(f"{name} is xz-compressed and this Python has no lzma module ({exc})") from exc
        return lzma.open(name, "rt", encoding="utf-8")
    if name.endswith(".gz"):
        return gzip.open(name, "rt", encoding="utf-8")
    return open(name, encoding="utf-8")


def load_texts(path: str | os.PathLike) -> list[dict]:
    """A jsonl (.xz, .gz or plain) of {"id", "family", "text", ...} documents, ordered by id."""
    docs = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                if not isinstance(d.get("id"), str) or not isinstance(d.get("text"), str):
                    raise WorkloadError(f"{path}: a document without a string id and text")
                docs.append(d)
    ids = [d["id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise WorkloadError(f"{path}: duplicate document ids")
    return sorted(docs, key=lambda d: d["id"])


@dataclasses.dataclass
class Prompt:
    pid: str
    phase: str
    doc: str
    family: str
    offset: int
    ids: list[int]

    @property
    def ids_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.ids))

    def describe(self) -> dict:
        return {"id": self.pid, "phase": self.phase, "doc": self.doc, "family": self.family,
                "offset": self.offset, "isl": len(self.ids), "ids_sha256": self.ids_sha256}


@dataclasses.dataclass
class Workload:
    params: dict
    prompts: list[Prompt]
    skipped: list[dict]

    def phase(self, name: str) -> list[Prompt]:
        return [p for p in self.prompts if p.phase == name]

    @property
    def digest(self) -> str:
        return "sha256:" + sha256_bytes(canonical_json(
            {"params": self.params, "prompts": [p.describe() for p in self.prompts]}))


def phase_counts(concurrency: int, k: int = K_C1, warmups: int = WARMUPS_C1,
                 waves: int = WARMUP_WAVES, n_per_c: int = N_PER_C) -> dict[str, int]:
    return {"c1_warmup": warmups, "c1_timed": k, "cC_warmup": waves * concurrency,
            "cC_timed": n_per_c * concurrency}


def draw_workload(docs: list[dict], tokenizer, seed: str, concurrency: int, *, isl: int = ISL,
                  k: int = K_C1, warmups: int = WARMUPS_C1, waves: int = WARMUP_WAVES,
                  n_per_c: int = N_PER_C, max_offset: int = MAX_OFFSET) -> Workload:
    """Deterministic in (documents, tokenizer, seed, parameters): both sides draw the same prompts.

    Documents are shuffled by the seed, each is cut at a seeded offset to exactly `isl` ids (the
    tokenizer's own special prefix first), and a document is skipped when it is too short or its
    first LEAD_GUARD ids repeat an earlier prompt's. The phases take the prompts in order, so
    warm-ups, the S1 requests and the S_C requests never share a document.
    """
    if concurrency < 1:
        raise WorkloadError("concurrency must be at least 1")
    counts = phase_counts(concurrency, k, warmups, waves, n_per_c)
    need = sum(counts.values())
    rng = random.Random("infb-day0-efficiency|" + seed)
    order = list(docs)
    rng.shuffle(order)
    prefix = list(tokenizer.special_prefix())
    body = isl - len(prefix)
    if body <= 0:
        raise WorkloadError("the special prefix is longer than the input length")
    phase_of = [name for name in PHASES for _ in range(counts[name])]
    prompts: list[Prompt] = []
    skipped: list[dict] = []
    leads: set[tuple[int, ...]] = set()
    for doc in order:
        if len(prompts) == need:
            break
        ids = tokenizer.encode(doc["text"])
        if len(ids) < body:
            skipped.append({"doc": doc["id"], "reason": "shorter than the input length", "tokens": len(ids)})
            continue
        offset = rng.randrange(0, min(max_offset, len(ids) - body) + 1)
        cut = prefix + ids[offset:offset + body]
        lead = tuple(cut[:LEAD_GUARD])
        if lead in leads:
            skipped.append({"doc": doc["id"], "reason": "leading ids repeat an earlier prompt"})
            continue
        leads.add(lead)
        phase = phase_of[len(prompts)]
        prompts.append(Prompt(pid=f"p{len(prompts):04d}", phase=phase, doc=doc["id"],
                              family=str(doc.get("family", "")), offset=offset, ids=cut))
    if len(prompts) < need:
        raise WorkloadError(f"the texts yield {len(prompts)} usable documents; the draw needs {need}")
    params = {"isl": isl, "osl": OSL, "concurrency": concurrency, "counts": counts,
              "max_offset": max_offset, "lead_guard": LEAD_GUARD, "special_prefix": prefix,
              "tokenizer_sha256": tokenizer.sha256}
    return Workload(params=params, prompts=prompts, skipped=skipped)


class HttpError(Exception):
    pass


async def _with_deadline(aw, deadline: float, idle: float):
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise asyncio.TimeoutError("deadline")
    return await asyncio.wait_for(aw, timeout=min(idle, remaining))


async def http_stream(host: str, port: int, path: str, body: bytes, on_block: Callable[[bytes, float], None],
                      *, deadline: float, idle_timeout: float = IDLE_TIMEOUT_S,
                      method: str = "POST") -> tuple[int, float, bytes]:
    """POST `body`, feed every received body block to `on_block(bytes, arrival)`.

    Returns (status, t_send, error_body). Chunked, Content-Length and read-to-EOF bodies are all
    handled; a non-200 answer's body is returned rather than streamed.
    """
    reader, writer = await _with_deadline(asyncio.open_connection(host, port), deadline, idle_timeout)
    try:
        head = (f"{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                f"Content-Type: application/json\r\nAccept: text/event-stream\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
        t_send = time.perf_counter()
        writer.write(head + body)
        await _with_deadline(writer.drain(), deadline, idle_timeout)
        status_line = await _with_deadline(reader.readline(), deadline, idle_timeout)
        parts = status_line.decode("latin-1").split()
        if len(parts) < 2 or not parts[1].isdigit():
            raise HttpError(f"bad status line {status_line[:80]!r}")
        status = int(parts[1])
        headers: dict[str, str] = {}
        while True:
            line = await _with_deadline(reader.readline(), deadline, idle_timeout)
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        chunked = "chunked" in headers.get("transfer-encoding", "").lower()
        length = headers.get("content-length")
        sink: list[bytes] = []

        def emit(block: bytes) -> None:
            if status == 200:
                on_block(block, time.perf_counter())
            else:
                sink.append(block)

        if chunked:
            while True:
                size_line = await _with_deadline(reader.readline(), deadline, idle_timeout)
                if not size_line:
                    raise HttpError("connection closed inside a chunked body")
                size = int(size_line.split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while (await _with_deadline(reader.readline(), deadline, idle_timeout)) not in (b"\r\n", b"\n", b""):
                        pass
                    break
                emit(await _with_deadline(reader.readexactly(size), deadline, idle_timeout))
                await _with_deadline(reader.readexactly(2), deadline, idle_timeout)
        elif length is not None:
            remaining = int(length)
            while remaining > 0:
                block = await _with_deadline(reader.read(min(remaining, 1 << 16)), deadline, idle_timeout)
                if not block:
                    raise HttpError("connection closed before Content-Length was reached")
                remaining -= len(block)
                emit(block)
        else:
            while True:
                block = await _with_deadline(reader.read(1 << 16), deadline, idle_timeout)
                if not block:
                    break
                emit(block)
        return status, t_send, b"".join(sink)
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer.wait_closed(), timeout=2)


class SSE:
    """Assemble `data:` events across body blocks; each event keeps the arrival time of its last byte."""

    def __init__(self, on_event: Callable[[Any, float], None]):
        self.buf = b""
        self.on_event = on_event
        self.done = False

    def feed(self, block: bytes, arrival: float) -> None:
        self.buf += block
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.done = True
                continue
            if payload:
                self.on_event(json.loads(payload), arrival)


_METRIC = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)(\s+\d+)?\s*$")


def counter_total(metrics_text: str, name: str) -> float | None:
    """A Prometheus counter summed over every label set, or None where the server exports none.

    Label-aware on purpose: vLLM emits these per engine and per model, so a bare-name lookup finds
    nothing and reads as "no hits". Both the bare name and the `_total` spelling count.
    """
    wanted = {name, f"{name}_total"}
    total, seen = 0.0, False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC.match(line)
        if m and m.group("name") in wanted:
            total += float(m.group("value"))
            seen = True
    return total if seen else None


def scrape_counters(base_url: str, names: tuple[str, ...], deadline: float) -> dict:
    """GET /metrics once; {"available": bool, "counters": {name: total or None}}."""
    if not names:
        return {"available": False, "counters": {}}
    host, port = split_url(base_url)
    got: list[bytes] = []
    try:
        status, _, _ = asyncio.run(http_stream(host, port, "/metrics", b"", lambda b, t: got.append(b),
                                               deadline=min(deadline, time.perf_counter() + 30), method="GET"))
    except (OSError, HttpError, ValueError, asyncio.TimeoutError, asyncio.IncompleteReadError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:200], "counters": {}}
    if status != 200:
        return {"available": False, "error": f"HTTP {status}", "counters": {}}
    text = b"".join(got).decode("utf-8", "replace")
    return {"available": True, "counters": {n: counter_total(text, n) for n in names}}


@dataclasses.dataclass
class Result:
    pid: str
    phase: str
    concurrency: int
    t_send: float = 0.0
    t_first: float | None = None
    t_last: float | None = None
    n_ids: int = 0
    ids: list[int] = dataclasses.field(default_factory=list)
    text: str = ""
    ids_seen: bool = False
    id_mode: str = ""
    finish: str | None = None
    status: int | None = None
    error: str | None = None
    reported: dict = dataclasses.field(default_factory=dict)
    chunks_t: list[float] = dataclasses.field(default_factory=list)
    chunks_n: list[int] = dataclasses.field(default_factory=list)
    target: int = OSL

    def count(self) -> int:
        """Output tokens: the streamed ids where the engine returned any, else its reported count."""
        if self.ids_seen:
            return self.n_ids
        rep = self.reported.get("completion_tokens")
        return int(rep) if isinstance(rep, (int, float)) else 0

    @property
    def count_source(self) -> str:
        return "stream_ids" if self.ids_seen else "reported"

    def _advance(self, n_now: int, arrival: float) -> None:
        if n_now > (self.chunks_n[-1] if self.chunks_n else 0):
            self.chunks_t.append(arrival)
            self.chunks_n.append(n_now)
            if self.t_first is None:
                self.t_first = arrival
            if self.t_last is None and n_now >= self.target:
                self.t_last = arrival

    def valid(self) -> tuple[bool, str]:
        if self.error:
            return False, self.error
        if self.status != 200:
            return False, f"HTTP {self.status}"
        if self.finish != "length":
            return False, f"finish reason {self.finish!r}, not 'length'"
        if self.count() != self.target:
            return False, f"{self.count()} output tokens ({self.count_source}), not {self.target}"
        if self.t_first is None or self.t_last is None or self.t_last <= self.t_first:
            return False, "no measurable decode interval"
        return True, ""

    def per_user_tok_s(self) -> float:
        return (self.target - 1) / (self.t_last - self.t_first)

    def as_record(self, tokenizer=None, side: str = "", axis: str = "") -> dict:
        ok, why = self.valid()
        rec = {
            "side": side, "axis": axis, "prompt": self.pid, "phase": self.phase,
            "concurrency": self.concurrency, "status": self.status, "error": self.error,
            "valid": ok, "invalid_reason": why or None, "finish_reason": self.finish,
            "count": self.count(), "count_source": self.count_source, "id_mode": self.id_mode or None,
            "t_send": self.t_send,
            "ttft_s": (self.t_first - self.t_send) if self.t_first is not None else None,
            "decode_s": (self.t_last - self.t_first) if (self.t_first is not None and self.t_last is not None) else None,
            "done_s": (self.t_last - self.t_send) if self.t_last is not None else None,
            "chunk_ms": [round((t - self.t_send) * 1000.0, 3) for t in self.chunks_t],
            "chunk_n": self.chunks_n,
            "server": self.reported,
            "output_ids": self.ids if self.ids_seen else None,
            "text_sha256": sha256_bytes(self.text.encode("utf-8")),
            "text": self.text,
        }
        rec.update(integrity(self, tokenizer))
        return rec


def integrity(r: Result, tokenizer) -> dict:
    """Recorded, never scored alone: a heuristic that can fail a valid server is not a gate."""
    out: dict[str, Any] = {"first_id": r.ids[0] if (r.ids_seen and r.ids) else None}
    if tokenizer is None:
        return out
    if r.ids_seen and r.text:
        try:
            out["ids_decode_to_text"] = tokenizer.decode(r.ids) == r.text
        except Exception as exc:  # noqa: BLE001
            out["ids_decode_to_text"] = None
            out["decode_error"] = repr(exc)[:200]
    if r.text:
        try:
            out["retokenized_count"] = len(tokenizer.encode(r.text))
        except Exception as exc:  # noqa: BLE001
            out["retokenized_count"] = None
            out["retokenize_error"] = repr(exc)[:200]
    return out


class VLLMAdapter:
    """vLLM's OpenAI `/v1/completions`, streaming, with a token-list prompt, `return_token_ids` (delta
    ids per chunk) and a `cache_salt` unique to every request, so no request can read another's
    prefix cache on either side."""

    name = "vllm"
    path = "/v1/completions"
    # Per-request cached tokens need `enable_prompt_tokens_details`, off by default (usage then carries
    # `prompt_tokens_details: null`), so the prefix cache is read here.
    counters = ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:prompt_tokens_cached")

    def __init__(self, base_url: str, model: str | None = None):
        self.base_url = base_url
        self.model = model

    def resolve_model(self, deadline: float) -> None:
        if self.model:
            return
        host, port = split_url(self.base_url)
        got: list[bytes] = []
        status, _, err = asyncio.run(http_stream(host, port, "/v1/models", b"", lambda b, t: got.append(b),
                                                 deadline=deadline, method="GET"))
        if status != 200:
            raise HttpError(f"/v1/models answered {status}: {err[:200]!r}")
        data = json.loads(b"".join(got))
        self.model = data["data"][0]["id"]

    def payload(self, prompt: Prompt, osl: int) -> dict:
        return {"model": self.model, "prompt": prompt.ids, "max_tokens": osl, "temperature": 0,
                "ignore_eos": True, "skip_special_tokens": False, "stream": True,
                "stream_options": {"include_usage": True}, "return_token_ids": True,
                "cache_salt": "infb-" + uuid.uuid4().hex}

    def on_event(self, ev: Any, arrival: float, r: Result) -> None:
        if not isinstance(ev, dict):
            return
        if ev.get("error") or ev.get("object") == "error":
            r.error = "server error event: " + json.dumps(ev.get("error") or ev)[:300]
            return
        for ch in ev.get("choices") or []:
            tids = ch.get("token_ids")
            if isinstance(tids, list):
                r.ids_seen = True
                r.id_mode = "delta"
                r.ids.extend(tids)
                r.n_ids = len(r.ids)
            if isinstance(ch.get("text"), str):
                r.text += ch["text"]
            if ch.get("finish_reason"):
                r.finish = str(ch["finish_reason"])
        usage = ev.get("usage")
        if isinstance(usage, dict):
            for key in ("completion_tokens", "prompt_tokens"):
                if key in usage:
                    r.reported[key] = usage[key]
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict) and "cached_tokens" in details:
                r.reported["cached_tokens"] = details["cached_tokens"]
        r._advance(r.count(), arrival)


ADAPTERS = {"vllm": VLLMAdapter}


def split_url(base_url: str) -> tuple[str, int]:
    m = re.fullmatch(r"http://([^/:]+):(\d+)/?", base_url.strip())
    if not m:
        raise ValueError(f"base URL must look like http://host:port, got {base_url!r}")
    return m.group(1), int(m.group(2))


async def _one(adapter, host: str, port: int, prompt: Prompt, osl: int, r: Result, deadline: float) -> None:
    """Fill `r` in place, so a request cut off at the deadline keeps its partial timings."""
    sse = SSE(lambda ev, t: adapter.on_event(ev, t, r))
    body = json.dumps(adapter.payload(prompt, osl)).encode()
    r.t_send = time.perf_counter()
    try:
        status, t_send, err = await http_stream(host, port, adapter.path, body, sse.feed, deadline=deadline)
        r.status, r.t_send = status, t_send
        if status != 200:
            r.error = f"HTTP {status}: {err[:300].decode('utf-8', 'replace')}"
    except asyncio.TimeoutError:
        r.error = "deadline reached" if time.perf_counter() >= deadline else f"no data for {IDLE_TIMEOUT_S:.0f} s"
    except (OSError, HttpError, ValueError, asyncio.IncompleteReadError) as exc:
        r.error = f"{type(exc).__name__}: {exc}"[:300]


async def run_phase_async(adapter, prompts: list[Prompt], concurrency: int, osl: int, deadline: float,
                          sink: Callable[[Result], None]) -> tuple[list[Result], bool]:
    """Keep `concurrency` requests in flight until every prompt is sent; stop at `deadline`.

    Returns (results in prompt order, censored). `sink` sees every request once, as it ends; a
    request cut off at the deadline keeps its partial timings and says so in its error.
    """
    host, port = split_url(adapter.base_url)
    sem = asyncio.Semaphore(concurrency)
    results = {p.pid: Result(pid=p.pid, phase=p.phase, concurrency=concurrency, target=osl) for p in prompts}
    started: set[str] = set()
    sunk: set[str] = set()

    async def run(p: Prompt) -> None:
        async with sem:
            started.add(p.pid)
            await _one(adapter, host, port, p, osl, results[p.pid], deadline)
        sunk.add(p.pid)
        sink(results[p.pid])

    tasks = [asyncio.create_task(run(p)) for p in prompts]
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline - time.perf_counter()))
    censored = bool(pending)
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for t in done:
        if not t.cancelled() and t.exception() is not None:
            raise t.exception()
    for p in prompts:
        if p.pid not in sunk:
            r = results[p.pid]
            r.error = r.error or ("cut off at the deadline" if p.pid in started else "not started before the deadline")
            sink(r)
    return [results[p.pid] for p in prompts], censored


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = statistics.quantiles(v, n=4, method="inclusive") if len(v) >= 2 else [v[0], v[0], v[0]]
    return {"n": len(v), "median": statistics.median(v), "mean": statistics.fmean(v), "min": v[0],
            "max": v[-1], "q1": q[0], "q3": q[2], "values": v}


def exact_median(values: list[float]) -> Fraction:
    v = sorted(Fraction(x) for x in values)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def compute_axis(axis: str, timed: list[Result], osl: int) -> dict:
    """The axis value from the timed requests, or the reason there is none.

    S1: the median of (n - 1) / (t_last - t_first). S_C: the sum of output tokens over the window
    from the first timed request's send to the last timed request's final token. Every timed request
    must be valid; one that is not fails the axis, and so does an empty set (a censored warm-up).
    """
    invalid = [(r.pid, r.valid()[1]) for r in timed if not r.valid()[0]]
    out: dict[str, Any] = {"timed": len(timed), "invalid": [{"prompt": p, "reason": w} for p, w in invalid]}
    ttft = [r.t_first - r.t_send for r in timed if r.t_first is not None]
    tpot = [(r.t_last - r.t_first) / (osl - 1) for r in timed if r.t_first is not None and r.t_last is not None]
    out["ttft_s"] = quantiles(ttft)
    out["tpot_s"] = quantiles(tpot)
    if invalid or not timed:
        out["value"] = None
        out["value_exact"] = None
        return out
    if axis == "decode_c1":
        per = [r.per_user_tok_s() for r in timed]
        exact = exact_median([Fraction(osl - 1) / (Fraction(r.t_last) - Fraction(r.t_first)) for r in timed])
        out["per_request_tok_s"] = quantiles(per)
        out["definition"] = "median over the timed requests of (n - 1) / (t_last - t_first), n = output tokens"
    else:
        t0 = min(r.t_send for r in timed)
        t1 = max(r.t_last for r in timed)
        tokens = sum(r.count() for r in timed)
        exact = Fraction(tokens) / (Fraction(t1) - Fraction(t0))
        out["window"] = {"definition": "sum of output tokens / (last timed completion - first timed send)",
                         "t0": t0, "t1": t1, "seconds": t1 - t0, "tokens": tokens}
    out["value"] = float(exact)
    out["value_exact"] = f"{exact.numerator}/{exact.denominator}"
    return out


def meets_rho(candidate_exact: str, reference_exact: str, rho: str) -> bool:
    """candidate >= rho * reference, in exact rational arithmetic (no float at the boundary)."""
    return Fraction(candidate_exact) >= Fraction(rho) * Fraction(reference_exact)


def environment_snapshot(label: str) -> dict:
    snap: dict[str, Any] = {"label": label, "captured_at": utcnow(), "host": platform.node(),
                            "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    with contextlib.suppress(OSError):
        snap["loadavg"] = list(os.getloadavg())
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                snap["cpu_model"] = line.split(":", 1)[1].strip()
                break
    smi = shutil.which("nvidia-smi")
    if not smi:
        snap["gpus_unreadable"] = "nvidia-smi is not on PATH"
    else:
        fields = ("index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,clocks.sm,"
                  "clocks.max.sm,clocks.mem,power.limit,power.draw,temperature.gpu")
        keys = fields.split(",")
        try:
            p = subprocess.run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            rows = [[c.strip() for c in row.split(",")] for row in p.stdout.strip().splitlines() if row.strip()]
            good = [dict(zip(keys, r)) for r in rows if len(r) == len(keys)]
            if p.returncode == 0 and good:
                snap["gpus"] = good
            if p.returncode != 0 or not good or len(good) != len(rows):
                # A failing nvidia-smi prints its complaint on stdout and exits non-zero; parsed, each
                # line became a "GPU" whose index was the error. Recorded as unreadable, under its own
                # key, and never as rows.
                snap["gpus_unreadable"] = (f"nvidia-smi exited {p.returncode}: "
                                           + ((p.stderr or p.stdout).strip() or "no output"))[:300]
            v = subprocess.run([smi], capture_output=True, text=True, timeout=20)
            m = re.search(r"CUDA Version:\s*([\d.]+)", v.stdout) if v.returncode == 0 else None
            snap["cuda_driver_api"] = m.group(1) if m else None
        except Exception as exc:  # noqa: BLE001 - a diagnostic never fails a measurement
            snap["gpus_unreadable"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        from importlib import metadata  # noqa: PLC0415 - reads dist-info only; imports no package code
        libs = {}
        for dist in ("torch", "sglang", "vllm", "flashinfer-python", "sgl-deep-gemm", "deep-gemm",
                     "sglang-kernel", "sgl-kernel", "tokenizers", "transformers", "triton"):
            with contextlib.suppress(metadata.PackageNotFoundError):
                libs[dist] = metadata.version(dist)
        snap["libraries"] = libs
    except Exception:  # noqa: BLE001
        pass
    return snap


def new_record() -> dict:
    return {"schema": RECORD_SCHEMA, "kit": KIT, "status": "in_progress", "created_at": utcnow(),
            "updated_at": utcnow(), "provenance": {}, "workload": {}, "environment": [], "phases": [],
            "axes": {"reference": {}, "candidate": {}},
            "summary": {"rho": None, "verdicts": {}, "ratios": {}, "flags": [], "deadlines_hit": []},
            "sidecar": {"requests": "requests.jsonl.gz"}}


@contextlib.contextmanager
def locked(path: Path):
    lock = path.with_name(path.name + ".lock")
    with open(lock, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def read_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_record(path: Path, rec: dict) -> None:
    rec["updated_at"] = utcnow()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def update_record(path: Path, mutate: Callable[[dict], None]) -> dict:
    """Read, mutate, write atomically under a lock: a crash at any point leaves the last good file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path):
        rec = read_record(path) if path.exists() else new_record()
        mutate(rec)
        write_record(path, rec)
        return rec


def append_request(sidecar: Path, rec: dict) -> None:
    """One gzip member per request, appended and closed at once, so a kill keeps every earlier one."""
    with gzip.open(sidecar, "ab") as fh:
        fh.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))


def read_requests(sidecar: Path) -> list[dict]:
    out = []
    if not sidecar.exists():
        return out
    with contextlib.suppress(EOFError, OSError), gzip.open(sidecar, "rt", encoding="utf-8") as fh:
        for line in fh:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


#: decode_c1's paired statistic: the median over prompts of candidate speed / reference speed.
DECODE_C1_STATISTICS = ("median_of_ratios",)


def prompt_speed(r, osl: int) -> dict:
    """One timed decode_c1 request's own speed, exact: (n - 1) / (t_last - t_first) -- the term the axis's median
    is taken over, recorded per prompt so a paired statistic can be computed from the record."""
    if r.t_first is None or r.t_last is None or r.t_last <= r.t_first:
        return {}
    x = Fraction(osl - 1) / (Fraction(r.t_last) - Fraction(r.t_first))
    return {"tok_s": float(x), "tok_s_exact": f"{x.numerator}/{x.denominator}"}


def median_of_ratios(ref_axis: dict, cand_axis: dict) -> Fraction | None:
    """decode_c1's paired statistic: the median over the prompts both sides timed of candidate speed / reference speed."""
    rp, cp = ref_axis.get("per_prompt") or {}, cand_axis.get("per_prompt") or {}
    common = [p for p in rp if p in cp and rp[p].get("tok_s_exact") and cp[p].get("tok_s_exact")]
    if not common:
        return None
    return exact_median([Fraction(cp[p]["tok_s_exact"]) / Fraction(rp[p]["tok_s_exact"]) for p in common])


def summarize(rec: dict) -> None:
    """Recompute the summary from the axes: values, exact ratios and verdicts, and the quality flags."""
    s = rec["summary"]
    rho = s.get("rho")
    ref, cand = rec["axes"].get("reference", {}), rec["axes"].get("candidate", {})
    flags: list[dict] = []
    for side, axes in (("reference", ref), ("candidate", cand)):
        for axis, a in axes.items():
            for f in a.get("flags", []):
                flags.append({"side": side, "axis": axis, **f})
            if a.get("censored"):
                flags.append({"side": side, "axis": axis, "flag": "censored",
                              "detail": "cut off by its deadline; partial timings are in the sidecar"})
    s["flags"] = flags
    s["reference"] = {axis: ref.get(axis, {}).get("value") for axis in AXES}
    s["candidate"] = {axis: cand.get(axis, {}).get("value") for axis in AXES}
    for axis in AXES:
        r_ex = ref.get(axis, {}).get("value_exact")
        c_ex = cand.get(axis, {}).get("value_exact")
        if axis not in cand:
            continue
        bar = rho
        if r_ex and c_ex and bar is not None and axis == "decode_c1":
            paired = median_of_ratios(ref.get(axis, {}), cand.get(axis, {}))
            s["ratios"][axis] = float(paired) if paired is not None else None
            s["verdicts"][axis] = "pass" if paired is not None and paired >= Fraction(str(bar)) else "fail"
        elif r_ex and c_ex and bar is not None:
            s["ratios"][axis] = float(Fraction(c_ex) / Fraction(r_ex))
            s["verdicts"][axis] = "pass" if meets_rho(c_ex, r_ex, str(bar)) else "fail"
        else:
            s["ratios"][axis] = None
            s["verdicts"][axis] = "fail"


def axis_flags(side: str, results: list[Result], recs: list[dict], reference_prompts: dict | None) -> list[dict]:
    flags = []
    timed = [(r, rec) for r, rec in zip(results, recs) if r.phase.endswith("_timed")]
    errors = [r.pid for r, _ in timed if r.error]
    if errors:
        flags.append({"flag": "request_errors", "detail": f"{len(errors)} timed request(s) errored", "prompts": errors[:20]})
    for r, rec in timed:
        rep = r.reported.get("completion_tokens")
        if r.ids_seen and isinstance(rep, (int, float)) and int(rep) != r.n_ids:
            flags.append({"flag": "client_server_count_disagree", "prompt": r.pid,
                          "detail": f"streamed ids {r.n_ids}, server says {rep}"})
        lat = r.reported.get("e2e_latency")
        if isinstance(lat, (int, float)) and r.t_last is not None and lat > 0:
            client = r.t_last - r.t_send
            if abs(client - lat) / client > 0.02:
                flags.append({"flag": "client_server_latency_disagree", "prompt": r.pid,
                              "detail": f"client {client:.4f} s, server {lat:.4f} s"})
        cached = r.reported.get("cached_tokens")
        if isinstance(cached, (int, float)) and cached > 0:
            flags.append({"flag": "cached_tokens_nonzero", "prompt": r.pid, "detail": f"{cached} cached prompt tokens"})
        if rec.get("ids_decode_to_text") is False:
            flags.append({"flag": "ids_do_not_decode_to_text", "prompt": r.pid})
        if reference_prompts and r.pid in reference_prompts:
            ref = reference_prompts[r.pid]
            rt, rr = rec.get("retokenized_count"), ref.get("retokenized_count")
            if isinstance(rt, int) and isinstance(rr, int) and rr > 0 and abs(rt - rr) / rr > 0.02:
                flags.append({"flag": "retokenized_count_off_reference", "prompt": r.pid,
                              "detail": f"{rt} against the reference's {rr}"})
            if rec.get("first_id") is not None and ref.get("first_id") is not None and rec["first_id"] != ref["first_id"]:
                flags.append({"flag": "first_token_differs_from_reference", "prompt": r.pid})
    return flags


def measure(args) -> int:
    t_start = time.perf_counter()
    deadline = t_start + float(args.deadline_s) if args.deadline_s else float("inf")
    tokenizer = load_tokenizer(args.tokenizer)
    docs = load_texts(args.texts)
    if args.seed_file:
        seed = json.loads(Path(args.seed_file).read_text())["seed"]
    elif args.seed:
        seed = args.seed
    else:
        seed = secrets.token_hex(8)
        if not args.quiet:
            print(f"[efficiency] seed {seed} (random; pass --seed {seed} to repeat this draw)", flush=True)
    workload = draw_workload(docs, tokenizer, str(seed), args.concurrency, isl=args.isl)
    adapter = ADAPTERS[args.engine](args.base_url, args.model)
    if hasattr(adapter, "resolve_model"):
        adapter.resolve_model(min(deadline, time.perf_counter() + 60))
    axes = list(AXES) if args.axis == "both" else [args.axis]
    record = Path(args.record) if args.record else None
    sidecar = (Path(args.sidecar) if args.sidecar else record.with_name("requests.jsonl.gz")) if record else None
    side = args.side
    osl = OSL

    texts_digest = sha256_file(args.texts)
    workload_section = {
        "texts": {"path": str(args.texts), "sha256": texts_digest, "documents": len(docs)},
        "seed_commitment": "sha256:" + sha256_bytes(str(seed).encode()),
        "isl": args.isl, "osl": osl,
        "sampling": {"temperature": 0, "ignore_eos": True, "skip_special_tokens": False, "max_tokens": osl},
        "k": K_C1, "c1_warmups": WARMUPS_C1, "C": args.concurrency, "N": N_PER_C * args.concurrency,
        "cC_warmup_waves": WARMUP_WAVES, "phase_order": list(PHASES),
        "tokenizer": {"path": str(args.tokenizer), "sha256": tokenizer.sha256,
                      "special_prefix": workload.params["special_prefix"]},
        "engine": args.engine, "skipped_documents": workload.skipped,
        "prompts": [p.describe() for p in workload.prompts], "digest": workload.digest,
    }
    if record:
        def begin(rec: dict) -> None:
            if not rec["workload"]:
                rec["workload"] = workload_section
                rec["sidecar"]["requests"] = sidecar.name
            elif rec["workload"].get("digest") != workload.digest:
                raise WorkloadError(f"workload digest {workload.digest} differs from the record's "
                                    f"{rec['workload'].get('digest')}: the two sides would not see the same prompts")
            if args.provenance:
                rec["provenance"].update(json.loads(Path(args.provenance).read_text()))
            rec["provenance"].setdefault("started_at", utcnow())
            if args.rho is not None:
                rec["summary"]["rho"] = str(args.rho)
            labels = {e.get("label") for e in rec["environment"]}
            if side not in labels:
                rec["environment"].append(environment_snapshot(side))
        try:
            update_record(record, begin)
        except WorkloadError as exc:
            print(f"[efficiency] {exc}", flush=True)
            return EXIT_WORKLOAD_MISMATCH

    reference_prompts = None
    if record and side == "candidate":
        reference_prompts = {}
        for axis in axes:
            reference_prompts.update(read_record(record)["axes"].get("reference", {}).get(axis, {}).get("per_prompt", {}))

    worst = EXIT_OK
    report: dict[str, Any] = {"engine": args.engine, "side": side, "concurrency": args.concurrency,
                              "seed_commitment": workload_section["seed_commitment"], "axes": {}}
    for axis in axes:
        warm_phase, timed_phase = (("c1_warmup", "c1_timed") if axis == "decode_c1" else ("cC_warmup", "cC_timed"))
        conc = 1 if axis == "decode_c1" else args.concurrency
        started = utcnow()
        if record:
            update_record(record, lambda rec: rec["axes"].setdefault(side, {}).__setitem__(
                axis, {"status": "running", "started_at": started, "concurrency": conc,
                       "deadline_s": args.deadline_s}))
        all_results: list[Result] = []
        all_recs: list[dict] = []

        def sink(r: Result) -> None:
            rec = r.as_record(tokenizer, side=side, axis=axis)
            all_results.append(r)
            all_recs.append(rec)
            if sidecar:
                append_request(sidecar, rec)

        t_axis = time.perf_counter()
        warm, censored_w = asyncio.run(run_phase_async(adapter, workload.phase(warm_phase), conc, osl, deadline, sink))
        before = None if censored_w else scrape_counters(args.base_url, adapter.counters, deadline)
        timed, censored_t = ([], True) if censored_w else asyncio.run(
            run_phase_async(adapter, workload.phase(timed_phase), conc, osl, deadline, sink))
        after = None if censored_t else scrape_counters(args.base_url, adapter.counters, deadline)
        censored = censored_w or censored_t
        result = compute_axis(axis, timed, osl)
        warm_bad = [r.pid for r in warm if r.error]
        result.update({
            "axis": axis, "side": side, "concurrency": conc, "started_at": started, "finished_at": utcnow(),
            "duration_s": time.perf_counter() - t_axis, "deadline_s": args.deadline_s, "censored": censored,
            "warmups": len(warm), "warmup_errors": warm_bad,
            "status": "censored" if censored else ("complete" if result.get("value") is not None else "failed"),
        })
        by_pid = {r.pid: rec for r, rec in zip(all_results, all_recs)}
        result["per_prompt"] = {r.pid: {"retokenized_count": by_pid[r.pid].get("retokenized_count"),
                                        "first_id": by_pid[r.pid].get("first_id"), "count": r.count(),
                                        **(prompt_speed(r, osl) if axis == "decode_c1" else {})}
                                for r in timed if r.pid in by_pid}
        result["flags"] = axis_flags(side, all_results, all_recs, reference_prompts)
        if before is not None and after is not None and before["available"] and after["available"]:
            delta = {n: (after["counters"][n] - before["counters"][n])
                     if after["counters"].get(n) is not None and before["counters"].get(n) is not None else None
                     for n in adapter.counters}
            result["server_counters"] = {"phase": timed_phase, "delta": delta, "before": before["counters"],
                                         "after": after["counters"]}
            hits = delta.get("vllm:prefix_cache_hits")
            if hits:
                result["flags"].append({"flag": "server_prefix_cache_hits", "detail":
                                        f"the server's prefix cache hit {hits:g} tokens during the timed phase"})
        elif adapter.counters and not censored:
            result["server_counters"] = {"phase": timed_phase, "available": False,
                                         "error": ((after or before or {}).get("error"))}
        if record:
            def finish(rec: dict, result=result) -> None:
                rec["axes"].setdefault(side, {})[axis] = result
                if censored:
                    rec["summary"].setdefault("deadlines_hit", []).append({"side": side, "axis": axis,
                                                                           "deadline_s": args.deadline_s})
                summarize(rec)
            rec = update_record(record, finish)
            verdict = rec["summary"]["verdicts"].get(axis) if side == "candidate" else None
            ratio = rec["summary"]["ratios"].get(axis) if side == "candidate" else None
        else:
            verdict = ratio = None
        code = (EXIT_DEADLINE if censored else EXIT_MEASUREMENT_FAILED if result.get("value") is None
                else EXIT_BELOW_RHO if verdict == "fail" else EXIT_OK)
        worst = max(worst, code)  # OK < below rho < failed < deadline
        report["axes"][axis] = {"value_tok_s": result.get("value"), "status": result["status"],
                                "invalid": result["invalid"][:5], "ratio": ratio, "verdict": verdict,
                                "ttft_median_s": result["ttft_s"].get("median"),
                                "duration_s": round(result["duration_s"], 3)}
        if not args.quiet:
            v = result.get("value")
            label = "S1 (per-user decode, c=1)" if axis == "decode_c1" else f"S_C (output throughput, c={conc})"
            print(f"[efficiency] {label}: {'%.2f tok/s' % v if v is not None else 'not measured'}"
                  f" [{result['status']}]" + (f", ratio {ratio:.4f} -> {verdict}" if ratio is not None else "")
                  + (f"; invalid: {result['invalid'][:3]}" if result["invalid"] else ""), flush=True)
        if censored:
            break
    if record:
        def close(rec: dict) -> None:
            if side == "candidate" and all(a in rec["axes"].get("candidate", {}) for a in AXES):
                rec["status"] = "complete" if not rec["summary"]["deadlines_hit"] else "censored"
            elif worst == EXIT_DEADLINE:
                rec["status"] = "censored"
        update_record(record, close)
    print("INFB_EFFICIENCY " + json.dumps(report, sort_keys=True), flush=True)
    return worst


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="measure S1 and/or S_C against a running server")
    m.add_argument("--engine", choices=sorted(ADAPTERS), required=True)
    m.add_argument("--base-url", required=True, help="http://host:port of the server")
    m.add_argument("--tokenizer", required=True, help="the model's tokenizer.json")
    m.add_argument("--texts", required=True, help="the documents: jsonl, jsonl.gz or jsonl.xz")
    m.add_argument("--concurrency", type=int, required=True, help="C, the recipe's load")
    m.add_argument("--axis", choices=(*AXES, "both"), default="both")
    m.add_argument("--seed", help="the draw's seed (default: random, printed)")
    m.add_argument("--seed-file", help="a JSON file {\"seed\": ...} (the verifier's hidden seed)")
    m.add_argument("--model", help="vLLM: the served model name (default: the first of /v1/models)")
    m.add_argument("--isl", type=int, default=ISL)
    m.add_argument("--deadline-s", type=float, help="stop, censored, after this many seconds")
    m.add_argument("--record", help="the core record to update (verifier)")
    m.add_argument("--sidecar", help="the per-request sidecar (default: next to the record)")
    m.add_argument("--side", choices=("reference", "candidate", "dev"), default="dev")
    m.add_argument("--rho", help="the verifier's tolerance (set by the verifier)")
    m.add_argument("--provenance", help="a JSON file merged into the record's provenance (verifier)")
    m.add_argument("--quiet", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        return measure(args)
    except (WorkloadError, HttpError, ValueError) as exc:
        print(f"[efficiency] {type(exc).__name__}: {exc}", flush=True)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
