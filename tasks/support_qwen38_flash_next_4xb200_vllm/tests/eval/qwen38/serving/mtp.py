#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Collect complete greedy series and draft-counter draws; inputs and policy are caller-owned."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from qwen38.measurement import budget
from qwen38.measurement import observations as obs


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def raw_record(protocol, value):
    """Optional caller-owned raw evidence, written before interpretation."""
    root = os.environ.get('QWEN38_OBSERVATIONS_DIR')
    if root:
        directory = Path(root); directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{time.time_ns()}-{protocol}-{uuid.uuid4().hex}.json'
        path.write_text(json.dumps(value, allow_nan=False))


def request(base: str, path: str, body=None, *, timeout: float = 1800):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # Requests address the explicitly supplied serving endpoint, including localhost.
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                req, timeout=budget.timeout(timeout)) as response:
            raw = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw_record('http', {'base': base, 'path': path, 'request': body,
                            'status': exc.code, 'response_text': exc.read().decode(errors='replace')})
        raise
    raw_record('http', {'base': base, 'path': path, 'request': body, 'status': status, 'response_text': raw})
    return raw if path == "/metrics" else json.loads(raw)


def stream_request(base: str, path: str, body, *, timeout: float = 600) -> list[str]:
    """Every SSE ``data:`` payload of a streamed request, in order, ending with ``[DONE]`` when the
    server sent it; the caller validates the assembled record."""
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    payloads = []
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            req, timeout=budget.timeout(timeout)) as response:
        for line in response:
            line = line.decode().strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data: "):
                raise obs.ObservationError("malformed SSE line")
            payloads.append(line[6:])
            if payloads[-1] == "[DONE]":
                break
    raw_record('stream', {'base': base, 'path': path, 'request': body, 'payloads': payloads})
    return payloads


def source_identity(app: str | Path = "/app") -> str:
    """Record the commit, tracked diff and new source files during development."""
    root = str(app)
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"])
    changes = subprocess.check_output(["git", "-C", root, "diff", "--no-ext-diff", "--binary", "HEAD"])
    h = hashlib.sha256(head + b"\0" + changes)
    new = subprocess.check_output(["git", "-C", root, "ls-files", "--others", "--exclude-standard", "-z"])
    for encoded in sorted(name for name in new.split(b"\0") if name):
        path = Path(root) / os.fsdecode(encoded)
        contents = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        h.update(encoded + b"\0" + hashlib.sha256(contents).digest())
    return h.hexdigest()


def tokenizer_identity(model: str | Path = "/model") -> str:
    root = Path(model)
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        h.update(name.encode() + b"\0" + (root / name).read_bytes())
    return h.hexdigest()


def fixed_length(base: str, model: str, ids: list[int], n: int, *, case_id: str, vocabulary: int,
                 salt: str | None = None, timeout: float = 1800) -> dict:
    """One greedy fixed-length continuation with top-five probabilities, validated."""
    body = {"model": model, "prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": True,
            "logprobs": 5, "seed": 0, "skip_special_tokens": False, "return_tokens_as_token_ids": True}
    if salt is not None:
        body["cache_salt"] = salt
    begun = time.monotonic()
    response = request(base, "/v1/completions", body, timeout=timeout)
    row = obs.completion(response, case_id, requested=n, ignore_eos=True, vocabulary=vocabulary,
                         prompt_tokens=len(ids))
    row["seconds"] = time.monotonic() - begun
    return row


def collect_series(base: str, n: int, cases: list[tuple[str, str | list[int]]], *, recipe: str,
                   evaluation_id: str, source: str, tokenizer: str, vocabulary: int,
                   timeout: float = 1800) -> dict:
    """Cold fixed-length continuations. A failed request is recorded as that case's error and fails
    the series' structural validation; it is never dropped."""
    model = request(base, "/v1/models", timeout=60)["data"][0]["id"]
    tag = time.time_ns()
    rows, tokenized = [], []
    for cid, prompt in cases:
        try:
            ids = prompt if isinstance(prompt, list) else request(
                base, "/tokenize", {"model": model, "prompt": prompt, "add_special_tokens": True},
                timeout=60)["tokens"]
            obs.require(isinstance(ids, list) and len(ids) > 0, f"{cid}: empty tokenization")
            ids = [obs.token_id(t, f"{cid}/prompt", vocabulary=vocabulary) for t in ids]
            tokenized.append({"id": cid, "tokens": ids})
            rows.append(fixed_length(base, model, ids, n, case_id=cid, vocabulary=vocabulary,
                                     salt=f"qwen38-series-{tag}-{cid}", timeout=timeout))
        except Exception as exc:
            rows.append({"id": cid, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    return {"schema": "qwen38-greedy-series/3", "max_tokens": n, "ignore_eos": True,
            "recipe": recipe, "evaluation_id": evaluation_id, "source_identity": source,
            "tokenizer_identity": tokenizer, "workload_identity": digest(tokenized),
            "vocabulary": vocabulary, "inputs": tokenized, "cases": rows}


def acceptance(base: str, n: int, cases: list[tuple[str, str | list[int]]], *,
               evaluation_id: str, source: str, tokenizer: str, vocabulary: int) -> dict:
    """Isolated fixed-length requests and physically valid draft-counter deltas."""
    before = obs.parse_spec_counters(request(base, "/metrics", timeout=60))
    series = collect_series(base, n, cases, recipe="mtp3", evaluation_id=evaluation_id,
                            source=source, tokenizer=tokenizer, vocabulary=vocabulary)
    after = obs.parse_spec_counters(request(base, "/metrics", timeout=60))
    obs.validate_series(series, [cid for cid, _ in cases], n, vocabulary=vocabulary)
    return {**obs.counter_delta(before, after, completion_tokens=n * len(cases)), "series": series}
