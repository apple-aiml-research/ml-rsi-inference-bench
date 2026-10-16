#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""HTTP helpers shared by the serving protocols (a running SGLang server at a base URL).

Every protocol talks to the server through the engine's own interfaces: `/generate` with token IDs,
`/flush_cache`, `/server_info`, `/health`. Nothing here changes how the server runs.
"""
from __future__ import annotations

import time

import requests


def generate(base_url: str, input_ids: list[int], *, max_new_tokens: int, timeout: float,
             return_logprob: bool = True, logprob_start_len: int = -1, top_logprobs_num: int = 0,
             ignore_eos: bool = True) -> dict:
    """One greedy `/generate` call from token IDs; returns the response JSON (raises on HTTP errors)."""
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens, "ignore_eos": ignore_eos},
        "return_logprob": return_logprob,
        "logprob_start_len": logprob_start_len,
    }
    if top_logprobs_num:
        payload["top_logprobs_num"] = top_logprobs_num
    r = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def flush_cache(base_url: str, timeout: float = 300.0) -> None:
    """`POST /flush_cache` until it succeeds; the protocol requires it to succeed.

    The server refuses (400) while it still counts a request as running, which it can do for a moment
    after that request's response has arrived; the call is repeated every 0.5 s until `timeout`.
    """
    deadline = time.monotonic() + timeout
    while True:
        r = requests.post(f"{base_url}/flush_cache", timeout=timeout)
        if r.status_code != 400 or time.monotonic() >= deadline:
            r.raise_for_status()
            return
        time.sleep(0.5)


def server_info(base_url: str, timeout: float = 300.0) -> dict:
    r = requests.get(f"{base_url}/server_info", timeout=timeout)
    r.raise_for_status()
    return r.json()


def wait_healthy(base_url: str, timeout: float) -> float:
    """Poll `/health` until it answers 200; returns the seconds waited."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if requests.get(f"{base_url}/health", timeout=10).status_code == 200:
                return time.monotonic() - t0
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"{base_url} not healthy after {timeout} s")


def output_series(response: dict) -> tuple[list[int], list[float]]:
    """The generated token IDs and their log-probabilities from a `/generate` response."""
    rows = response["meta_info"]["output_token_logprobs"]
    return [int(r[1]) for r in rows], [float(r[0]) for r in rows]


def input_series(response: dict) -> tuple[list[int], list[float | None]]:
    """The prompt token IDs and their log-probabilities (the first position has none)."""
    rows = response["meta_info"]["input_token_logprobs"]
    return [int(r[1]) for r in rows], [None if r[0] is None else float(r[0]) for r in rows]
