#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Plain HTTP to the explicitly supplied local serving endpoint (no proxies), and the observation error type."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class ObservationError(ValueError):
    """A missing, failed or malformed observation of the served model."""


def request(base: str, path: str, body=None, *, timeout: float = 900, method: str | None = None):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        raise ObservationError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')[:2000]}") from None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def stream(base: str, path: str, body: dict, *, timeout: float = 900) -> list[dict]:
    """The decoded ``data:`` events of a server-sent-event response, in order (``[DONE]`` excluded)."""
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    events = []
    try:
        with opener.open(req, timeout=timeout) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                events.append(json.loads(data))
    except urllib.error.HTTPError as exc:
        raise ObservationError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')[:2000]}") from None
    return events


def metrics(base: str, *, timeout: float = 60) -> dict[str, float]:
    """The Prometheus text of ``/metrics`` as {sample name with labels: value}."""
    req = urllib.request.Request(base.rstrip("/") + "/metrics")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def metric_total(samples: dict[str, float], name: str) -> float:
    """Sum of a counter over its label sets (``name`` or ``name_total``)."""
    return sum(v for k, v in samples.items() if k.split("{", 1)[0] in (name, name + "_total"))
