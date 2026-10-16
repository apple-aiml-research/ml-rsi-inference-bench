#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The single-request acceptance probe: upstream's `send_one_prompt` on a running server."""
from __future__ import annotations

import requests


def acceptance_probe(base_url: str, bar: float, attempts: int = 3,
                     max_new_tokens: int = 2048) -> tuple[float, float, list[tuple[float, float]]]:
    from sglang.test.send_one import BenchArgs, send_one_prompt

    args = BenchArgs(port=int(base_url.split(":")[-1]), max_new_tokens=max_new_tokens)
    acc_length, speed = 0.0, 0.0
    readings = []
    for attempt in range(1, attempts + 1):
        requests.get(base_url + "/flush_cache")
        acc_length, speed = send_one_prompt(args, label=f"attempt {attempt}", print_output=False)
        readings.append((acc_length, speed))
        if acc_length > bar:
            break
    requests.get(base_url + "/flush_cache")
    return acc_length, speed, readings
