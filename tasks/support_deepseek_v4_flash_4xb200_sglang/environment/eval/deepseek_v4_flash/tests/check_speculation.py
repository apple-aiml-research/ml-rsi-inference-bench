#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The speculative acceptance length on a running server, through upstream's own `send_one_prompt`
(`python/sglang/test/send_one.py`), which computes `completion_tokens / spec_verify_ct` for one greedy request of 256
new tokens at batch size 1. Three attempts; the grader records the same three as a diagnostic. There is no pass mark:
an acceptance length of exactly 1.0 means nothing was ever accepted from the draft."""
from __future__ import annotations


def acceptance_lengths(base_url: str, attempts: int = 3) -> list[tuple[float, float]]:
    """(acceptance length, bs=1 decode speed in tok/s) for each attempt."""
    from sglang.test.send_one import BenchArgs, send_one_prompt  # noqa: PLC0415 - the engine's own probe

    host, port = base_url.removeprefix("http://").split(":")
    values = []
    for _ in range(attempts):
        acc_length, speed = send_one_prompt(
            BenchArgs(
                host=host,
                port=int(port),
                batch_size=1,
                temperature=0.0,
                max_new_tokens=256,
            )
        )
        values.append((acc_length, speed))
    return values


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="speculative acceptance length (a diagnostic, no pass mark)")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = ap.parse_args(argv)
    values = acceptance_lengths(args.base_url)
    best = max(values, key=lambda v: v[0])
    print(f"speculative acceptance length {best[0]:.3f} (three attempts: {[round(v[0], 3) for v in values]}), "
          f"bs=1 decode speed {best[1]:.1f} tok/s (attempts: {[round(v[1], 1) for v in values]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
