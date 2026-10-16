#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Sequential time-to-first-token client for a running SGLang server.

One request at a time to /generate with stream=true; the first streamed chunk is the first token.
Images are noise JPEGs at seeded random resolutions inside the declared range. Writes one record in
the shape sglang.bench_serving writes.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

# One image, wrapped the way this model's chat template wraps one.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
# The text half of an image request: one word repeated, sized by --input-len.
FILLER_WORD = "context "
# Resolutions are drawn on a 32-pixel grid.
STEP = 32
# Vocabulary for text-only prompts of an exact token count.
WORDS = (
    "context", "window", "report", "signal", "matter", "record", "table", "figure", "system",
    "model", "value", "point", "line", "note", "state", "field", "order", "level", "group",
    "result", "number", "series", "region", "sample", "detail", "period", "source", "market",
    "public", "member", "policy", "answer", "second", "reason", "moment", "change", "thing",
    "place", "world", "house",
)


def parse_hw(value: str) -> tuple[int, int]:
    """`HEIGHTxWIDTH` -> (h, w)."""
    try:
        h_s, w_s = value.lower().split("x", 1)
        h, w = int(h_s), int(w_s)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--image-min-hw/--image-max-hw want HEIGHTxWIDTH, got {value!r}") from exc
    if h <= 0 or w <= 0:
        raise SystemExit(f"non-positive image size: {value!r}")
    return h, w


def predicted_tokens(imgs: list[tuple[int, int]], step: int = STEP) -> int:
    return sum((h // step) * (w // step) for h, w in imgs)


def draw_resolutions(rng: random.Random, n_requests: int, per_request: int,
                     lo: tuple[int, int], hi: tuple[int, int],
                     step: int = STEP) -> list[list[tuple[int, int]]]:
    """Seeded sizes on the grid; per-request totals are kept distinct where the range allows."""
    out: list[list[tuple[int, int]]] = []
    taken: set[int] = set()
    attempts_left = 200 * n_requests
    while len(out) < n_requests and attempts_left > 0:
        attempts_left -= 1
        imgs = [(rng.randrange(lo[0], hi[0] + 1, step), rng.randrange(lo[1], hi[1] + 1, step))
                for _ in range(per_request)]
        total = predicted_tokens(imgs, step)
        if total in taken:
            continue
        taken.add(total)
        out.append(imgs)
    while len(out) < n_requests:
        out.append([(rng.randrange(lo[0], hi[0] + 1, step), rng.randrange(lo[1], hi[1] + 1, step))
                    for _ in range(per_request)])
    return out


def encode_image(rng: random.Random, h: int, w: int, fmt: str) -> str:
    """A base64 data URI holding an image of random noise at (h, w)."""
    from PIL import Image  # imported here so `--help` works without Pillow

    n = h * w * 3
    img = Image.frombytes("RGB", (w, h), rng.getrandbits(8 * n).to_bytes(n, "little"))
    sink = io.BytesIO()
    pil_fmt = "JPEG" if fmt.lower() in ("jpeg", "jpg") else fmt.upper()
    img.save(sink, format=pil_fmt, quality=85)
    mime = "jpeg" if pil_fmt == "JPEG" else pil_fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(sink.getvalue()).decode("ascii")


def exact_length_text(rng: random.Random, tokenizer, n_tokens: int) -> str:
    """A text prompt of exactly n_tokens tokens when a tokenizer is given; n words otherwise."""
    words = [rng.choice(WORDS) for _ in range(max(1, n_tokens))]
    text = " ".join(words)
    if tokenizer is None:
        return text
    for _ in range(64):
        got = len(tokenizer.encode(text, add_special_tokens=False))
        if got == n_tokens:
            break
        if got > n_tokens:
            words = words[: max(1, len(words) - (got - n_tokens))]
        else:
            words += [rng.choice(WORDS) for _ in range(n_tokens - got)]
        text = " ".join(words)
    return text


def build_payloads(seed: int, num_prompts: int, image_count: int, lo: tuple[int, int],
                   hi: tuple[int, int], fmt: str, input_len: int, tokenizer=None) -> list[dict]:
    """Every request of one point, a pure function of the seed and the shape."""
    rng = random.Random(seed)
    if image_count <= 0:
        return [{"prompt": exact_length_text(rng, tokenizer, input_len), "images": [],
                 "sizes": [], "predicted": 0} for _ in range(num_prompts)]
    sizes = draw_resolutions(rng, num_prompts, image_count, lo, hi)
    prompt = IMAGE_PLACEHOLDER * image_count + "\n" + FILLER_WORD * max(1, input_len)
    out: list[dict] = []
    for per in sizes:
        out.append({"prompt": prompt, "images": [encode_image(rng, h, w, fmt) for h, w in per],
                    "sizes": [list(s) for s in per], "predicted": predicted_tokens(per)})
    return out


def warmup_payload(seed: int, image_count: int, lo: tuple[int, int], fmt: str, input_len: int) -> dict:
    """One request at a size just below the drawn range (or a short text), sent before the window."""
    rng = random.Random(seed ^ 0x5EED)
    if image_count <= 0:
        return {"prompt": " ".join(rng.choice(WORDS) for _ in range(16)), "images": []}
    h, w = max(STEP, lo[0] - STEP), max(STEP, lo[1] - STEP)
    return {"prompt": IMAGE_PLACEHOLDER * image_count + "\n" + FILLER_WORD * max(1, input_len),
            "images": [encode_image(rng, h, w, fmt) for _ in range(image_count)]}


def post_stream(opener, url: str, body: dict, timeout: float) -> dict:
    """POST one streaming request; returns first-chunk and end-to-end times plus the last meta_info."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first: float | None = None
    meta: dict = {}
    chunks: list[str] = []
    with opener.open(req, timeout=timeout) as resp:  # noqa: S310 -- fixed localhost URL
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            now = time.perf_counter()
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            info = chunk.get("meta_info") if isinstance(chunk, dict) else None
            info = info if isinstance(info, dict) else {}
            done = info.get("completion_tokens")
            if first is None and (done is None or int(done) >= 1):
                first = now
            if isinstance(chunk, dict) and chunk.get("text"):
                chunks.append(str(chunk["text"]))
            meta = info or meta
    end = time.perf_counter()
    return {"ttft_s": (first - t0) if first is not None else None, "e2e_s": end - t0,
            "meta": meta, "text": "".join(chunks)}


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def _stats(values_ms: list[float]) -> dict:
    if not values_ms:
        return {"mean": None, "median": None, "std": None, "p90": None, "p95": None, "p99": None}
    return {"mean": statistics.fmean(values_ms), "median": statistics.median(values_ms),
            "std": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
            "p90": _pct(values_ms, 90), "p95": _pct(values_ms, 95), "p99": _pct(values_ms, 99)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--tokenizer", default="",
                    help="model path; read for text-only points so every prompt has the exact token count")
    ap.add_argument("--input-len", type=int, default=96)
    ap.add_argument("--output-len", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=120)
    ap.add_argument("--image-count", type=int, default=2)
    ap.add_argument("--image-format", default="jpeg")
    ap.add_argument("--image-min-hw", default="512x512")
    ap.add_argument("--image-max-hw", default="768x1024")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--request-timeout-sec", type=float, default=300.0)
    ap.add_argument("--time-budget-sec", type=float, default=0.0,
                    help="stop issuing requests once the window has run this long (0 = no limit)")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    lo, hi = parse_hw(args.image_min_hw), parse_hw(args.image_max_hw)
    tokenizer = None
    if args.image_count <= 0 and args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    payloads = build_payloads(args.seed, args.num_prompts, args.image_count, lo, hi,
                              args.image_format, args.input_len, tokenizer)
    predicted = [p["predicted"] for p in payloads]
    print(f"[ttft_client] {len(payloads)} request(s), {args.image_count} image(s) each, "
          f"{len(set(predicted))} distinct predicted image-token totals "
          f"({min(predicted)}..{max(predicted)}), seed {args.seed}", flush=True)

    sampling = {"max_new_tokens": args.output_len, "temperature": 0, "ignore_eos": True}
    url = args.base_url.rstrip("/") + "/generate"
    # An empty ProxyHandler: a proxied request to 127.0.0.1 answers 502.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def body_for(p: dict) -> dict:
        body = {"text": p["prompt"], "sampling_params": sampling, "stream": True}
        if p["images"]:
            body["image_data"] = p["images"]
        return body

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = out_path.with_suffix(".t0")
    sentinel.unlink(missing_ok=True)

    warm = warmup_payload(args.seed, args.image_count, lo, args.image_format, args.input_len)
    t_warm = time.perf_counter()
    post_stream(opener, url, body_for(warm), args.request_timeout_sec)
    warmup_ms = (time.perf_counter() - t_warm) * 1000.0
    # Marks the moment the timed window opens.
    sentinel.write_text(f"{time.time():.6f}\n", encoding="utf-8")

    ttfts: list[float] = []
    e2els: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    errors: list[str] = []
    completed = failed = 0
    window_open = time.time()
    t0 = time.perf_counter()
    for index, p in enumerate(payloads):
        if args.time_budget_sec and (time.perf_counter() - t0) > args.time_budget_sec:
            errors.append(f"request {index}: not sent, time budget {args.time_budget_sec:.0f}s exhausted")
            failed += 1
            continue
        try:
            r = post_stream(opener, url, body_for(p), args.request_timeout_sec)
        except Exception as exc:  # noqa: BLE001 -- a failed request is counted, never fatal
            errors.append(f"request {index}: {type(exc).__name__}: {str(exc)[:200]}")
            failed += 1
            continue
        if r["ttft_s"] is None:
            errors.append(f"request {index}: no streamed chunk before [DONE]")
            failed += 1
            continue
        completed += 1
        ttfts.append(r["ttft_s"] * 1000.0)
        e2els.append(r["e2e_s"] * 1000.0)
        meta = r["meta"]
        prompt_tokens.append(int(meta.get("prompt_tokens") or 0))
        completion_tokens.append(int(meta.get("completion_tokens") or 0))
    duration = time.perf_counter() - t0

    ttft = _stats(ttfts)
    e2el = _stats(e2els)
    total_in, total_out = sum(prompt_tokens), sum(completion_tokens)
    tpots = [(e - t) / max(1, c - 1) for e, t, c in zip(e2els, ttfts, completion_tokens) if c > 1]
    record = {
        "backend": "sglang",
        "dataset_name": "seeded_noise_images" if args.image_count > 0 else "seeded_text",
        "completed": completed,
        "failed": failed,
        "num_prompts": args.num_prompts,
        "duration": duration,
        "request_throughput": completed / duration if duration else 0.0,
        "output_throughput": total_out / duration if duration else 0.0,
        "total_token_throughput": (total_in + total_out) / duration if duration else 0.0,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "mean_ttft_ms": ttft["mean"], "median_ttft_ms": ttft["median"], "std_ttft_ms": ttft["std"],
        "p90_ttft_ms": ttft["p90"], "p95_ttft_ms": ttft["p95"], "p99_ttft_ms": ttft["p99"],
        "mean_e2el_ms": e2el["mean"], "median_e2el_ms": e2el["median"], "std_e2el_ms": e2el["std"],
        "p99_e2el_ms": e2el["p99"],
        "mean_tpot_ms": statistics.fmean(tpots) if tpots else None,
        "ttfts": ttfts,
        "e2els": e2els,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "distinct_prompt_token_counts": len(set(prompt_tokens)),
        "min_prompt_tokens": min(prompt_tokens) if prompt_tokens else 0,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "image_sizes": [p["sizes"] for p in payloads],
        "predicted_image_tokens": predicted,
        "distinct_predicted_image_tokens": len(set(predicted)),
        "seed": args.seed,
        "image_count": args.image_count,
        "image_format": args.image_format,
        "image_min_hw": args.image_min_hw,
        "image_max_hw": args.image_max_hw,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": 1,
        "sampling_params": sampling,
        "stream": True,
        "tokenizer_used": tokenizer is not None,
        "warmup_ms": warmup_ms,
        "window_opened_at": window_open,
        "errors": errors[:20],
    }
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mean = ttft["mean"]
    print(f"[ttft_client] {completed}/{args.num_prompts} requests in {duration:.1f}s; mean TTFT "
          f"{mean:.1f} ms" if mean is not None else
          f"[ttft_client] {completed}/{args.num_prompts} requests in {duration:.1f}s; no TTFT", flush=True)
    print(f"[ttft_client] prompt tokens {record['min_prompt_tokens']}..{record['max_prompt_tokens']} in "
          f"{record['distinct_prompt_token_counts']} distinct counts; failed {failed}", flush=True)
    for err in errors[:5]:
        print(f"[ttft_client] {err}", file=sys.stderr, flush=True)
    return 0 if (completed == args.num_prompts and failed == 0) else 3


if __name__ == "__main__":
    raise SystemExit(main())
