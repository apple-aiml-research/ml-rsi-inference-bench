#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Offline multimodal prefill bench for the tree in `repo/`. One fresh engine per call."""

from __future__ import annotations
import argparse
import asyncio
import base64
import io
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(_v, None)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_FILLER_WORD = "context "
def _parse_hw(value: str) -> tuple[int, int]:
    try:
        h_s, w_s = value.lower().split("x", 1)
        h, w = int(h_s), int(w_s)
    except Exception as exc:
        raise SystemExit(f"--image-min-hw/--image-max-hw want HEIGHTxWIDTH, got {value!r}") from exc
    if h <= 0 or w <= 0:
        raise SystemExit(f"non-positive image size: {value!r}")
    return h, w
def _draw_resolutions(rng: random.Random, n_requests: int, per_request: int,
                      lo: tuple[int, int], hi: tuple[int, int],
                      step: int = 32) -> list[list[tuple[int, int]]]:
    def _predict(imgs: list[tuple[int, int]]) -> int:
        return sum((h // step) * (w // step) for h, w in imgs)
    out: list[list[tuple[int, int]]] = []
    taken: set[int] = set()
    attempts_left = 200 * n_requests
    while len(out) < n_requests and attempts_left > 0:
        attempts_left -= 1
        imgs = [(rng.randrange(lo[0], hi[0] + 1, step), rng.randrange(lo[1], hi[1] + 1, step))
                for _ in range(per_request)]
        total = _predict(imgs)
        if total in taken:
            continue
        taken.add(total)
        out.append(imgs)
    while len(out) < n_requests:
        out.append([(rng.randrange(lo[0], hi[0] + 1, step), rng.randrange(lo[1], hi[1] + 1, step))
                    for _ in range(per_request)])
    return out
def _encode_image(rng: random.Random, h: int, w: int, fmt: str) -> str:
    from PIL import Image
    buf = bytearray(rng.getrandbits(8) for _ in range(h * w * 3))
    img = Image.frombytes("RGB", (w, h), bytes(buf))
    sink = io.BytesIO()
    pil_fmt = "JPEG" if fmt.lower() in ("jpeg", "jpg") else fmt.upper()
    img.save(sink, format=pil_fmt, quality=85)
    mime = "jpeg" if pil_fmt == "JPEG" else pil_fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(sink.getvalue()).decode("ascii")
_SCRATCH_ROOTS = ("HOME",)
_MIN_SCRATCH_ENTRIES = 24
def _fresh_scratch(out_json: str) -> Path:
    root = Path(out_json).resolve().parent / (Path(out_json).stem + ".caches")
    shutil.rmtree(root, ignore_errors=True)
    shared = Path(os.environ.get("HOME", "/root")) / ".cache"
    for index, var in enumerate(_SCRATCH_ROOTS):
        d = root / chr(ord("a") + index)
        d.mkdir(parents=True, exist_ok=True)
        left = list(d.rglob("*"))
        if left:
            raise SystemExit(f"scratch dir {d} is not empty after the wipe: {left[:5]}")
        os.environ[var] = str(d)
    if shared.is_dir():
        (root / "a" / ".cache").symlink_to(shared, target_is_directory=True)
    print(f"[bench] SCRATCH ISOLATED for this run: {root}", flush=True)
    return root
def _assert_scratch_used(root: Path) -> None:
    entries = sum(1 for _ in root.rglob("*"))
    if entries < _MIN_SCRATCH_ENTRIES:
        raise SystemExit(
            f"scratch root {root} holds {entries} entries after the run, want at least "
            f"{_MIN_SCRATCH_ENTRIES}. The run wrote its state somewhere else, so it inherited "
            "state from an earlier run and its timing is not comparable to another run's."
        )
def _completion_len(out, fallback: int) -> int:
    meta = (out or {}).get("meta_info", {}) if isinstance(out, dict) else {}
    n = meta.get("completion_tokens")
    return int(n) if isinstance(n, int) else fallback
def _prompt_len(out, fallback: int) -> int:
    meta = (out or {}).get("meta_info", {}) if isinstance(out, dict) else {}
    n = meta.get("prompt_tokens")
    return int(n) if isinstance(n, int) else fallback
def main() -> int:
    ap = argparse.ArgumentParser(description="offline multimodal prefill bench")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=96,
                    help="approximate token length of the CONSTANT text half of each request")
    ap.add_argument("--output-len", type=int, default=4)
    ap.add_argument("--num-prompts", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="max in-flight requests (0 = unbounded)")
    ap.add_argument("--image-count", type=int, default=4)
    ap.add_argument("--image-format", default="jpeg")
    ap.add_argument("--image-min-hw", default="512x512")
    ap.add_argument("--image-max-hw", default="768x1024")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-level", default="warning")
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--cuda-graph-max-bs-decode", type=int, default=4)
    ap.add_argument("--disable-radix-cache", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    cache_dir = _fresh_scratch(args.out_json)
    sentinel = Path(args.out_json).resolve().parent / (Path(args.out_json).stem + ".t0")
    sentinel.unlink(missing_ok=True)
    import sglang as sgl
    resolved = str(Path(sgl.__file__).resolve())
    print(f"[bench] engine package resolved to {resolved}", flush=True)
    kw = dict(
        model_path=args.model,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=True,
        disable_radix_cache=bool(args.disable_radix_cache),
        random_seed=args.seed,
        log_level=args.log_level,
        context_length=args.context_length,
        chunked_prefill_size=args.chunked_prefill_size,
        cuda_graph_max_bs_decode=args.cuda_graph_max_bs_decode,
    )
    print(f"[bench] engine kwargs: {kw}", flush=True)
    rng = random.Random(args.seed)
    sizes = _draw_resolutions(rng, args.num_prompts, args.image_count,
                              _parse_hw(args.image_min_hw), _parse_hw(args.image_max_hw))
    predicted = [sum((h // 32) * (w // 32) for h, w in per) for per in sizes]
    print(f"[bench] SWEPT IMAGE SIZES: {len(sizes)} request(s), "
          f"{len(set(map(tuple, sizes)))} DISTINCT size tuples, "
          f"{len(set(predicted))} distinct predicted vision-token totals "
          f"({min(predicted)}..{max(predicted)})", flush=True)
    images = [[_encode_image(rng, h, w, args.image_format) for h, w in per] for per in sizes]
    lo_h, lo_w = _parse_hw(args.image_min_hw)
    warm_hw = (max(32, lo_h - 32), max(32, lo_w - 32))
    warm_images = [_encode_image(rng, warm_hw[0], warm_hw[1], args.image_format)
                   for _ in range(args.image_count)]
    text = _FILLER_WORD * max(1, args.input_len)
    prompt = IMAGE_PLACEHOLDER * args.image_count + "\n" + text
    engine = sgl.Engine(**kw)
    try:
        top_k = -1 if args.top_k <= 0 else args.top_k
        sp = {"temperature": args.temperature, "top_p": args.top_p, "top_k": top_k,
              "max_new_tokens": args.output_len, "ignore_eos": True}
        sem = asyncio.Semaphore(args.concurrency) if args.concurrency > 0 else None
        lat: list[float] = []
        async def one(imgs):
            async def go():
                t = time.perf_counter()
                out = await engine.async_generate(
                    prompt=prompt, image_data=imgs, sampling_params=sp)
                lat.append(time.perf_counter() - t)
                return out
            if sem is None:
                return await go()
            async with sem:
                return await go()
        async def drive():
            await engine.async_generate(
                prompt=prompt, image_data=warm_images,
                sampling_params={**sp, "max_new_tokens": 4})
            sentinel.write_text("t0\n", encoding="utf-8")
            t0 = time.perf_counter()
            outs = await asyncio.gather(*[one(i) for i in images])
            measured = time.perf_counter() - t0
            corr = None
            return measured, outs, corr
        elapsed, outs, _ = asyncio.run(drive())
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
    counters = {}
    _assert_scratch_used(cache_dir)
    out_toks = sum(_completion_len(o, args.output_len) for o in outs)
    comp = [_completion_len(o, args.output_len) for o in outs]
    per_prompt = [_prompt_len(o, 0) for o in outs]
    prompt_toks = sum(per_prompt)
    distinct_prompt_tokens = len(set(per_prompt))
    output_tps = out_toks / elapsed if elapsed else 0.0
    total_tps = (prompt_toks + out_toks) / elapsed if elapsed else 0.0
    res = {
        "elapsed_time": elapsed,
        "num_requests": len(outs),
        "num_prompts": args.num_prompts,
        "input_tokens": prompt_toks,
        "output_tokens": out_toks,
        "total_num_tokens": prompt_toks + out_toks,
        "output_throughput": output_tps,
        "output_tokens_per_second": output_tps,
        "total_token_throughput": total_tps,
        "tokens_per_second": total_tps,
        "requests_per_second": len(outs) / elapsed if elapsed else 0.0,
        "mean_request_latency_s": (sum(lat) / len(lat)) if lat else None,
        "p99_request_latency_s": (sorted(lat)[min(len(lat) - 1, int(0.99 * len(lat)))]
                                  if lat else None),
        "max_request_latency_s": max(lat) if lat else None,
        "mean_per_output_token_ms": ((sum(lat) / len(lat)) / args.output_len * 1000.0)
                                    if (lat and args.output_len) else None,
        "min_completion_tokens": min(comp) if comp else 0,
        "max_completion_tokens": max(comp) if comp else 0,
        "output_len_requested": args.output_len,
        "concurrency": args.concurrency,
        "regime": {
            "distinct_prompt_token_counts": distinct_prompt_tokens,
            "min_prompt_tokens": min(per_prompt) if per_prompt else 0,
            "max_prompt_tokens": max(per_prompt) if per_prompt else 0,
            "distinct_image_size_tuples": len(set(map(tuple, sizes))),
        },
        "engine_kwargs": {k: v for k, v in kw.items() if k != "model_path"},
        "engine_package": resolved,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[bench] SHAPE: {len(outs)} request(s), {prompt_toks} prompt token(s) in "
          f"{distinct_prompt_tokens} DISTINCT counts "
          f"({min(per_prompt) if per_prompt else 0}..{max(per_prompt) if per_prompt else 0}), "
          f"{out_toks} generated token(s) over {elapsed:.1f}s", flush=True)
    mean_ms = 1000.0 * (sum(lat) / len(lat)) if lat else float("nan")
    print(f"[bench] {output_tps:.1f} generated tok/s ({total_tps:.1f} total tok/s), "
          f"mean request {mean_ms:.1f} ms", flush=True)
    if comp and min(comp) < 0.95 * args.output_len:
        print(f"[bench] WARNING the shortest generation is {min(comp)} token(s) against a "
              f"requested {args.output_len}; the scored metric is per generated token, so a "
              f"truncated run is not comparable.", file=sys.stderr, flush=True)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
