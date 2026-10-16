#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Offline prefill-dominated generation throughput for ONE engine tree. Subprocess only."""
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

_CORRECTNESS_N = 16
_CORRECTNESS_TIMEOUT_S = 180


def _prompt_logprobs(out) -> list[float | None]:
    """Per-position logprob over the PROMPT, from `meta_info.input_token_logprobs`."""
    raw = ((out or {}).get("meta_info") or {}).get("input_token_logprobs") or []
    seq: list[float | None] = []
    for entry in raw:
        if entry is None:
            seq.append(None)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
            seq.append(float(entry[0]) if entry[0] is not None else None)
        else:
            try:
                seq.append(float(entry))
            except (TypeError, ValueError):
                seq.append(None)
    return seq


IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

_FILLER_WORD = "context "


def _parse_hw(value: str) -> tuple[int, int]:
    """`HEIGHTxWIDTH` -> (h, w). Rejects anything else loudly; a silent default here would be a
    different workload wearing the same name."""
    try:
        h_s, w_s = value.lower().split("x", 1)
        h, w = int(h_s), int(w_s)
    except Exception as exc:  # noqa: BLE001 -- the message matters more than the class
        raise SystemExit(f"--image-min-hw/--image-max-hw want HEIGHTxWIDTH, got {value!r}") from exc
    if h <= 0 or w <= 0:
        raise SystemExit(f"non-positive image size: {value!r}")
    return h, w


def _draw_resolutions(rng: random.Random, n_requests: int, per_request: int,
                      lo: tuple[int, int], hi: tuple[int, int],
                      step: int = 32) -> list[list[tuple[int, int]]]:
    """Seeded image sizes, quantised to `step` pixels, with DISTINCT per-request totals."""
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
    """A base64 data URI holding a JPEG of random noise at (h, w)."""
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
    """Point the run's scratch roots at a per-run directory, wipe them, prove they are empty."""
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
    """Fail the run if it did not write its state where it was pointed."""
    entries = sum(1 for _ in root.rglob("*"))
    if entries < _MIN_SCRATCH_ENTRIES:
        raise SystemExit(
            f"scratch root {root} holds {entries} entries after the run, want at least "
            f"{_MIN_SCRATCH_ENTRIES}. The run wrote its state somewhere else, so it inherited "
            "state from an earlier run and its timing is not comparable to another run's."
        )


def _install_probe(probe_json: Path, sentinel: Path) -> Path:
    """Write a sitecustomize that counts l2norm_fwd inside the SPAWNED scheduler process."""
    probe_dir = probe_json.parent / (probe_json.stem + ".probe")
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "sitecustomize.py").write_text(
        "import atexit, importlib.abc, importlib.util, json, os, sys\n"
        f"_OUT = {str(probe_json)!r}\n"
        f"_SENTINEL = {str(sentinel)!r}\n"
        "_TARGET = 'sglang.kernels.ops.attention.fla.l2norm'\n"
        "_S = {'calls': 0, 'distinct_T': set(), 'first_sight_in_window': 0,\n"
        "      'callers': set(), 'scored': False}\n"
        "\n"
        "def _dump():\n"
        "    try:\n"
        "        d = {'calls': _S['calls'], 'distinct_T': len(_S['distinct_T']),\n"
        "             'first_sight_in_window': _S['first_sight_in_window'],\n"
        "             'min_T': min(_S['distinct_T']) if _S['distinct_T'] else None,\n"
        "             'max_T': max(_S['distinct_T']) if _S['distinct_T'] else None,\n"
        "             'callers': sorted(_S['callers'])[:8], 'pid': os.getpid()}\n"
        "        with open(_OUT + '.' + str(os.getpid()), 'w') as fh:\n"
        "            json.dump(d, fh)\n"
        "    except Exception:\n"
        "        pass\n"
        "atexit.register(_dump)\n"
        "\n"
        "def _patch(mod):\n"
        "    real = getattr(mod, 'l2norm_fwd', None)\n"
        "    if real is None:\n"
        "        return\n"
        "    seen = _S['distinct_T']\n"
        "    def wrapped(x, *a, **kw):\n"
        "        t = x.shape[0] if x.ndim == 2 else x.numel() // x.shape[-1]\n"
        "        _S['calls'] += 1\n"
        "        if t not in seen:\n"
        "            seen.add(t)\n"
        "            if not _S['scored']:\n"
        "                _S['scored'] = os.path.exists(_SENTINEL)\n"
        "            if _S['scored']:\n"
        "                _S['first_sight_in_window'] += 1\n"
        "            if _S['calls'] <= 4:\n"
        "                try:\n"
        "                    f = sys._getframe(1)\n"
        "                    _S['callers'].add(f.f_globals.get('__name__', '?'))\n"
        "                except Exception:\n"
        "                    pass\n"
        "            # ON EVERY NEW T, not on a call-count multiple. The scheduler is a spawned\n"
        "            # process and does not always exit cleanly, so atexit is not guaranteed --\n"
        "            # measured: three runs in one container reported calls=4000 distinct_T=84\n"
        "            # against a clean run's 14000/175, i.e. the last 2000-multiple dump rather\n"
        "            # than a different regime. The gates read distinct_T and\n"
        "            # first_sight_in_window, so those two must be durable the moment they move.\n"
        "            # There are a few hundred such events, not 14000.\n"
        "            _dump()\n"
        "        elif _S['calls'] % 500 == 0:\n"
        "            # `calls` moves without a new T, and it is reported for triage, so it gets a\n"
        "            # periodic dump too. 500 rather than 2000: ~28 small JSON writes per run\n"
        "            # against 14000 calls, which bounds the undercount at ~3.5% if the process is\n"
        "            # killed. The GATED quantities do not depend on this branch.\n"
        "            _dump()\n"
        "        return real(x, *a, **kw)\n"
        "    mod.l2norm_fwd = wrapped\n"
        "\n"
        "class _Hook(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname != _TARGET:\n"
        "            return None\n"
        "        sys.meta_path.remove(self)\n"
        "        try:\n"
        "            spec = importlib.util.find_spec(fullname)\n"
        "        finally:\n"
        "            sys.meta_path.insert(0, self)\n"
        "        if spec is None or spec.loader is None:\n"
        "            return None\n"
        "        real_exec = spec.loader.exec_module\n"
        "        def exec_module(module, _r=real_exec):\n"
        "            _r(module)\n"
        "            _patch(module)\n"
        "        spec.loader.exec_module = exec_module\n"
        "        return spec\n"
        "\n"
        "if not any(type(h).__name__ == '_Hook' for h in sys.meta_path):\n"
        "    sys.meta_path.insert(0, _Hook())\n",
        encoding="utf-8",
    )
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{probe_dir}{os.pathsep}{old}" if old else str(probe_dir)
    return probe_dir


def _collect_probe(probe_json: Path) -> dict:
    """Merge the per-pid counter files. The scheduler is a separate process and there may be more
    than one; the maxima are what the gates read."""
    merged = {"calls": 0, "distinct_T": 0, "first_sight_in_window": 0,
              "min_T": None, "max_T": None, "callers": [], "pids": 0}
    for p in sorted(probe_json.parent.glob(probe_json.name + ".*")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 -- a half-written counter must not lose the measurement
            continue
        merged["pids"] += 1
        merged["calls"] += int(d.get("calls") or 0)
        merged["distinct_T"] = max(merged["distinct_T"], int(d.get("distinct_T") or 0))
        merged["first_sight_in_window"] = max(merged["first_sight_in_window"],
                                              int(d.get("first_sight_in_window") or 0))
        for k, fn in (("min_T", min), ("max_T", max)):
            v = d.get(k)
            if v is not None:
                merged[k] = v if merged[k] is None else fn(merged[k], v)
        merged["callers"] = sorted(set(merged["callers"]) | set(d.get("callers") or []))
    return merged


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
    probe_json = Path(args.out_json).resolve().parent / (Path(args.out_json).stem + ".probe.json")
    sentinel = Path(args.out_json).resolve().parent / (Path(args.out_json).stem + ".t0")
    sentinel.unlink(missing_ok=True)
    for stale in probe_json.parent.glob(probe_json.name + ".*"):
        stale.unlink(missing_ok=True)
    _install_probe(probe_json, sentinel)

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

        async def _correctness(engine_, prompt_, args_):
            """Compare per-position prompt logprobs between the arms on IDENTICAL inputs."""
            out = {"comparison": "prompt_token_logprobs", "n": _CORRECTNESS_N}
            order = sorted(range(len(images)), key=lambda i: predicted[i])
            area = [max(h * w for h, w in per) for per in sizes]
            k = max(1, min(_CORRECTNESS_N, len(order)))
            picks = []
            for reserved in (order[0], order[-1],
                             min(range(len(area)), key=lambda i: area[i]),
                             max(range(len(area)), key=lambda i: area[i])):
                if reserved not in picks and len(picks) < k:
                    picks.append(reserved)
            for j in range(k):
                if len(picks) >= k:
                    break
                cand = order[round(j * (len(order) - 1) / max(1, k - 1))]
                if cand not in picks:
                    picks.append(cand)
            picks.sort()
            imgs = [images[i] for i in picks]
            out["picked_predicted_tokens"] = [predicted[i] for i in picks]
            out["picked_sizes"] = [sizes[i] for i in picks]
            sp = {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                  "max_new_tokens": 1, "ignore_eos": True}
            outs = []
            for i in imgs:
                outs.append(await engine_.async_generate(
                    prompt=prompt_, image_data=i, sampling_params=sp,
                    return_logprob=True, logprob_start_len=0))
            out["logprobs"] = [_prompt_logprobs(o) for o in outs]
            out["lengths"] = [len(x) for x in out["logprobs"]]
            out["error"] = None
            return out

        async def drive():
            await engine.async_generate(
                prompt=prompt, image_data=warm_images,
                sampling_params={**sp, "max_new_tokens": 4})
            sentinel.write_text("t0\n", encoding="utf-8")
            t0 = time.perf_counter()
            outs = await asyncio.gather(*[one(i) for i in images])
            measured = time.perf_counter() - t0
            try:
                corr = await asyncio.wait_for(_correctness(engine, prompt, args),
                                              timeout=_CORRECTNESS_TIMEOUT_S)
            except BaseException as exc:  # noqa: BLE001 -- CancelledError is a BaseException
                corr = {"logprobs": None, "error": f"{type(exc).__name__}: {exc}"}
            return measured, outs, corr

        elapsed, outs, correctness = asyncio.run(drive())

    finally:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001 -- a failed shutdown must not lose the measurement
            pass

    counters = _collect_probe(probe_json)
    _assert_scratch_used(cache_dir)
    cubins = len(list(cache_dir.rglob("*.cubin")))
    cache_files = len(list(cache_dir.rglob("*")))

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
        "correctness_check": correctness,
        "regime": {
            "distinct_prompt_token_counts": distinct_prompt_tokens,
            "min_prompt_tokens": min(per_prompt) if per_prompt else 0,
            "max_prompt_tokens": max(per_prompt) if per_prompt else 0,
            "distinct_image_size_tuples": len(set(map(tuple, sizes))),
            "l2norm_calls": counters["calls"],
            "l2norm_distinct_T": counters["distinct_T"],
            "l2norm_first_sight_in_scored_window": counters["first_sight_in_window"],
            "l2norm_min_T": counters["min_T"],
            "l2norm_max_T": counters["max_T"],
            "l2norm_callers": counters["callers"],
            "probe_processes": counters["pids"],
            "triton_cache_files": cache_files,
            "triton_cubins": cubins,
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
    print(f"[bench] REGIME: l2norm calls={counters['calls']} distinct_T={counters['distinct_T']} "
          f"first_sight_in_window={counters['first_sight_in_window']} "
          f"callers={counters['callers']} cubins={cubins} cache_files={cache_files}", flush=True)
    mean_ms = 1000.0 * (sum(lat) / len(lat)) if lat else float("nan")
    print(f"[bench] {output_tps:.1f} generated tok/s ({total_tps:.1f} total tok/s), "
          f"mean request {mean_ms:.1f} ms", flush=True)
    if comp and min(comp) < 0.95 * args.output_len:
        print(f"[bench] WARNING the shortest generation is {min(comp)} token(s) against a "
              f"requested {args.output_len}; the scored metric is per generated token, so a "
              f"truncated run is not comparable.", file=sys.stderr, flush=True)
    if counters["calls"] == 0:
        print("[bench] WARNING the linear-attention l2-norm helper was never called. Either this "
              "model does not route through it or a non-Triton GDN backend was selected; either "
              "way both arms execute the same code and the ratio is meaningless.",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
