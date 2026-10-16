#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The SCORED streaming client. Consumes like a real one, and reports how much like one.

WHY THIS FILE EXISTS. Against the same server in the same boot, aiperf reads a markedly higher
per-token latency than a minimal asyncio client does, and that gap IS the defect under
measurement. The client's per-token cost is part of the mechanism: with the coarse barrier the
consumer's drain rate cannot overlap the forward, so a consumer that does no work per token
measures nothing.

A harness whose sensitivity depends on how slow it accidentally is would be a bad harness. So the
per-token cost here is EXPLICIT and REPORTED rather than emergent:

  * `--detokenize` does the work any real client must do -- decode the streamed delta and count its
    tokens with the model's own tokenizer. This is the realistic component and it is on by default.
  * `--per-chunk-work-us N` adds a deterministic, calibrated busy-loop on top, for pinning the
    consumer profile to a target when the tokenizer alone lands short. It is a knob with a number,
    not an accident.
  * Either way the run REPORTS its own consumer cost (`client_overhead_us_per_token`), measured as
    the wall time spent inside the per-chunk handler divided by tokens. Two runs whose ratios differ
    can be checked against their consumer costs before anything is concluded about the engine.

The metric is generated tokens / wall and mean inter-token latency, the same pair aiperf reports
under `output_token_throughput` and `inter_token_latency`.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path


def build_prompts(rng: random.Random, input_len: int, n: int, tokenizer) -> list[str]:
    """Prompts whose TOKEN length is input_len, not whose character count is.

    Measured rather than estimated: the earlier version emitted `input_len // 2` space-separated
    integers and hoped, which is a text property standing in for a token property.
    """
    out = []
    for _ in range(n):
        toks = [rng.randrange(1000, 30000) for _ in range(input_len)]
        if tokenizer is not None:
            text = tokenizer.decode(toks)
            for _ in range(8):
                got = len(tokenizer.encode(text))
                if got == input_len:
                    break
                if got > input_len:
                    text = text[: int(len(text) * input_len / got)]
                else:
                    text = text + " " + tokenizer.decode(
                        [rng.randrange(1000, 30000) for _ in range(input_len - got)])
        else:
            text = " ".join(str(t) for t in toks)
        out.append(text)
    return out


def _busy(us: float) -> None:
    if us <= 0:
        return
    end = time.perf_counter() + us / 1e6
    h = hashlib.sha256(b"x")
    while time.perf_counter() < end:
        h = hashlib.sha256(h.digest())


async def one_request(session, url, prompt, out_len, sem, tokenizer, work_us, stats, extra=None):
    body = {"model": "default", "prompt": prompt, "max_tokens": out_len,
            "temperature": 0.0, "stream": True, "ignore_eos": True}
    if extra:
        body.update(extra)

    async with sem:
        chunks: list[str] = []
        arrivals: list[float] = []
        overhead = 0.0
        ntok = 0
        t0 = time.perf_counter()
        async with session.post(url, json=body) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                h0 = time.perf_counter()
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                txt = (d.get("choices") or [{}])[0].get("text", "")
                if txt:
                    ntok += len(tokenizer.encode(txt)) if tokenizer is not None else 1
                    chunks.append(txt)
                    arrivals.append(h0)
                    _busy(work_us)
                overhead += time.perf_counter() - h0
        stats.append({"text": "".join(chunks), "ntok": ntok, "arrivals": arrivals,
                      "overhead": overhead, "latency": time.perf_counter() - t0})


async def drive(base, prompts, out_len, conc, tokenizer, work_us):
    import aiohttp
    url = f"{base}/v1/completions"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3000)) as session:
        sem = asyncio.Semaphore(conc)
        warm: list[dict] = []
        await one_request(session, url, prompts[0][:400], 4, sem, tokenizer, work_us, warm)
        stats: list[dict] = []
        t0 = time.perf_counter()
        await asyncio.gather(*[one_request(session, url, p, out_len, sem, tokenizer, work_us, stats)
                               for p in prompts])
        return time.perf_counter() - t0, stats


async def floor_probe(base, prompt, tokenizer, floor: int, cap: int):
    """Is the min-new-tokens floor still held AND still released? Three requests, one of them a control.

    WHY NOT AN EQUIVALENCE CHECK. The scored requests send `ignore_eos: True`, so the penalizer never
    PREPARES and none of the three files the repair touches executes while `output_equivalent` is
    decided. Making a scored probe ask for a floor does not fix it: the repair swaps an additive
    penalty over the whole logits row for a targeted mask, so exercising the path makes the arms
    differ BY THE REPAIR -- measured, both the broken arm and the donor oracle failed equivalence
    when the path was exercised that way.

    TWO EARLIER DESIGNS WERE FALSIFIED BY CALIBRATION, in the same direction each time: both honest
    arms failed, which makes a gate worthless rather than merely weak.

      design 1  `ignore_eos: False` and wait for EOS                 -> both arms ran to the cap.
                          These prompts are random token ids; that stream has no natural ending, so
                          EOS never arrives and the floor is never the binding constraint.
      design 2  one stop token, taken from a 16-token scout          -> both arms ran to the cap.
                          Masking a token makes the greedy stream DIVERGE at that position, so a
                          token seen once at scout position 4 need never reappear in the 224 tokens
                          after the floor lifts.

    THIS DESIGN. Scout 256 tokens and take the most frequent DISTINCT ids from that continuation --
    a set survives the divergence where a single token does not. Then:

        A  stop_token_ids=S, no floor       the CONTROL the first two designs lacked. It must stop
                                            EARLY. If it does not, `stop_token_ids` is doing nothing
                                            on this path and the probe says so instead of blaming the
                                            candidate.
        B  stop_token_ids=S, min_tokens=F   floor <= n < cap   enforced and released
                                            n == cap           the counter never advances
                                            n < floor          the floor is not enforced at all

    Without A, "ran to the cap" is unattributable -- which is precisely what the two calibrations
    produced.

    CALIBRATED, THEN SCORED. In calibration both honest arms read, identically:
        control (stop tokens, no floor) ->  1 token      the stop set works
        probe   (stop tokens, floor 32) -> 45 tokens     the floor held, then released
    a 44-token separation. The verifier fails only `never_released` and `floor_not_enforced` -- the
    two shapes that MEAN the floor is broken. Everything else this can return is inconclusive and
    does not zero a run, because a probe that cannot decide must not.
    """
    import aiohttp
    from collections import Counter
    url = f"{base}/v1/completions"
    out: dict = {"min_tokens_requested": floor, "max_tokens_cap": cap}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        sem = asyncio.Semaphore(1)
        scout: list[dict] = []
        await one_request(session, url, prompt, cap, sem, tokenizer, 0.0, scout,
                          extra={"ignore_eos": True})
        text = scout[0]["text"] if scout else ""
        ids = tokenizer.encode(text, add_special_tokens=False) if (tokenizer and text) else []
        out["scout_tokens"] = len(ids)
        if len(ids) < 64:
            out["classification"] = "scout_too_short"
            out["error"] = f"the scout produced {len(ids)} tokens; no stop set can be chosen from that"
            return out
        stop_ids = [int(t) for t, _ in Counter(ids).most_common(20)]
        out["stop_token_ids"] = stop_ids

        control: list[dict] = []
        await one_request(session, url, prompt, cap, sem, tokenizer, 0.0, control,
                          extra={"ignore_eos": False, "stop_token_ids": stop_ids,
                                 "no_stop_trim": True})
        control_n = control[0]["ntok"] if control else None
        out["control_generated"] = control_n
        if control_n is None or control_n >= floor:
            out["classification"] = "stop_token_not_honoured"
            out["error"] = (f"the control asked for no floor and still produced {control_n} tokens "
                            f"(>= the {floor} floor), so this request shape cannot show a floor at "
                            f"all. Nothing here is a statement about the candidate.")
            return out

        probe: list[dict] = []
        t0 = time.perf_counter()
        await one_request(session, url, prompt, cap, sem, tokenizer, 0.0, probe,
                          extra={"ignore_eos": False, "min_tokens": floor,
                                 "stop_token_ids": stop_ids, "no_stop_trim": True})
        out["elapsed_s"] = time.perf_counter() - t0
    got = probe[0]["ntok"] if probe else None
    out["generated"] = got
    out["reached_the_floor"] = (got is not None and got >= floor)
    out["stopped_before_cap"] = (got is not None and got < cap)
    out["classification"] = (
        "no_output" if got is None else
        "floor_not_enforced" if got < floor else
        "never_released" if got >= cap else
        "floor_enforced_and_released")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--tokenizer", default="", help="hub id or local path; '' disables detokenize")
    ap.add_argument("--input-len", type=int, default=1000)
    ap.add_argument("--output-len", type=int, default=1000)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--per-chunk-work-us", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--floor-probe-out", default="",
                    help="also run the min-new-tokens floor probe and write its record here")
    ap.add_argument("--floor-min-tokens", type=int, default=32)
    ap.add_argument("--floor-max-tokens", type=int, default=256)
    args = ap.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rng = random.Random(args.seed)
    prompts = build_prompts(rng, args.input_len, args.num_prompts, tokenizer)
    elapsed, stats = asyncio.run(
        drive(args.base_url, prompts, args.output_len, args.concurrency, tokenizer,
              args.per_chunk_work_us))

    total = sum(s["ntok"] for s in stats)
    tps = total / elapsed if elapsed else 0.0
    itls: list[float] = []
    for s in stats:
        a = s["arrivals"]
        itls += [(a[i] - a[i - 1]) * 1000.0 for i in range(1, len(a))]
    overhead_us = (sum(s["overhead"] for s in stats) / total * 1e6) if total else 0.0
    texts = [s["text"] for s in stats]
    out = {
        "elapsed_time": elapsed,
        "num_requests": len(stats),
        "output_tokens": total,
        "output_tokens_per_second": tps,
        "output_throughput": tps,
        "inter_token_latency_ms": statistics.mean(itls) if itls else None,
        "inter_token_latency_p50_ms": statistics.median(itls) if itls else None,
        "mean_request_latency_s": statistics.mean([s["latency"] for s in stats]) if stats else None,
        "client_overhead_us_per_token": overhead_us,
        "detokenize": bool(tokenizer),
        "per_chunk_work_us": args.per_chunk_work_us,
        "concurrency": args.concurrency,
        "output_len_requested": args.output_len,
        "min_completion_tokens": min((s["ntok"] for s in stats), default=0),
        "output_digest": hashlib.sha256("\x00".join(texts).encode("utf-8", "replace")).hexdigest(),
    }
    p = Path(args.out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    if args.floor_probe_out:
        try:
            record = asyncio.run(floor_probe(args.base_url, prompts[0][:400], tokenizer,
                                             args.floor_min_tokens, args.floor_max_tokens))
        except Exception as exc:
            record = {"error": f"{type(exc).__name__}: {exc}", "verdict_is_advisory": True}
        fp = Path(args.floor_probe_out)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[stream2] floor probe: {json.dumps(record)}", flush=True)
    print(f"[stream2] {len(stats)} req, {total} tok over {elapsed:.1f}s -> {tps:.2f} tok/s", flush=True)
    print(f"[stream2] ITL {out['inter_token_latency_ms']:.3f} ms   "
          f"consumer cost {overhead_us:.1f} us/token   detok={bool(tokenizer)}", flush=True)
    if out["min_completion_tokens"] < 0.95 * args.output_len:
        print(f"[stream2] WARNING shortest generation {out['min_completion_tokens']} vs "
              f"{args.output_len}", file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
