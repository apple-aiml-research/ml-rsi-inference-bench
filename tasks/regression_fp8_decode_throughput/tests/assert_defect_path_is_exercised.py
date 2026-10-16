#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Did the scored run actually enter the defect's code path?

This is a GATE, not a report. The defect is a compile-time specialisation choice, so a configuration
that never reaches it produces a clean-looking ratio of 1.00 with no error anywhere -- the
`exercised: false` failure mode, which is the cheapest way for a regression task to be silently
worthless. Three preconditions have to hold simultaneously, and each one has been observed failing:

1. **fp8 KV cache.** The op is reached only from
   `if fp8_attention and self.impl.supports_quant_query_input:`, where
   `fp8_attention = self.kv_cache_dtype.startswith("fp8")`. Here it is not set by a flag at all --
   it comes from the checkpoint's `hf_quant_config` (`kv_cache_quant_algo: FP8`). Swap in a
   checkpoint without that key and the whole task goes inert while still serving perfectly.

2. **The compiled path, not the hand-written kernel.** `dispatch_forward` returns the compiled
   `forward_native` only when the op is NOT enabled, which under Inductor means
   `custom_ops: ['none']`. With `custom_ops: ['all']`, or `enforce_eager`, the op takes
   `forward_cuda` and the oracle patch is inert.

3. **Enough decode rows per rank.** The saving goes as (per-rank decode batch x heads per rank) and
   is flat below roughly 19,500 of that product.
   Tensor parallelism divides heads per rank; data parallelism replicates them.

It reads the engine's own resolved configuration out of the run's logs rather than re-deriving it,
and a missing log is a FAILURE rather than a skip: an advisory probe that cannot find its input is
indistinguishable from one that passed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# Below this product of (per-rank decode rows x heads per rank) the fixed and broken kernels are
# both at their launch floor and the fix saves nothing. 19,500 is where the measured curves leave
# that floor; the scored workload sits far above it and this is the guard against a future edit
# quietly dropping concurrency or switching to TP.
MIN_ROWS_TIMES_HEADS = 19_500


def _read(paths: list[pathlib.Path]) -> str:
    return "\n".join(p.read_text(errors="replace") for p in paths if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="engine/bench logs to read the resolved configuration out of")
    ap.add_argument("--model-config", default="",
                    help="the served checkpoint's config.json; num_attention_heads is not in the "
                         "engine log, and it is half of precondition 3")
    ap.add_argument("--out", default="", help="where to write the verdict json")
    a = ap.parse_args()

    found = [pathlib.Path(p) for p in a.logs if pathlib.Path(p).is_file()]
    verdict: dict[str, object] = {"exercised": False, "reasons": []}
    if not found:
        verdict["reasons"].append(
            f"none of the logs exist: {a.logs}. Refusing to report 'exercised' from no evidence."
        )
        _emit(verdict, a.out)
        return 2

    text = _read(found)
    verdict["logs_read"] = [str(p) for p in found]

    # 1. fp8 KV cache
    m = re.search(r"kv_cache_dtype=([^\s,)]+)", text)
    kv = m.group(1).strip("'\"") if m else None
    verdict["kv_cache_dtype"] = kv
    if not (kv or "").startswith("fp8"):
        verdict["reasons"].append(
            f"kv_cache_dtype={kv!r} does not start with 'fp8', so the decode concat+quant op is "
            f"never called and the patched code cannot run."
        )

    # 2. compiled forward_native rather than forward_cuda
    c = re.findall(r"'custom_ops': \[([^\]]*)\]", text)
    resolved = next((x for x in c if x.strip()), "")
    verdict["custom_ops"] = resolved or None
    if "'none'" not in resolved:
        verdict["reasons"].append(
            f"custom_ops resolved to [{resolved}] rather than ['none'], so the op dispatches to its "
            f"hand-written kernel and the fix is inert."
        )

    # 3. per-rank decode rows x heads per rank
    dp = _int(text, r"data_parallel_size=(\d+)") or _int(text, r"'data_parallel_size': (\d+)") or 1
    tp = _int(text, r"tensor_parallel_size=(\d+)") or 1
    heads = _heads_from_config(a.model_config) or _int(text, r"num_attention_heads['\"]?[=: ]+(\d+)")
    # The engine logs this as a dict entry (`'max_num_seqs': 2048`) in its non-default args, and as
    # `max_num_seqs=2048` in the config dump; accept either rather than depending on which line
    # happened to survive truncation.
    seqs = _int(text, r"'max_num_seqs':\s*(\d+)") or _int(text, r"max_num_seqs=(\d+)")
    verdict |= {"data_parallel_size": dp, "tensor_parallel_size": tp,
                "num_attention_heads": heads, "max_num_seqs": seqs}
    if heads and seqs:
        # DP replicates heads onto every rank; TP divides them.
        heads_per_rank = heads // max(tp, 1)
        rows_per_rank = seqs  # max_num_seqs is already the per-rank cap
        product = heads_per_rank * rows_per_rank
        verdict |= {"heads_per_rank": heads_per_rank, "rows_x_heads": product}
        if product < MIN_ROWS_TIMES_HEADS:
            verdict["reasons"].append(
                f"rows_per_rank({rows_per_rank}) x heads_per_rank({heads_per_rank}) = {product}, "
                f"below {MIN_ROWS_TIMES_HEADS}: both kernels sit at their launch floor here and the "
                f"fix saves nothing measurable."
            )
        # 4. Attention must NOT be sharded. The product check above is a floor on the KERNEL being off
        # its launch plateau, and TP=8 with a large batch can clear it -- at 16 heads x 2048 rows the
        # whether THIS TASK'S GATE MEANS ANYTHING: 1.20 was derived end to end with all 128 heads on
        # every rank, and sharding attention changes the op's share of the step without changing
        # anything the other three checks look at. So this is a contract check, not a proxy.
        if heads_per_rank != heads:
            verdict["reasons"].append(
                f"attention is sharded: {heads} heads split across tensor_parallel_size={tp} leaves "
                f"{heads_per_rank} per rank. This task's gate was measured with data parallelism, "
                f"which replicates all {heads} onto every rank; under TP the measured ratio does not "
                f"transfer even though the op still runs."
            )
    else:
        verdict["reasons"].append(
            "could not read num_attention_heads and max_num_seqs out of the logs, so the batch x "
            "heads precondition is unverified."
        )

    verdict["exercised"] = not verdict["reasons"]
    _emit(verdict, a.out)
    return 0 if verdict["exercised"] else 4


def _heads_from_config(path: str) -> int | None:
    if not path:
        return None
    f = pathlib.Path(path)
    if not f.is_file():
        return None
    try:
        cfg = json.loads(f.read_text())
    except Exception:
        return None
    inner = cfg.get("text_config", cfg)
    v = inner.get("num_attention_heads", cfg.get("num_attention_heads"))
    return int(v) if v is not None else None


def _int(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _emit(verdict: dict, out: str) -> None:
    print(json.dumps(verdict, indent=2))
    if out:
        pathlib.Path(out).write_text(json.dumps(verdict, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
