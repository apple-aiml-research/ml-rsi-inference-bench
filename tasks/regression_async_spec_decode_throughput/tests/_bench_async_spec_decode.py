#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Throughput harness for this task, because the stock bench cannot express its preconditions."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SPEC_TOKENS = 3


def _mxfp4_backend_name() -> str:
    """Which MoE backend did the engine ACTUALLY select, asked of the engine rather than inferred."""
    try:
        import inspect

        from vllm.model_executor.layers.quantization.mxfp4 import get_mxfp4_backend
        try:
            n_required = sum(
                1
                for p in inspect.signature(get_mxfp4_backend).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            )
        except (TypeError, ValueError):
            n_required = 0
        backend = get_mxfp4_backend(*([False] * n_required))
        return str(getattr(backend, "name", backend))
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}: {exc}"


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link.resolve())
    raise SystemExit("no model: set INFB_MODEL_PATH or stage <workspace>/model")


def _drafter() -> str:
    """The EAGLE3 drafter directory — a SECOND checkpoint, and the engine will not start without it."""
    direct = os.environ.get("INFB_SPEC_DRAFTER_PATH")
    if direct and Path(direct, "config.json").exists():
        return direct
    for entry in (os.environ.get("INFB_AUX_MODEL_PATHS") or "").split(os.pathsep):
        entry = entry.strip()
        if entry and "eagle3" in Path(entry).name.lower() and Path(entry, "config.json").exists():
            return entry
    baked = "/opt/infb/aux-models/nvidia/gpt-oss-120b-Eagle3"
    if Path(baked, "config.json").exists():
        return baked
    raise SystemExit(
        "no speculative drafter: set INFB_SPEC_DRAFTER_PATH to a directory holding the "
        "nvidia/gpt-oss-120b-Eagle3 checkpoint (config.json + model.safetensors). "
        "environment/Dockerfile bakes it there at build time; if this fires, that build step was "
        "skipped or the image is not this task's."
    )


def _build_llm(model: str, drafter: str, *, max_num_seqs: int, max_model_len: int,
               enforce_eager: bool):
    from vllm import LLM

    extra = {}
    if not enforce_eager:
        extra["compilation_config"] = {
            "max_cudagraph_capture_size": (1 + SPEC_TOKENS) * max_num_seqs,
        }
    return LLM(
        model=model,
        # The drafter publishes no tokenizer of its own, so the target's is used for both.
        tokenizer=model,
        async_scheduling=True,          # precondition 1
        # precondition 2, and this is upstream's contract verbatim: a SEPARATE EAGLE3 checkpoint,
        # not the target's own head. Set on BOTH arms, so enabling speculation is the verifier's
        # action and never the agent's lever -- the agent can only make already-on speculation
        # faster, and cannot raise its score by turning speculation off.
        speculative_config={
            "method": "eagle3",
            "model": drafter,
            "num_speculative_tokens": SPEC_TOKENS,
        },
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
        seed=12345,
        **extra,
    )


def _attention_backend_name() -> str:
    """Which attention backend the engine RESOLVED, asked of it rather than read off the env var."""
    try:
        from vllm import envs as _envs
        asked = getattr(_envs, "VLLM_ATTENTION_BACKEND", None)
    except Exception:  # noqa: BLE001
        asked = None
    resolved = None
    try:
        from vllm.attention.selector import get_attn_backend
        import torch as _torch

        b = get_attn_backend(128, _torch.bfloat16, "auto", 16)
        resolved = getattr(b, "__name__", None) or str(b)
    except Exception as exc:  # noqa: BLE001
        resolved = f"unavailable: {type(exc).__name__}: {exc}"
    return f"asked={asked!r} resolved={resolved}"


def _numeric_fingerprint(llm, model: str) -> dict:  # noqa: ANN001
    """Per-position logprobs on a FIXED token sequence, so the two arms can be compared pointwise."""
    import hashlib

    from vllm import SamplingParams, TokensPrompt

    out: dict = {"available": False}
    try:
        vocab = _vocab_size(model)
        rng = random.Random(20260917)
        ids = [rng.randrange(16, max(32, vocab - 16)) for _ in range(160)]
        out["prompt_sha256"] = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()[:16]
        res = llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
        )
        lps = getattr(res[0], "prompt_logprobs", None)
        if not lps:
            out["reason"] = "engine returned no prompt_logprobs"
            return out
        vals = []
        for pos, entry in enumerate(lps):
            if entry is None:
                continue
            lp = entry.get(ids[pos])
            if lp is None:
                continue
            vals.append(float(getattr(lp, "logprob", lp)))
        if not vals:
            out["reason"] = "no per-token logprob could be read"
            return out
        out.update(
            available=True,
            n_positions=len(vals),
            logprobs=[round(v, 6) for v in vals],
            sum_logprob=round(sum(vals), 6),
        )
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def _runner_class_name(llm) -> str:  # noqa: ANN001
    """Fully-qualified class of the model runner the engine actually built."""
    for chain in (
        ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker",
         "worker", "model_runner"),
        ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker",
         "model_runner"),
    ):
        obj = llm
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            cls = type(obj)
            return f"{cls.__module__}.{cls.__qualname__}"
    return ""


def _sampling(out_len: int, temperature: float, top_p: float, top_k: int,
              *, n: int, jitter: float):
    """One SamplingParams PER REQUEST, so output lengths CAN differ — at `jitter=0` they do not."""
    from vllm import SamplingParams

    rng = random.Random(20260913)
    lo, hi = int(out_len * (1.0 - jitter)), int(out_len * (1.0 + jitter))
    lo = max(8, lo)
    hi = max(lo, hi)
    return [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ignore_eos=True,                # the length is the draw above, not the model's choice
            max_tokens=(out_len if jitter <= 0 else rng.randint(lo, hi)),
            detokenize=False,
        )
        for _ in range(n)
    ]


def _prompts(n: int, in_len: int, vocab: int):
    """Random token-id prompts: no tokenizer round-trip, so the shape is exact."""
    from vllm import TokensPrompt

    rng = random.Random(12345)
    return [
        TokensPrompt(prompt_token_ids=[rng.randrange(16, max(32, vocab - 16)) for _ in range(in_len)])
        for _ in range(n)
    ]


def _vocab_size(model: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return int(getattr(cfg, "vocab_size", 32000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path(os.environ.get("INFB_WORKSPACE", "/app")))
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--output-len-jitter", type=float, default=0.0,
                    help="per-request max_tokens ~ U[L(1-j), L(1+j)]. The scored run uses 0. Only "
                         "the reduced-scale probe needs a non-zero value.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--probe", action="store_true",
                    help="tiny instrumented run that asserts the path under repair executes")
    args = ap.parse_args()

    ws = args.workspace
    model = _model(ws)
    drafter = _drafter()

    if args.probe:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        from vllm.v1.core.sched.scheduler import Scheduler

        fingerprint: dict = {"available": False, "reason": "probe did not reach the equivalence pass"}

        seen = {"decode": 0, "mixed": 0, "uniform_full": 0, "uniform_other": 0}
        shape = {"cur": None}
        orig_sched = Scheduler.schedule

        def counting_schedule(self):  # noqa: ANN001
            out = orig_sched(self)
            widths = sorted(set(out.num_scheduled_tokens.values())) if out.num_scheduled_tokens else []
            if widths and max(widths) <= 1 + SPEC_TOKENS:
                seen["decode"] += 1
                shape["cur"] = "uniform" if len(widths) == 1 else "mixed"
                if shape["cur"] == "mixed":
                    seen["mixed"] += 1
            else:
                shape["cur"] = None
            return out

        Scheduler.schedule = counting_schedule

        dispatcher_hooked = False
        try:
            from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

            orig_dispatch = CudagraphDispatcher.dispatch

            def counting_dispatch(self, *a, **kw):  # noqa: ANN001
                res = orig_dispatch(self, *a, **kw)
                mode = res[0] if isinstance(res, tuple) else res
                if shape["cur"] == "uniform":
                    key = "uniform_full" if getattr(mode, "name", "") == "FULL" else "uniform_other"
                    seen[key] += 1
                return res

            CudagraphDispatcher.dispatch = counting_dispatch
            dispatcher_hooked = True
        except Exception:  # noqa: BLE001
            orig_dispatch = None

        runner_class = ""
        try:
            llm = _build_llm(model, drafter, max_num_seqs=16, max_model_len=192,
                             enforce_eager=False)
            runner_class = _runner_class_name(llm)
            llm.generate(
                _prompts(64, 32, _vocab_size(model)),
                _sampling(48, args.temperature, args.top_p, args.top_k, n=64, jitter=0.3),
            )
            fingerprint = _numeric_fingerprint(llm, model)
        finally:
            Scheduler.schedule = orig_sched
            if dispatcher_hooked and orig_dispatch is not None:
                CudagraphDispatcher.dispatch = orig_dispatch

        path_entered = seen["decode"] > 0
        full_available = (not dispatcher_hooked) or seen["uniform_full"] > 0
        clamp_fired = seen["mixed"] > 0
        ok = path_entered and full_available
        out = {
            "probe": True,
            "mxfp4_backend": _mxfp4_backend_name(),
            "flashinfer_mxfp8_env": os.environ.get("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"),
            "attention_backend": _attention_backend_name(),
            "numeric_fingerprint": fingerprint,
            "attention_backend_env": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "decode_steps": seen["decode"],
            "non_uniform_decode_steps": seen["mixed"],
            "uniform_steps_on_full_cudagraph": seen["uniform_full"],
            "uniform_steps_on_other_cudagraph": seen["uniform_other"],
            "cudagraph_dispatcher_hooked": dispatcher_hooked,
            "model_runner_class": runner_class,
            "clamp_fired": clamp_fired,
            "full_graph_path_available": full_available,
            "exercised": ok,
        }
        args.output_json.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out))
        if not ok:
            print("FATAL: this task would score a no-op as a pass.", file=sys.stderr)
            if not path_entered:
                print("CAUSE: no decode step ran at all, so nothing was measured. Check async "
                      "scheduling, the speculative config and the concurrency.", file=sys.stderr)
            if not full_available:
                print("CAUSE: uniform decode batches never reached the FULL cudagraph, so there is "
                      "no fast path for the clamp to cost. max_cudagraph_capture_size defaults to "
                      "min(max_num_seqs*2, 512) and is not scaled by 1+num_speculative_tokens.",
                      file=sys.stderr)
        return 0 if ok else 1

    llm = _build_llm(
        model,
        drafter,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    prompts = _prompts(args.num_prompts, args.input_len, _vocab_size(model))
    sampling = _sampling(args.output_len, args.temperature, args.top_p, args.top_k,
                         n=args.num_prompts, jitter=args.output_len_jitter)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    produced = sum(len(c.token_ids) for o in outs for c in o.outputs)
    total = args.num_prompts * args.input_len + produced
    result = {
        "elapsed_time": elapsed,
        "num_requests": args.num_prompts,
        "total_num_tokens": total,
        "total_output_tokens": produced,
        "requests_per_second": args.num_prompts / elapsed,
        "tokens_per_second": total / elapsed,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
