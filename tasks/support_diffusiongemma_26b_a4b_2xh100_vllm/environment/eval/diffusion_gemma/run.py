#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The public protocols of the DiffusionGemma task, one command each (see README.md).

    python /app/eval/diffusion_gemma/run.py --help
    python /app/eval/diffusion_gemma/run.py <command> --help

Serving commands talk to a running server (`--base-url`, default http://127.0.0.1:8000) and read the
public inputs under examples/ unless `--inputs` names others; `--log` names the server's log where a
protocol reads it. Component commands load the operator contract (`--impl`, default
/app/diffusion_gemma_attend.py). Every command writes its record to `--output` and prints a summary.
Outputs are measurements; grading supplies its own inputs and acceptance policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
EXAMPLES = HERE / "examples"


def _inputs(args, default: str):
    return json.loads(Path(args.inputs or EXAMPLES / default).read_text())


def _prose(n: int) -> list[int]:
    """n public prose tokens (the efficiency sample's documents, tokenized and joined)."""
    import lzma

    from eval.diffusion_gemma.serving.common import tokenizer

    out: list[int] = []
    for line in lzma.open("/public/efficiency/sample.jsonl.xz", "rt"):
        out += tokenizer()(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
        if len(out) >= n:
            return out[:n]
    while len(out) < n:
        out += out
    return out[:n]


def cmd(args) -> dict:
    from eval.diffusion_gemma.serving import (accuracy, api, branches, chat, configuration, long_context, mixed,
                                              policy, preemption, reads, recipes, repairs, replay, semantic, tools)

    c = args.command
    if c == "gencfg":
        return recipes.write_gencfg(args.out)
    if c.startswith("attend-"):
        import torch

        from eval.diffusion_gemma.attend import check, measure

        impl = check.load_impl(args.impl)
        if c == "attend-check":
            cases = _inputs(args, "attend_cases.json")["cases"]
            if args.example:
                cases = [x for x in cases if x["id"] == args.example]
            return {"records": [check.run_case(x, impl, torch.device("cuda")) for x in cases]}
        fam = [args.family] if args.family else list(measure.CONTEXTS)
        rec = {f"{f}/{L}": {lk: measure.measure(impl, f, L, lk) for lk in measure.LAYERS} for f in fam for L in measure.CONTEXTS[f]}
        if args.identity:  # the identity sets `binding` looks for in the served model
            from eval.diffusion_gemma.attend import binding

            calls = {(f, lk): check.Inputs(measure.case(f, 4096, lk), impl, torch.device("cuda")).call
                     for f in ("encode", "denoise", "mixed") for lk in measure.LAYERS}
            Path(args.identity).write_text(json.dumps({**binding.identity_sets(calls), "exclusions": binding.exclusions()}, indent=1))
        return rec
    from eval.diffusion_gemma.serving.server import Client

    srv = Client(args.base_url, args.log)
    if c == "configuration":
        return configuration.served_configuration(srv, args.deployment)
    if c == "api":
        rec = api.api_probes(srv)
        rec["repairs"] = {"structured_outputs": repairs.structured_outputs(srv), "long_output": repairs.long_output(srv, _prose(1000)),
                          "concurrent_logprobs": repairs.concurrent_logprobs(srv),
                          "first_token": repairs.first_token(srv, _inputs(args, "chat.json")["first_token"])}
        return rec
    if c == "chat":
        return chat.chat_reasoning(srv, _inputs(args, "chat.json")["reasoning"])
    if c == "tools":
        return tools.tool_calls(srv, _inputs(args, "chat.json")["tools"])
    if c in ("reads", "step1"):
        data = _inputs(args, "reads.json")
        sel = [r for r in data["reads"] if (r.get("steps", 1) == 2) == (c == "step1") or args.all]
        return {"reads": reads.run(srv, data, sel)}
    if c == "replay":
        return replay.replay(srv, _inputs(args, "replay.json")["cases"])
    if c == "prefix-branch":
        spec = _inputs(args, "prefix.json")
        return branches.branches(srv, spec["cases"], block_size=int(spec.get("block_size", 16)))
    if c == "mixed":
        data = _inputs(args, "reads.json")
        return mixed.mixed(srv, data, [r for r in data["reads"] if r.get("family") != "image"],
                           [_prose(4096)[k:] + _prose(4096)[:k] for k in (0, 1000, 2000)])
    if c == "preemption":
        return preemption.preemption_resume(srv, [_prose(args.prompt_len)] * args.requests, args.max_tokens)
    if c == "semantic":
        return semantic.semantic_majority(srv, _inputs(args, "chat.json")["semantic"])
    if c == "policy":
        spec = _inputs(args, "chat.json")["policy"]
        w, n = policy.walls(srv, spec["adaptive"])
        return {"schedule": policy.schedule(srv, spec["schedule"]), "walls": w, "completion_tokens": n}
    if c == "long-context":
        return long_context.near_limit(srv, _prose(long_context.NEAR_LIMIT_PROMPT))
    if c == "accuracy":
        return accuracy.gsm8k(srv, limit=args.limit)
    if c == "binding":
        from eval.diffusion_gemma.serving import binding

        ident = json.loads(Path(args.identity).read_text())
        data = _inputs(args, "reads.json")
        rd = next(r for r in data["reads"] if r.get("family") != "image")
        body = reads.text_read(data["prompts"][rd["prompt"]]["ids"], rd["seed"], rd["targets"], 1)
        return binding.witness(srv, Path(args.trace_dir), ident, (_prose(6144), [_prose(2000)] * 3), body, args.ranks)
    raise SystemExit(f"unknown command {c}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    serving = ["configuration", "api", "chat", "tools", "reads", "step1", "replay", "prefix-branch", "mixed", "preemption",
               "semantic", "policy", "long-context", "accuracy", "binding"]
    for name in serving + ["attend-check", "attend-measure", "gencfg"]:
        p = sub.add_parser(name)
        p.add_argument("--output", default=f"/tmp/diffusion_gemma_{name}.json")
        if name in serving:
            p.add_argument("--base-url", default="http://127.0.0.1:8000")
            p.add_argument("--inputs")
            p.add_argument("--log", help="the server's log (configuration)")
        if name == "configuration":
            p.add_argument("--deployment", choices=("tp1", "tp2"), default="tp1")
        if name in ("reads", "step1"):
            p.add_argument("--all", action="store_true", help="every read of the inputs")
        if name == "preemption":
            p.add_argument("--requests", type=int, default=4)
            p.add_argument("--prompt-len", type=int, default=190_000)
            p.add_argument("--max-tokens", type=int, default=16_384)
        if name == "accuracy":
            p.add_argument("--limit", type=int)
        if name == "binding":
            p.add_argument("--identity", required=True, help="an identity record written by attend-measure --identity")
            p.add_argument("--trace-dir", required=True, help="the server's --profiler-config torch_profiler_dir")
            p.add_argument("--ranks", type=int, default=1)
        if name.startswith("attend-"):
            p.add_argument("--impl", default="/app/diffusion_gemma_attend.py")
            p.add_argument("--inputs")
            p.add_argument("--example")
            p.add_argument("--family", choices=("encode", "denoise", "mixed"))
            p.add_argument("--identity", help="attend-measure: also write the identity sets here")
        if name == "gencfg":
            p.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = cmd(args)
    Path(args.output).write_text(json.dumps(rec, indent=1, default=str))
    summary = {k: rec[k] for k in ("pass",) if isinstance(rec, dict) and k in rec}
    print(json.dumps({"command": args.command, "output": args.output, **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
