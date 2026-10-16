#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""MiniMax-M3 evaluation protocols (see README.md beside this file).

Server protocols take ``--base-url`` of a running deployment (``serving/recipes.py``); MSA protocols run the operator
contract ``/app/minimax_m3_msa.py`` on one GPU with no server resident. Every command writes its measurements to
``--output``; none declares the task passed.

  configuration  served configuration read from a launch log and /v1/models
  api            API structure probes
  chat           reasoning split checks (streaming and non-streaming)
  tools          tool-call checks (streaming and non-streaming)
  fidelity       teacher-forced log-probabilities against recorded model-author values
  replay         generated-continuation replay
  prefix-branch  prefix-branch replay with prefix-cache reuse evidence
  accuracy       GSM8K chat with thinking
  msa-check      MSA operator correctness on the public examples
  msa-measure    MSA operator timing
  msa-sparsity   MSA context scaling
  binding        MSA kernel attribution (component identity sets; serving traces when given)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

EXAMPLES = HERE / "examples"


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, default=str) + "\n")
    print(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("configuration", "api", "chat", "tools", "fidelity", "replay", "prefix-branch", "accuracy"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8000")
        p.add_argument("--output", type=Path, required=True)
        if name == "configuration":
            p.add_argument("--log", type=Path, required=True, help="the server's launch log")
            p.add_argument("--deployment", choices=("bf16kv", "fp8kv"), required=True)
        if name in ("fidelity",):
            p.add_argument("--inputs", type=Path, default=Path("/public/fidelity-examples/examples.json"))
        if name in ("replay", "prefix-branch", "chat", "tools"):
            p.add_argument("--inputs", type=Path)
        if name == "accuracy":
            p.add_argument("--limit", type=int)
    for name in ("msa-check", "msa-measure", "msa-sparsity", "binding"):
        p = sub.add_parser(name, add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER)
    a, extra = ap.parse_known_args(argv)

    if a.cmd == "msa-check":
        from minimax_m3.msa import check
        return check.main(extra + a.rest)
    if a.cmd == "msa-measure":
        from minimax_m3.msa import measure
        return measure.main(extra + a.rest)
    if a.cmd == "msa-sparsity":
        from minimax_m3.msa import sparsity
        return sparsity.main(extra + a.rest)
    if a.cmd == "binding":
        from minimax_m3.msa import binding
        return binding.main(extra + a.rest)

    base = a.base_url
    if a.cmd == "configuration":
        from minimax_m3.serving import configuration
        _write(a.output, configuration.served_configuration(a.log.read_text(), a.deployment, base_url=base))
    elif a.cmd == "api":
        from minimax_m3.serving import api
        ids = json.loads((EXAMPLES / "replay.json").read_text())["prompts"][0]["token_ids"][:512]
        _write(a.output, api.probes(base, prompt_ids=ids))
    elif a.cmd in ("chat", "tools"):
        from minimax_m3.serving import chat
        cases = json.loads((a.inputs or EXAMPLES / "chat.json").read_text())
        out = []
        for c in cases["reasoning" if a.cmd == "chat" else "tools"]:
            for streaming in (False, True):
                resp = chat.chat(base, c["messages"], mode=c.get("thinking_mode"), streaming=streaming,
                                 tools=c.get("tools"))
                verdict = (chat.check_reasoning(resp, c["thinking_mode"]) if a.cmd == "chat"
                           else chat.check_tools(resp, c["tools"]))
                out.append({"id": c["id"], "streaming": streaming, **verdict})
        _write(a.output, {"cases": out, "note": "local checks on public examples only"})
    elif a.cmd == "fidelity":
        from minimax_m3.serving import fidelity
        seqs = json.loads(a.inputs.read_text())["sequences"]
        res = fidelity.evaluate(base, seqs)
        res.pop("reads", None)
        _write(a.output, {**res, "note": "measurements only; grading uses its own sequences and bars"})
    elif a.cmd == "replay":
        from minimax_m3.serving import replay
        spec = json.loads((a.inputs or EXAMPLES / "replay.json").read_text())
        rows = replay.generation_replay(base, [{"concurrency": 1, "prompts": spec["prompts"]}], spec["generate"])
        _write(a.output, {"rows": rows, "note": "measurements only; grading uses its own prompts and bars"})
    elif a.cmd == "prefix-branch":
        from minimax_m3.serving import replay
        spec = json.loads((a.inputs or EXAMPLES / "prefix.json").read_text())
        res = replay.prefix_branches(base, spec["prefix"], spec["suffixes"], spec["generate"])
        _write(a.output, {**res, "note": "measurements only; grading uses its own prefixes and bars"})
    elif a.cmd == "accuracy":
        from minimax_m3.serving import accuracy
        _write(a.output, accuracy.evaluate(base, limit=a.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
