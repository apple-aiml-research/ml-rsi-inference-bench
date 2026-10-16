#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The public protocols of `support_inkling_small_4xb200_sglang` (README.md has the task's commands).

    python /app/eval/inkling_small/run.py <command> [options]      (`<command> --help` for its options)

  registered     the registered test's classes verbatim on /app (--launch deterministic|production)
  configuration  the served configuration a running server reports, against the registered recipe
  api            structure probes of /generate and /v1/completions (logprobs, token-ID prompts, streaming, concurrency)
  fidelity       prompt log-probabilities against the author's reference series (k3 per position)
  replay         generated continuations against a cold teacher-forced replay of them (d and k3 per token)
  prefix-branch  continuations after prefix-cache reuse against their cold replay, with the reuse evidence
  conv-state     the first token generated after a prefix-cache restore against a cold run, with the reuse evidence
                 (--history also reads whether the restore depends on what ran before it)
  efficiency     per-user decode speed (c = 1) and output throughput (c = 16) with the grader's own client
  rel-check      the relative-attention entry point on the public examples, by the numerical policy
  rel-measure    the entry point timed by the grading protocol
  binding        the kernels the entry point launches; with --base-url, those kernels observed in the served model

Each command prints and optionally writes (--output) measurements. None prints the hidden verdict: grading uses
hidden inputs and private bars, and compares the efficiency measurements with its own paired reference.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "registered": HERE / "serving" / "deterministic.py",
    "configuration": HERE / "serving" / "configuration.py",
    "api": HERE / "serving" / "api.py",
    "fidelity": HERE / "serving" / "fidelity.py",
    "efficiency": HERE / "serving" / "measure.py",
    "rel-check": HERE / "rel_attention" / "check.py",
    "rel-measure": HERE / "rel_attention" / "measure.py",
    "binding": HERE / "rel_attention" / "binding.py",
}
WITNESS = HERE / "serving" / "witness.py"
PACKAGE = ("replay", "prefix-branch", "conv-state")


def run_script(path: Path, argv: list[str]) -> int:
    sys.argv = [str(path), *argv]
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0) if not isinstance(exc.code, str) else 1
    return 0


def run_package(command: str, argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog=f"run.py {command}")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    if command == "conv-state":
        inputs_help = ('{"new_tokens": 16, "windows": [{"id": ..., "L0": ..., "token_ids": [at least L0 + 2 IDs]}]}; '
                       "any token windows work, none ships as an example")
    else:
        inputs_help = f"e.g. {HERE / 'examples' / ('replay.json' if command == 'replay' else 'prefix.json')}"
    ap.add_argument("--inputs", type=Path, required=True, help=inputs_help)
    ap.add_argument("--new-tokens", type=int, help="default: the inputs' new_tokens")
    ap.add_argument("--timeout", type=float, default=900.0, help="per HTTP request (limits.json http_request)")
    if command == "conv-state":
        ap.add_argument("--history", action="store_true",
                        help="also read history independence: each window's restored request twice and its cold "
                             "request twice, under different slot histories; each pair must come back identical")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    doc = json.loads(args.inputs.read_text())
    new_tokens = args.new_tokens or doc.get("new_tokens", 16 if command == "conv-state" else 132)
    sys.path.insert(0, str(HERE.parent))
    if command == "replay":
        from inkling_small.serving import replay
        groups = doc["groups"]
        res = replay.run(args.base_url, [[s["token_ids"] for s in g] for g in groups], new_tokens=new_tokens,
                         timeout=args.timeout)
        ids = [s.get("id") for g in groups for s in g]
        rec = {"inputs": str(args.inputs), "new_tokens": new_tokens,
               "sequences": [{"id": i, **{k: v for k, v in r.items() if k not in ("d", "k3")}} for i, r in zip(ids, res)],
               "per_token_d": [{"id": i, "d": r.get("d")} for i, r in zip(ids, res)]}
        for s in rec["sequences"]:
            print(json.dumps(s))
    elif command == "conv-state":
        from inkling_small.serving import conv_state
        rec = {"inputs": str(args.inputs), "new_tokens": new_tokens,
               **conv_state.run(args.base_url, doc["windows"], new_tokens=new_tokens, timeout=args.timeout)}
        for w in rec["windows"]:
            print(json.dumps({k: w.get(k) for k in ("id", "L0", "cached_tokens", "prompt_len", "flipped",
                                                     "first_divergence", "first_token_d", "prompt_mode_d",
                                                     "structural_problems")}))
        print(json.dumps(rec["summary"]))
        if args.history:
            hist = conv_state.history_run(args.base_url, doc["windows"], new_tokens=new_tokens,
                                          timeout=args.timeout)
            rec["history"] = hist
            for w in hist["windows"]:
                print(json.dumps({k: w.get(k) for k in (
                    "id", "L0", "prompt_len", "cached_tokens_A", "cached_tokens_B", "restored_pair_identical",
                    "restored_first_diff", "restored_abs_d", "cold_pair_identical", "cold_first_diff",
                    "cold_abs_d", "structural_problems")}))
            print(json.dumps(hist["summary"]))
    else:
        from inkling_small.serving import prefix_branch
        rec = {"inputs": str(args.inputs), "new_tokens": new_tokens,
               **prefix_branch.run(args.base_url, doc["waves"], new_tokens=new_tokens, timeout=args.timeout)}
        for s in rec.get("served", []):
            print(json.dumps({k: s.get(k) for k in ("wave", "id", "prompt_len", "cached_tokens")}))
        for r in rec.get("replays", []):
            print(json.dumps({k: v for k, v in r.items() if k not in ("d", "k3")}))
    if args.output:
        args.output.write_text(json.dumps(rec, indent=1))
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, argv = sys.argv[1], sys.argv[2:]
    if command in PACKAGE:
        return run_package(command, argv)
    if command == "binding" and any(a == "--base-url" or a.startswith("--base-url=") for a in argv):
        return run_script(WITNESS, argv)
    if command not in SCRIPTS:
        print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
        return 2
    return run_script(SCRIPTS[command], argv)


if __name__ == "__main__":
    sys.exit(main())
