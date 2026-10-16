#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run public Qwen38 observations and measurements against a server at either stated recipe.

Grading supplies its own inputs and acceptance policies to these same modules. Every command
reports complete observations and measurements without guessing private thresholds, and fails on
a structurally invalid observation. Run one workload at a time on an idle server. `--help` on a
command lists its options; `/app/eval/qwen38/README.md` documents each protocol.

    configuration   the served configuration recorded in a launch log
    api             chat, streaming, length termination and concurrent isolation; completions,
                    streamed completions and prompt probabilities
    fidelity        teacher-forced observations against the model-author examples
    mtp-series      fixed-length greedy series (either recipe)
    mtp-compare     ordinary against MTP3 series: validity and first-divergence agreement
    mtp-acceptance  isolated draft counters and complete responses (MTP3)
    prefix-reuse    one warm/cold prefix-reuse pair at the served cache's own boundary (ordinary)
    components      native QSA/PLE capture, three timing matrices, PLE memory and use evidence
    declaration     the native regions the server resolves (default or /app/qwen38-adapter.json)
    accuracy        the offline EvalScope accuracy inventories
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).resolve().parent


def _prompts(path: Path) -> list[tuple[str, str]]:
    return [(row["id"], row["prompt"]) for row in json.loads(path.read_text())["cases"]]


def _model(base: str) -> str:
    from qwen38.serving.mtp import request
    return request(base, "/v1/models", timeout=60)["data"][0]["id"]


def _tokenize(base: str, model: str, text: str) -> list[int]:
    from qwen38.serving.mtp import request
    return request(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": True},
                   timeout=60)["tokens"]


def _vocabulary(args) -> int:
    from qwen38.measurement import observations
    return observations.vocabulary_size(args.model_dir)


def _identity(args) -> dict:
    from qwen38.serving import mtp
    return {"evaluation_id": args.evaluation_id, "source": mtp.source_identity(args.app),
            "tokenizer": mtp.tokenizer_identity(args.model_dir)}


def run(args) -> dict:
    from qwen38.measurement import budget, observations
    from qwen38.serving import accuracy, api, configuration, fidelity, mtp, recipes, reuse
    if args.protocol == "configuration":
        if args.launch_log is None:
            raise SystemExit("configuration requires --launch-log")
        return configuration.served_configuration(args.launch_log.read_text(), args.recipe)
    if args.protocol == "mtp-compare":
        if args.ordinary is None or args.mtp3 is None:
            raise SystemExit("mtp-compare requires --ordinary and --mtp3")
        ordinary, draft = json.loads(args.ordinary.read_text()), json.loads(args.mtp3.read_text())
        ids = [row["id"] for row in ordinary["inputs"]]
        return {"comparison": observations.compare(ordinary, draft, ids, ordinary["max_tokens"],
                                                   vocabulary=ordinary["vocabulary"])}
    vocabulary = _vocabulary(args)
    if args.protocol == "api":
        prompt = _prompts(args.inputs or HERE / "examples/prompts.json")[0][1]
        with budget.operation("api_observation_batch"):
            probes = api.probes(args.base_url, recipes.MODEL, prompt, vocabulary=vocabulary)
        ids = _tokenize(args.base_url, recipes.MODEL, prompt)
        return {"recipe": args.recipe, "probes": probes,
                "completions": api.completions_example(args.base_url, recipes.MODEL, ids, vocabulary=vocabulary)}
    if args.protocol == "fidelity":
        data = json.loads((args.inputs or HERE / "examples/fidelity/examples.json").read_text())
        with budget.operation("fidelity_evaluation"):
            return fidelity.evaluate(args.base_url, recipes.MODEL, data["sequences"], vocabulary=vocabulary)
    if args.protocol in ("mtp-series", "mtp-acceptance"):
        cases = _prompts(args.inputs or HERE / "examples/prompts.json")
        if args.protocol == "mtp-acceptance":
            with budget.operation("mtp_acceptance"):
                return mtp.acceptance(args.base_url, args.tokens, cases, vocabulary=vocabulary, **_identity(args))
        with budget.operation("comparison_series_recording"):
            report = mtp.collect_series(args.base_url, args.tokens, cases, recipe=args.recipe,
                                        vocabulary=vocabulary, **_identity(args))
        observations.validate_series(report, [cid for cid, _ in cases], args.tokens, vocabulary=vocabulary)
        return report
    if args.protocol == "prefix-reuse":
        data = json.loads((args.inputs or HERE / "examples/prefix_reuse.json").read_text())
        ids = _tokenize(args.base_url, recipes.MODEL, data["prefix"])
        with budget.operation("prefix_agreement"):
            return reuse.pair(args.base_url, recipes.MODEL, ids, tail=data["tail_tokens"], tokens=data["tokens"],
                              vocabulary=vocabulary, salt=f"qwen38-cold-{uuid.uuid4().hex}")
    if args.protocol == "components":
        return components(args, vocabulary)
    if args.protocol == "declaration":
        from qwen38.measurement import client
        return {"regions": client.rpc(args.base_url, {"action": "identity"}, timeout=120)}
    if args.protocol == "accuracy":
        import yaml
        config = yaml.safe_load(args.inputs.read_text())
        return {"scores": accuracy.run_evalscope(config, args.base_url, args.work_dir, dataset_root=args.dataset_root),
                "passed": None}
    raise SystemExit(f"unknown protocol {args.protocol}")


def components(args, vocabulary: int) -> dict:
    """Public example cases take token content from the public prompts; a complete input file
    (schema qwen38-component-inputs/1) carries its own."""
    from qwen38.measurement import cases as case_rules
    from qwen38.measurement import client
    data = json.loads((args.inputs or HERE / "examples/components.json").read_text())
    if data.get("schema") == "qwen38-component-inputs/1":
        if data["recipe"] != args.recipe:
            raise SystemExit(f"these inputs are for the {data['recipe']} recipe")
        cases, tokens, tolerance = data["cases"], data["tokens"], data.get("use_tolerance")
        enclosing = data.get("enclosing")
    else:
        cases = [row for row in data[args.recipe]] if args.recipe in data else data["cases"]
        from qwen38.serving import recipes
        pool = [_tokenize(args.base_url, recipes.MODEL, prompt) for _, prompt in _prompts(HERE / "examples/prompts.json")]
        tokens = {c["id"]: case_rules.tokens(c, pool, tag=f"example-{uuid.uuid4().hex[:8]}") for c in cases}
        tolerance = enclosing = None
    bounds = case_rules.scope()
    for case in cases:
        case_rules.validate(dict(case), bounds)
    return client.collect(args.base_url, args.recipe, cases, tokens, output=args.output, vocabulary=vocabulary,
                          use_tolerance=tolerance, budget_s=args.budget_s, matrix_budget_s=args.matrix_budget_s,
                          enclosing=enclosing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("protocol", choices=("configuration", "api", "fidelity", "mtp-series", "mtp-compare",
                                             "mtp-acceptance", "prefix-reuse", "components", "declaration",
                                             "accuracy"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--recipe", choices=("ordinary", "mtp3"), default="ordinary")
    parser.add_argument("--inputs", type=Path, help="the protocol's input file (defaults: the examples/ files)")
    parser.add_argument("--ordinary", type=Path)
    parser.add_argument("--mtp3", type=Path)
    parser.add_argument("--launch-log", type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--evaluation-id", default="qwen38-development")
    parser.add_argument("--app", type=Path, default=Path("/app"))
    parser.add_argument("--model-dir", type=Path, default=Path("/model"))
    parser.add_argument("--budget-s", type=float, help="components: the collection allowance (budgets.json default)")
    parser.add_argument("--matrix-budget-s", type=float, help="components: one matrix's allowance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/qwen38-accuracy"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/public/datasets"))
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.protocol != "components":  # the collection client writes its own record as it goes
        args.output.write_text(json.dumps(report, indent=1, allow_nan=False))
    print(str(args.output))
    return int(bool(report.get("departures")) or report.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
