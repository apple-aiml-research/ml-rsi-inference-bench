#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Log-probability comparison protocols against a server at the prescribed recipe.

Start the server with `/app/eval/kimi_linear/launch.py`, then:

    python check.py --mode author  --url http://127.0.0.1:8000     # examples.json
    python check.py --mode decode  --url http://127.0.0.1:8000     # fixtures/decode.json
    python check.py --mode cross   --url http://127.0.0.1:8000     # fixtures/cross.json
    python check.py --mode router  --url http://127.0.0.1:8000     # fixtures/router.json

Each mode reports every sequence's reading and each band's median. It reports measurements only:
the numerical bars are not part of these materials, and a run of these examples does not determine
task acceptance. The exit status is 0 when every reading was a structurally valid observation, 1
when a served response was not (the check stops there), and 2 when the recorded side of a
comparison is itself unusable.

The statistic (`kl.py`) is SGLang's k3 form: for one sequence, with `ref` the reference side's
log-probability of each actual next token and `cand` the other side's at the same position,
`logr = ref - cand` and `k3 = mean over positions of ((exp(logr) - 1) - logr)`. A band's reading is
the median over its sequences.

A reading is structurally valid only when the response has exactly one entry per requested
position, the null first entry, the requested token's own key at every conditioned position, and a
finite number there. A response that is not is a failed observation of the server, and no reading
made afterwards replaces it. Among valid readings, a finite difference whose exponential overflows
is k3 = +infinity; such a sequence is sent once more at once and the lower of its two valid
readings counts.

* author: prefill each sequence and compare with the model author's implementation's values
  (the Transformers implementation of `reference/modeling_kimi.py`, BF16, one sequence at a time,
  unpadded): `examples.json`, eight short and two long sequences.
* decode: generate a 1,024-token continuation at temperature 1.0 with the prompt's seed,
  reporting each token's log-probability while decoding; prefill prompt + continuation; compare
  the decoding values (`cand`) with the prefill's (`ref`) over the generated positions. This
  compares the server with itself.
* cross: prefill the prompt plus a continuation an engine implementation of this model generated
  at the recipe, and compare with the values that engine reported while decoding it (recorded
  engine values, not the model author's). `fixtures/cross.json` also carries each prompt's self
  reading, the decode mode's.
* router: prefill each sequence and compare with the recorded engine values, over the whole
  sequence and over the fixture's selected positions.

Values are retrieved teacher-forced through the completions API:

    POST /v1/completions
    {"prompt": [<token ids>], "max_tokens": 0, "echo": true, "prompt_logprobs": 0, "temperature": 0}

Entry 0 of `prompt_logprobs` is null (the first token has no predecessor); entry i holds the
log-probability of prompt token i, keyed by its id. Every request carries its own `cache_salt`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def _sibling(name):
    spec = importlib.util.spec_from_file_location("infb_fidelity_" + name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kl = _sibling("kl")
LIMITS = json.loads((HERE.parent / "operation-limits.json").read_text())


class MalformedObservation(AssertionError):
    """A served response that is not a structurally valid reading. Terminal for its check.

    `detail` and `bands`, when set, carry every reading made before it so the record keeps them.
    """

    detail: list | None = None
    bands: dict | None = None


class OperationLimitExceeded(MalformedObservation):
    """A request that did not complete within its published operation limit. Terminal for its check."""


class ReferenceFault(Exception):
    """The recorded side of a comparison is unusable. It is never the candidate's failure."""


def finite_number(value) -> bool:
    """A real number that is finite. Booleans are not numbers here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_reference(values, count: int, label: str) -> list[float]:
    """The recorded values of one comparison: exactly `count` finite numbers, checked before any request."""
    if not isinstance(values, list) or len(values) != count:
        got = len(values) if isinstance(values, list) else type(values).__name__
        raise ReferenceFault(f"{label}: the recorded side holds {got} values, not {count}")
    bad = [i for i, v in enumerate(values) if not finite_number(v)]
    if bad:
        raise ReferenceFault(f"{label}: recorded values at {bad[:4]} are not finite numbers")
    return [float(v) for v in values]


def prompt_logprob_values(entries, ids: list[int]) -> list[float]:
    """The requested tokens' log-probabilities at positions 1..n-1 of one prompt_logprobs array.

    Exact length, the null first entry, the requested token's own key at every conditioned position
    and a finite number there are checked before anything is returned. A single entry keyed by some
    other token is not the requested token's value, whatever its number.
    """
    if not isinstance(entries, list):
        raise MalformedObservation(f"prompt_logprobs is {type(entries).__name__}, not a list")
    if len(entries) != len(ids):
        raise MalformedObservation(
            f"prompt_logprobs has {len(entries)} entries for a {len(ids)}-token prompt")
    if entries[0] is not None:
        raise MalformedObservation("the first prompt_logprobs entry is not null; position 0 has no predecessor")
    out = []
    for position, token_id in enumerate(ids[1:], start=1):
        entry = entries[position]
        if not isinstance(entry, dict) or not entry:
            raise MalformedObservation(f"prompt_logprobs entry {position} is {entry!r}, not a non-empty mapping")
        if str(token_id) in entry:
            value = entry[str(token_id)]
        elif token_id in entry:
            value = entry[token_id]
        else:
            raise MalformedObservation(
                f"prompt_logprobs entry {position} has no value for the requested token {token_id}; "
                f"its keys are {sorted(map(str, entry))[:4]}")
        raw = value.get("logprob") if isinstance(value, dict) else value
        if not finite_number(raw):
            raise MalformedObservation(
                f"prompt_logprobs entry {position}: the requested token's log-probability is {raw!r}, "
                "not a finite number")
        out.append(float(raw))
    return out


def http_limit_s(input_tokens: int) -> float:
    """The operation limit of one completions request with this many input tokens."""
    limits = LIMITS["limits_s"]
    return float(limits["http_workload_long"] if input_tokens >= LIMITS["long_input_tokens"]
                 else limits["http_workload"])


def _post(completions_url: str, body: dict) -> dict:
    """One completions request; a response that is not a JSON completion is a failed observation."""
    import requests

    limit = http_limit_s(len(body["prompt"]))
    try:
        r = requests.post(completions_url, json=body, timeout=limit)
    except requests.exceptions.Timeout:
        raise OperationLimitExceeded(
            f"a {len(body['prompt'])}-token request did not complete within its {limit:g} s limit") from None
    if r.status_code != 200:
        raise MalformedObservation(f"the server answered HTTP {r.status_code}: {r.text[:300]!r}")
    try:
        payload = r.json()
        choice = payload["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise MalformedObservation(f"the response is not a completion: {exc!r}: {r.text[:300]!r}") from None
    if not isinstance(choice, dict):
        raise MalformedObservation(f"the response's first choice is {type(choice).__name__}")
    return choice


def candidate_logprobs(completions_url: str, ids: list[int], salt: str) -> list[float]:
    """Per-token log-probabilities from the served model for positions 1..n-1."""
    choice = _post(completions_url, {
        "prompt": ids,
        "max_tokens": 0,
        "echo": True,
        "prompt_logprobs": 0,
        "temperature": 0,
        "cache_salt": salt,
    })
    if choice.get("prompt_logprobs") is None:
        raise MalformedObservation("the server returned no prompt_logprobs for a request that asked for them")
    return prompt_logprob_values(choice["prompt_logprobs"], ids)


def read_author(completions_url: str, seq: dict, ref: dict, tag: str, prefix: str = "infb-fidelity") -> float:
    """One reading of one sequence: its k3 against the model author's values."""
    cand = candidate_logprobs(completions_url, seq["ids"], salt=f"{prefix}-{seq['id']}-{tag}")
    return kl.k3_per_sequence(ref["logprobs"], cand)


def generate(completions_url: str, ids: list[int], seed: int, max_tokens: int, salt: str) -> tuple[list[int], list[float]]:
    """A sampled continuation and the log-probability the server reported for each token while decoding it."""
    choice = _post(completions_url, {"prompt": ids, "max_tokens": max_tokens, "temperature": 1.0, "seed": seed,
                                     "ignore_eos": True, "skip_special_tokens": False, "logprobs": 1,
                                     "return_token_ids": True, "cache_salt": salt})
    ids_out = choice.get("token_ids")
    lps = (choice.get("logprobs") or {}).get("token_logprobs") if isinstance(choice.get("logprobs"), dict) else None
    if not isinstance(ids_out, list) or len(ids_out) != max_tokens:
        raise MalformedObservation(
            f"the server returned {len(ids_out) if isinstance(ids_out, list) else ids_out!r} token ids for a "
            f"{max_tokens}-token continuation with ignore_eos")
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in ids_out):
        raise MalformedObservation("the continuation's token ids are not all integers")
    if not isinstance(lps, list) or len(lps) != max_tokens:
        raise MalformedObservation(
            f"the server returned {len(lps) if isinstance(lps, list) else lps!r} decode log-probabilities for a "
            f"{max_tokens}-token continuation")
    bad = [i for i, v in enumerate(lps) if not finite_number(v)]
    if bad:
        raise MalformedObservation(f"decode log-probabilities at {bad[:4]} are not finite numbers: "
                                   f"{[lps[i] for i in bad[:4]]!r}")
    return ids_out, [float(v) for v in lps]


def read_decode(completions_url: str, p: dict, max_tokens: int, tag: str) -> float:
    """One reading of one decode prompt: generate with its seed, prefill prompt + continuation, and the k3 of the
    decode-time values against the prefill's, over the generated positions."""
    gen_ids, dec = generate(completions_url, p["ids"], p["seed"], max_tokens, salt=f"infb-decode-{p['id']}-{tag}")
    pre_all = candidate_logprobs(completions_url, p["ids"] + gen_ids, salt=f"infb-decode-prefill-{p['id']}-{tag}")
    return kl.k3_per_sequence(pre_all[len(p["ids"]) - 1:], dec)


def read_cross(completions_url: str, p: dict, rec: dict, tag: str) -> float:
    """One cross reading of one prompt: prefill the prompt and the recorded continuation, and score each
    continuation token's log-probability by k3 against the one the recording's engine reported while decoding it."""
    pre_all = candidate_logprobs(completions_url, p["ids"] + rec["continuation"], salt=f"infb-long-cross-{p['id']}-{tag}")
    return kl.k3_per_sequence(rec["decode_logprobs"], pre_all[len(p["ids"]) - 1:])


def router_reads(completions_url: str, ref: dict, selected: dict):
    """The router probe's two readings; one request per sequence per reading serves both."""
    values: dict = {}

    def candidate(s: dict, tag: str) -> list:
        if (s["id"], tag) not in values:
            values[(s["id"], tag)] = candidate_logprobs(completions_url, s["ids"], salt=f"infb-router-probe-{s['id']}-{tag}")
        return values[(s["id"], tag)]

    def read_whole(s: dict, tag: str) -> float:
        return kl.k3_per_sequence(ref[s["id"]], candidate(s, tag))

    def read_selected(s: dict, tag: str) -> float:
        cand, ks = candidate(s, tag), selected[s["id"]]
        return kl.k3_per_sequence([ref[s["id"]][k] for k in ks], [cand[k] for k in ks])

    return {"whole": read_whole, "selected": read_selected}


def check_references(mode: str, data: dict) -> None:
    """Every recorded value a mode compares against, validated before its first request."""
    if mode == "author":
        for e in data["examples"]:
            validate_reference(e["reference_logprobs"], len(e["ids"]) - 1, f"example {e['id']}")
    elif mode == "cross":
        for p in data["prompts"]:
            rec = data["recording"].get(p["id"])
            if rec is None or len(rec.get("continuation") or []) != data["max_tokens"]:
                raise ReferenceFault(f"{p['id']}: no {data['max_tokens']}-token recorded continuation")
            validate_reference(rec["decode_logprobs"], data["max_tokens"], f"recording {p['id']}")
    elif mode == "router":
        for s in data["sequences"]:
            n = len(s["ids"]) - 1
            validate_reference(data["reference"].get(s["id"]), n, f"router {s['id']}")
            ks = data["selected"].get(s["id"], [])
            if len(set(ks)) != len(ks) or not all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < n for k in ks):
                raise ReferenceFault(f"router {s['id']}: selected positions are not distinct indices below {n}")


def read_band(read, items: list, draw: int) -> tuple[dict, list]:
    """One draw of one band: every item read once. A structurally valid reading of +infinity (an overflowing
    term) is sent once more at once, and the lower of the two valid readings counts; every reading is kept.
    A failed observation ends the band there, carrying the readings made so far."""
    values, detail = {}, []
    for item in items:
        row = {"id": item["id"], "readings": []}
        detail.append(row)
        for tag in (f"d{draw}", f"d{draw}-resent"):
            try:
                k3 = read(item, tag)
            except MalformedObservation as exc:
                row["readings"].append({"tag": tag, "valid": False, "error": str(exc)})
                row["malformed"] = str(exc)
                exc.detail = detail
                raise
            row["readings"].append({"tag": tag, "valid": True, "k3": k3})
            if math.isfinite(k3):
                break
        row["k3"] = min(r["k3"] for r in row["readings"])
        values[item["id"]] = row["k3"]
    return values, detail


def graded_bands(read, items_by_band: dict, bars: dict) -> dict:
    """The graded verdict: every band's median below its own bar, a failing band re-drawn once whole and failing only
    if both draws fail. `read` is one reading function for every band, or a dict of them by band. A failed
    observation anywhere is terminal: the exception carries every band's readings so far. Returns every draw."""
    out = {}
    for band, items in items_by_band.items():
        band_read = read[band] if isinstance(read, dict) else read
        record = {"T_b": bars[band], "draws": []}
        out[band] = record
        for draw in (1, 2):
            try:
                values, detail = read_band(band_read, items, draw)
            except MalformedObservation as exc:
                record["draws"].append({"median": None, "detail": exc.detail, "malformed": str(exc)})
                record["passed"] = False
                exc.bands = out
                raise
            record["draws"].append({"median": median(values.values()), "detail": detail})
            if record["draws"][-1]["median"] < bars[band]:
                break
        record["passed"] = any(d["median"] < bars[band] for d in record["draws"])
        record["ratio"] = min(d["median"] for d in record["draws"]) / bars[band]
    return out


def measured_bands(read, items_by_band: dict) -> dict:
    """The measurement without a bar: one draw of every band, its readings and its median."""
    out = {}
    for band, items in items_by_band.items():
        band_read = read[band] if isinstance(read, dict) else read
        try:
            values, detail = read_band(band_read, items, 1)
        except MalformedObservation as exc:
            out[band] = {"median": None, "detail": exc.detail, "malformed": str(exc)}
            exc.bands = out
            raise
        out[band] = {"median": median(values.values()), "detail": detail}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["author", "decode", "cross", "router"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the running server")
    parser.add_argument("--input", type=Path, help="the example records (default: this mode's shipped file)")
    parser.add_argument("--output", type=Path, help="write the readings as JSON here")
    args = parser.parse_args(argv)
    url = args.url.rstrip("/") + "/v1/completions"
    default = {"author": HERE / "examples.json", "decode": HERE / "fixtures" / "decode.json",
               "cross": HERE / "fixtures" / "cross.json", "router": HERE / "fixtures" / "router.json"}
    data = json.loads((args.input or default[args.mode]).read_text())
    report = {"mode": args.mode, "status": "measured", "evidence_complete": False,
              "task_acceptance": "not_determined",
              "verdict": "measurement only; the bars are not supplied and full task acceptance is not determined",
              "failed": None}
    rc = 0
    try:
        check_references(args.mode, data)
        if args.mode == "author":
            by_band: dict = {}
            for example in data["examples"]:
                by_band.setdefault(example["band"], []).append(example)
            refs = {e["id"]: {"logprobs": e["reference_logprobs"]} for e in data["examples"]}
            bands = measured_bands(lambda s, tag: read_author(url, s, refs[s["id"]], tag, prefix="infb-example"), by_band)
        elif args.mode == "decode":
            by_band = {}
            for p in data["prompts"]:
                by_band.setdefault(p["regime"], []).append(p)
            bands = measured_bands(lambda p, tag: read_decode(url, p, data["max_tokens"], tag), by_band)
        elif args.mode == "cross":
            prompts = data["prompts"]
            reads = {"cross": lambda p, tag: read_cross(url, p, data["recording"][p["id"]], tag),
                     "self": lambda p, tag: read_decode(url, p, data["max_tokens"], tag)}
            bands = measured_bands(reads, {"cross": prompts, "self": prompts})
        else:
            seqs, selected = data["sequences"], data["selected"]
            bands = measured_bands(router_reads(url, data["reference"], selected),
                                   {"whole": seqs, "selected": [s for s in seqs if s["id"] in selected]})
        report["band_medians"] = {band: b["median"] for band, b in bands.items()}
        report["bands"] = bands
        report["evidence_complete"] = True
    except ReferenceFault as exc:
        report.update(status="reference_fault", failed=str(exc))
        rc = 2
    except MalformedObservation as exc:
        report.update(status="candidate_fail", failed=str(exc), bands=exc.bands)
        rc = 1
    body = json.dumps(report, indent=2)
    print(body)
    if args.output is not None:
        args.output.write_text(body + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
