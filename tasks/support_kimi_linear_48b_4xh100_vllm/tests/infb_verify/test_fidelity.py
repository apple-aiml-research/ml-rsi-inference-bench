#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The fidelity gates: the candidate's per-token log-probabilities against their authorities.

These nodes run in the same pytest session as `test_hidden.py`, against the one launch the session
fixture makes, so the fidelity sequences are served by exactly the process that answered the accuracy
and probe nodes.

What the gates are for. A candidate can pass registry
resolution, build, and even GSM8K while being subtly wrong. The model-author gate compares the
candidate's per-token log-probabilities against the **model-author Transformers implementation** on
held-out sequences, which is what makes a pass credible against lookup. The longer comparisons use
distinct authorities, each named in its record: decode-versus-prefill consistency within the
candidate, and engine-reference recordings at 64,000 tokens of context.

The reference is data, recorded once and shipped in `hidden/`; the image carries no reference
implementation and no fla-core. Retrieval is vLLM's own completions API:

    {"prompt": <token ids>, "max_tokens": 0, "echo": true, "prompt_logprobs": 0, "temperature": 0}

The reading, alignment, generation and band functions are the evaluation tree's own `fidelity/check.py`
(the grading copy, byte-identical to the supplied one), which validates every response's structure --
exact length, the null first entry, the requested token's own key and a finite number at every
conditioned position -- before any statistic. A structurally invalid response is terminal for its check,
whatever was read before or after it. The recorded side is validated before the first request; a defect
there is the reference's, never the candidate's. Every check persists its record (records.py) before
its verdict.
"""

from __future__ import annotations

import json
import os
from importlib import util
from pathlib import Path
from statistics import median

import pytest

from infb_verify import records

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))


def _load(name: str, path: Path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIDELITY = _load("infb_kimi_fidelity", TESTS_DIR / "eval/kimi_linear/fidelity/check.py")
kl = FIDELITY.kl
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
SEQUENCES = json.loads((TESTS_DIR / "hidden" / "fidelity-sequences.json").read_text())
REFERENCE = json.loads((TESTS_DIR / "hidden" / "fidelity-reference.json").read_text())

_graded_bands = FIDELITY.graded_bands


def _sequences() -> list[dict]:
    """The hidden 36. The public ten ship under /app/eval/kimi_linear/fidelity/ and are not scored."""
    return list(SEQUENCES["hidden"])


def _graded(check: str, validate, reads, items_by_band: dict, bars: dict, authority: str,
            extra: dict | None = None) -> dict:
    """Every band graded under the fixed retry policy, recorded before and after, verdict raised last.

    `validate()` checks the recorded side inside the record, before the first request."""
    def body(record: dict):
        validate()
        bands = _graded_bands(reads, items_by_band, bars)
        record["bands"] = bands
        record["band_medians"] = {b: v["draws"][0]["median"] for b, v in bands.items()}
        return all(b["passed"] for b in bands.values()), bands

    return records.run(check, {"authority": authority, "T_per_band": bars,
                               "n_sequences": {b: len(v) for b, v in items_by_band.items()}, **(extra or {})}, body)


def _failures(bands: dict, what: str) -> list[str]:
    out = []
    for band, b in bands.items():
        if not b["passed"]:
            readings = " then ".join(format(d["median"], ".6g") for d in b["draws"])
            out.append(f"the {band} {what}'s median k3 is {readings}, not below {b['T_b']:.6g}")
    return out


def test_fidelity_against_model_author(server):
    """The gate, per band: every band's median k3 below its own bar T_b = max(0.05, 3 x the oracle's
    band median over its 13 draws), 0.062186 short and 0.05 long (config.json `fidelity.thresholds`). A structurally
    valid reading of +infinity is sent once more at once, and a band that fails is re-drawn once whole; a band fails
    only if both draws do. A structurally invalid response fails the node at once."""
    by_id = {r["id"]: r for r in REFERENCE["hidden"]}
    thresholds = X["fidelity"]["thresholds"]
    by_band: dict[str, list] = {}
    for seq in _sequences():
        by_band.setdefault(seq["band"], []).append(seq)

    def validate():
        for seq in _sequences():
            ref = by_id.get(seq["id"])
            if ref is None or ref.get("ids_sha256") != seq["ids_sha256"]:
                raise FIDELITY.ReferenceFault(f"{seq['id']}: no reference recorded against these token ids")
            FIDELITY.validate_reference(ref["logprobs"], len(seq["ids"]) - 1, f"reference {seq['id']}")

    url = server.url_for("v1/completions")
    bands = _graded("test_fidelity_against_model_author", validate,
                    lambda seq, tag: FIDELITY.read_author(url, seq, by_id[seq["id"]], tag), by_band, thresholds,
                    "model-author recordings (Transformers implementation of the supplied source)")
    first = {row["id"]: row["k3"] for b in bands.values() for row in b["draws"][0]["detail"]}
    print("INFB_FIDELITY " + json.dumps({
        "scored_statistic": max(b["draws"][0]["median"] for b in bands.values()),
        "scored_band": max(bands, key=lambda b: bands[b]["draws"][0]["median"]),
        "band_medians": {b: v["draws"][0]["median"] for b, v in bands.items()},
        "whole_set_median": median(first.values()), "n_sequences": len(first),
        "thresholds": thresholds, "bands": bands}))
    failures = _failures(bands, "band")
    assert not failures, (
        "the candidate's per-token distribution disagrees with the model-author implementation: " + "; ".join(failures))


DECODE = json.loads((TESTS_DIR / "hidden" / "decode-prompts.json").read_text())


def test_decode_agrees_with_prefill(server):
    """Decode-time numerics: on continuations the graded server generates itself -- 1,024 sampled tokens per hidden
    prompt, four long ones that cross several prefill chunks before decoding starts and four short ones -- the
    log-probability it reports for each generated token while decoding agrees with the one it gives the same token
    when prompt + continuation is prefilled. The authority is the candidate's own prefill (self-consistency), not
    independent model truth; the model-author gate ties the prefill to the model author. Each regime has its own bar,
    T_b = 0.05 (config.json `fidelity.decode.T_per_band`), under the fixed retry policy."""
    cfg = X["fidelity"]["decode"]
    bars = cfg["T_per_band"]
    by_band: dict[str, list] = {}
    for p in DECODE["prompts"]:
        by_band.setdefault(p["regime"], []).append(p)
    url = server.url_for("v1/completions")
    def validate():
        for p in DECODE["prompts"]:
            if not p.get("ids") or not all(isinstance(t, int) for t in p["ids"]) or not isinstance(p.get("seed"), int):
                raise FIDELITY.ReferenceFault(f"{p.get('id')}: the prompt is not a token list with its seed")

    bands = _graded("test_decode_agrees_with_prefill", validate,
                    lambda p, tag: FIDELITY.read_decode(url, p, cfg["max_tokens"], tag), by_band, bars,
                    "decode-versus-prefill consistency of the candidate")
    print("INFB_DECODE " + json.dumps({
        "scored_statistic": max(b["draws"][0]["median"] for b in bands.values()),
        "scored_band": max(bands, key=lambda b: bands[b]["draws"][0]["median"]),
        "band_medians": {b: v["draws"][0]["median"] for b, v in bands.items()},
        "n_sequences": sum(len(v) for v in by_band.values()), "T_per_band": bars, "bands": bands}))
    failures = _failures(bands, "regime")
    assert not failures, (
        "the log-probabilities the server reports while decoding disagree with the ones it gives the same "
        "continuation when prefilled: " + "; ".join(failures))


LONG_CONTEXT = json.loads((TESTS_DIR / "hidden" / "long-context.json").read_text())


def test_long_context_decode(server):
    """Prompts of 64,000 tokens, prefilled in 32 chunks of 2,048 before anything is decoded. Two bands, each with its
    own authority and bar (config.json `fidelity.long_context.T_per_band`):
      * cross: the candidate prefills the prompt and the landing's recorded continuation (512 tokens sampled at
        temperature 1.0 with the prompt's seed), and each continuation token's log-probability is scored by k3 against
        the one the landing reported while decoding it. The authority is an engine-reference recording, not the model
        author's;
      * self: the candidate's own continuation with the same seed, read by the decode check (self-consistency).
    The fixed retry policy applies; a structurally invalid response fails the node at once."""
    cfg = X["fidelity"]["long_context"]
    prompts = LONG_CONTEXT["prompts"]

    def validate():
        # A set silently emptied or shortened is a gate that silently stops gating.
        if len(prompts) != cfg["n_prompts"] or any(len(p["ids"]) != cfg["prompt_tokens"] for p in prompts):
            raise FIDELITY.ReferenceFault(
                f"the long-context set has {[len(p['ids']) for p in prompts]} tokens per prompt; expected "
                f"{cfg['n_prompts']} of {cfg['prompt_tokens']}")
        for p in prompts:
            rec = LONG_CONTEXT["recording"].get(p["id"])
            if (rec is None or rec.get("ids_sha256") != p["ids_sha256"]
                    or len(rec.get("continuation") or []) != cfg["max_tokens"]):
                raise FIDELITY.ReferenceFault(
                    f"{p['id']}: no {cfg['max_tokens']}-token continuation recorded against these ids")
            FIDELITY.validate_reference(rec["decode_logprobs"], cfg["max_tokens"], f"recording {p['id']}")

    url = server.url_for("v1/completions")
    reads = {"cross": lambda p, tag: FIDELITY.read_cross(url, p, LONG_CONTEXT["recording"][p["id"]], tag),
             "self": lambda p, tag: FIDELITY.read_decode(url, p, cfg["max_tokens"], tag)}
    bars = cfg["T_per_band"]
    bands = _graded("test_long_context_decode", validate, reads, {"cross": prompts, "self": prompts}, bars,
                    "cross: the landing's recorded 64,000+512 continuation (engine reference); self: the candidate's "
                    "decode-versus-prefill consistency")
    print("INFB_LONG_CONTEXT " + json.dumps({
        "scored_statistic": max(b["draws"][0]["median"] for b in bands.values()),
        "scored_band": max(bands, key=lambda b: bands[b]["draws"][0]["median"]),
        "band_medians": {b: v["draws"][0]["median"] for b, v in bands.items()},
        "n_sequences": len(prompts), "T_per_band": bars, "bands": bands}))
    failures = _failures(bands, "band")
    assert not failures, (
        "at 64,000 tokens of context, the server's log-probabilities disagree with the landing's continuation or "
        "with its own prefill: " + "; ".join(failures))


ROUTER_PROBE = json.loads((TESTS_DIR / "hidden" / "router-probe.json").read_text())


def test_router_probe(server):
    """The MoE router where its choices matter: forty-six hidden natural-text sequences of 2,048 tokens,
    prefilled once each, and the candidate's log-probability of every actual next token scored by k3 against the
    landing's own values (engine-reference recordings) in two bands, whole sequences and the hidden selected
    positions, each with its bar (config.json `fidelity.router_probe.T_per_band`), under the fixed retry policy."""
    cfg = X["fidelity"]["router_probe"]
    seqs = ROUTER_PROBE["sequences"]
    ref, selected = ROUTER_PROBE["reference"], ROUTER_PROBE["selected"]

    def validate():
        # A set silently emptied or reshaped is a gate that silently stops gating.
        if len(seqs) != cfg["n_sequences"] or sum(len(v) for v in selected.values()) != cfg["k_positions"]:
            raise FIDELITY.ReferenceFault(
                f"the router probe has {len(seqs)} sequences and {sum(len(v) for v in selected.values())} "
                f"selected positions; expected {cfg['n_sequences']} and {cfg['k_positions']}")
        for s in seqs:
            n = cfg["sequence_tokens"] - 1
            if len(s["ids"]) != cfg["sequence_tokens"]:
                raise FIDELITY.ReferenceFault(f"{s['id']}: {len(s['ids'])} tokens, not {cfg['sequence_tokens']}")
            FIDELITY.validate_reference(ref.get(s["id"]), n, f"router reference {s['id']}")
            ks = selected.get(s["id"], [])
            if len(set(ks)) != len(ks) or not all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < n
                                                  for k in ks):
                raise FIDELITY.ReferenceFault(f"{s['id']}: selected positions are not distinct indices below {n}")
    # One request per sequence per reading serves both bands.
    reads = FIDELITY.router_reads(server.url_for("v1/completions"), ref, selected)
    bars = cfg["T_per_band"]
    bands = _graded("test_router_probe", validate, reads,
                    {"whole": seqs, "selected": [s for s in seqs if s["id"] in selected]}, bars,
                    "the landing's recorded values at the recipe (engine reference)")
    print("INFB_ROUTER_PROBE " + json.dumps({
        "scored_band": max(bands, key=lambda b: bands[b]["draws"][0]["median"] / bars[b]),
        "band_medians": {b: v["draws"][0]["median"] for b, v in bands.items()},
        "n_sequences": len(seqs), "T_per_band": bars, "bands": bands}))
    failures = _failures(bands, "band")
    assert not failures, (
        "the server's log-probabilities disagree with the landing's where the MoE router's choices matter: "
        + "; ".join(failures))


@pytest.mark.parametrize("band", ["short", "long"])
def test_fidelity_band_is_populated(band):
    """A band silently emptied is a gate that silently stops gating."""
    counts = {"short": 32, "long": 4}
    got = sum(1 for s in _sequences() if s["band"] == band)
    assert got == counts[band], f"band {band} has {got} sequences, expected {counts[band]}"
