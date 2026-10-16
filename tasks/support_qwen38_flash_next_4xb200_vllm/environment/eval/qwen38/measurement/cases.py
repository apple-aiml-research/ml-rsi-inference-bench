#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Component cases: validation against the public scope, and token content for examples.

A case is `{"id", "component": "qsa"|"ple", "mode": "prefill"|"decode"|"spec", "lengths": [...],
"histories": [...], "steps", "layer", "seed", "acceptance", "joining", "timed"}`: per request, `lengths`
active tokens per measured step (prefill: the prefill tokens; decode: 1; spec: 1 + 3 draft positions)
after `histories` tokens already computed; `steps` measured steps (prefill: 1); `layer` indexes the
loaded regions of the component in checkpoint order; `acceptance` (spec only) is `"native"` (the
engine's own drafts and verification decide progress) or accepted-draft counts 0..3 per step.
`joining` (spec only) is `[[step, tokens], ...]`: at measured step `step` (0-based, increasing) one
fresh request joins with a prefill of `tokens` prompt tokens in the same runner step, a mixed
speculative/prefill iteration, and finishes after it. `timed` (default true): an untimed case is
captured and checked like any other but has no timed matrices and no memory accounting; a joining
case is untimed. `validate` enforces `scope.json`'s component envelope; a case outside it is refused,
never measured.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SCOPE = Path(__file__).resolve().parents[1] / "scope.json"
FIELDS = {"id", "component", "mode", "lengths", "histories", "steps", "layer", "seed", "acceptance", "joining",
          "timed"}
QSA_LAYERS = 12


class CaseError(ValueError):
    """A case outside the public scope or malformed."""


def scope() -> dict:
    return json.loads(SCOPE.read_text())


def validate(case: dict, bounds: dict | None = None) -> dict:
    b = (bounds or scope())["components"]
    if not isinstance(case, dict) or set(case) - FIELDS:
        raise CaseError(f"unknown case fields: {set(case) - FIELDS if isinstance(case, dict) else case!r}")
    cid = case.get("id")
    if not isinstance(cid, str) or not cid:
        raise CaseError("a case needs a nonempty id")
    if case.get("component") not in ("qsa", "ple") or case.get("mode") not in ("prefill", "decode", "spec"):
        raise CaseError(f"{cid}: unknown component or mode")
    lengths, histories = case.get("lengths"), case.get("histories")
    lo, hi = b["requests"]
    if not isinstance(lengths, list) or not lo <= len(lengths) <= hi or \
            any(type(n) is not int or n < 1 for n in lengths):
        raise CaseError(f"{cid}: lengths must describe {lo}..{hi} requests of positive token counts")
    if not isinstance(histories, list) or len(histories) != len(lengths) or \
            any(type(h) is not int or h < 0 for h in histories):
        raise CaseError(f"{cid}: one nonnegative history length per request")
    steps = case.get("steps", 1)
    s_lo, s_hi = b["measured_steps"]
    if type(steps) is not int or not s_lo <= steps <= s_hi:
        raise CaseError(f"{cid}: measured steps must be {s_lo}..{s_hi}")
    mode = case["mode"]
    if mode == "prefill":
        if steps != 1:
            raise CaseError(f"{cid}: a prefill case is one measured step")
        if sum(lengths) > b["max_active_prefill_tokens_per_step"]:
            raise CaseError(f"{cid}: {sum(lengths)} active prefill tokens exceed the per-step bound")
        if any(h + n > b["max_prefill_history_plus_new_tokens_per_request"] for h, n in zip(histories, lengths)):
            raise CaseError(f"{cid}: prefill history plus new tokens exceed the per-request bound")
    else:
        width = 1 if mode == "decode" else 1 + b["speculative_draft_tokens"]
        if set(lengths) != {width}:
            raise CaseError(f"{cid}: {mode} evaluates {width} positions per request per step")
        if width > b["max_target_positions_per_request_per_step"]:
            raise CaseError(f"{cid}: too many target positions per step")
        if any(h < 1 or h > b["max_decode_history_tokens_at_start"] for h in histories):
            raise CaseError(f"{cid}: decode histories must be 1..{b['max_decode_history_tokens_at_start']} at the start")
    layer = case.get("layer", 0)
    if type(layer) is not int or not 0 <= layer < (QSA_LAYERS if case["component"] == "qsa" else 1):
        raise CaseError(f"{cid}: invalid layer index")
    acceptance = case.get("acceptance")
    if mode == "spec":
        if acceptance != "native" and (not isinstance(acceptance, list) or not acceptance or any(
                type(k) is not int or not 0 <= k <= b["speculative_draft_tokens"] for k in acceptance)):
            raise CaseError(f"{cid}: speculative cases take \"native\" acceptance or accepted draft counts 0..3")
    elif acceptance is not None:
        raise CaseError(f"{cid}: only speculative cases carry acceptance")
    timed = case.get("timed", True)
    if type(timed) is not bool:
        raise CaseError(f"{cid}: timed is true or false")
    joining = case.get("joining")
    if joining is not None:
        if mode != "spec":
            raise CaseError(f"{cid}: only speculative cases carry joining prefills")
        if timed:
            raise CaseError(f"{cid}: a joining case is untimed")
        if not isinstance(joining, list) or not joining or any(
                not isinstance(j, list) or len(j) != 2 or any(type(x) is not int for x in j) for j in joining):
            raise CaseError(f"{cid}: joining is a nonempty list of [step, tokens] pairs")
        at = [j[0] for j in joining]
        if at != sorted(set(at)) or at[0] < 0 or at[-1] >= steps:
            raise CaseError(f"{cid}: joining steps must be distinct, increasing and within the measured steps")
        if len(lengths) + 1 > hi:
            raise CaseError(f"{cid}: a joining prefill exceeds the request bound")
        width = 1 + b["speculative_draft_tokens"]
        for step, n in joining:
            if n < 2 or n > b["max_prefill_history_plus_new_tokens_per_request"] or \
                    n + width * len(lengths) > b["max_active_prefill_tokens_per_step"]:
                raise CaseError(f"{cid}: a joining prefill of {n} tokens at step {step} is outside the envelope")
    return case


def tokens(case: dict, pool: list[list[int]], *, tag: str) -> dict:
    """Prompts (history plus prefill tokens) and planned streams for a case's requests, named
    `<tag>-<case>-<index>`, drawn from `pool` (token sequences, cycled), deterministic in the case's
    id and seed. Planned acceptance rotates per request."""
    if not pool or any(not seq for seq in pool):
        raise CaseError("the token pool needs nonempty sequences")
    rng = random.Random(f"{case['id']}/{case.get('seed', 0)}")
    steps = case.get("steps", 1)
    prompts, streams = {}, {}
    for index, (n, h) in enumerate(zip(case["lengths"], case["histories"])):
        prompt_len = h + (n if case["mode"] == "prefill" else 0)
        need = prompt_len + steps * 4 + 8
        sequence = rng.randrange(len(pool))
        offset = rng.randrange(len(pool[sequence]))
        out = []
        while len(out) < need:
            chunk = pool[sequence][offset:offset + need - len(out)]
            out.extend(chunk)
            sequence, offset = (sequence + 1) % len(pool), 0
        req = f"{tag}-{case['id']}-{index}"
        prompts[req], streams[req] = out[:prompt_len], out[prompt_len:]
    acceptance = case.get("acceptance")
    if isinstance(acceptance, list):
        planned = {req: acceptance[i % len(acceptance):] + acceptance[:i % len(acceptance)]
                   for i, req in enumerate(prompts)}
        case = dict(case, acceptance=planned)
    # Joining prefills: `<tag>-<case>-join-<k>`, a prompt of the joining length and a planned stream
    # of which only the first token is emitted (the request finishes after its prefill step).
    for k, (_, n) in enumerate(case.get("joining") or []):
        sequence = rng.randrange(len(pool))
        offset = rng.randrange(len(pool[sequence]))
        out = []
        while len(out) < n + 8:
            out.extend(pool[sequence][offset:offset + n + 8 - len(out)])
            sequence, offset = (sequence + 1) % len(pool), 0
        req = f"{tag}-{case['id']}-join-{k}"
        prompts[req], streams[req] = out[:n], out[n:]
    return {"case": case, "prompts": prompts, "streams": streams}
