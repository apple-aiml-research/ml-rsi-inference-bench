#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Structural validation of served observations, before any statistic.

Every API, speculative, prefix, fresh-state and fidelity observation passes
through these functions before it is compared, aggregated or retried. A
missing, truncated, misaligned or non-finite record is a structural failure
(``ObservationError``): it never becomes an apparent agreement, a partial
statistic or a numerical disagreement, and it never earns a numerical reread.

Records are normalized to integer token IDs. Requests therefore set
``return_tokens_as_token_ids`` (completions and chat) so that every emitted,
top-k and prompt token is an unambiguous ``token_id:<n>`` within the
checkpoint vocabulary. Log probabilities are finite and at most
``LOGPROB_EPSILON`` (serialization rounding). A greedy choice equals the
maximum of its reported distribution within ``TIE_ATOL``; a chosen token that
ties the maximum need not appear in a truncated top-k map, but when it does
appear its two scores agree.

k3 is ``expm1(r) - r`` for a log ratio ``r``. When finite input scores make
the exponential overflow, the value is a numerical ``+infinity`` verdict,
serialized as ``{"value": null, "overflow": true}`` (standard JSON).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

LOGPROB_EPSILON = 1e-6
TIE_ATOL = 1e-9
ENTRY_ATOL = 1e-6
TOKEN = re.compile(r"token_id:(0|[1-9][0-9]*)")


class ObservationError(ValueError):
    """Missing, failed or malformed candidate observation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservationError(message)


def vocabulary_size(model_dir: str | Path = "/model") -> int:
    """The checkpoint tokenizer's vocabulary, added tokens included."""
    from tokenizers import Tokenizer

    size = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")).get_vocab_size(
        with_added_tokens=True)
    require(type(size) is int and size > 0, "the checkpoint tokenizer reports no vocabulary")
    return size


_EOS: dict[str, frozenset] = {}


def checkpoint_eos(model_dir: str | Path = "/model") -> frozenset:
    """The checkpoint's end-of-sequence token IDs (generation and text configurations)."""
    key = str(model_dir)
    if key not in _EOS:
        ids: set[int] = set()
        for name in ("generation_config.json", "config.json"):
            path = Path(model_dir) / name
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            for block in (data, data.get("text_config") or {}):
                value = block.get("eos_token_id")
                ids.update(value if isinstance(value, list) else [] if value is None else [value])
        _EOS[key] = frozenset(int(i) for i in ids)
    return _EOS[key]


def token_id(value, context: str, *, vocabulary: int) -> int:
    """A ``token_id:<n>`` string (or an integer) naming a checkpoint token."""
    require(type(vocabulary) is int and vocabulary > 0, "vocabulary size must be positive")
    if type(value) is int:
        number = value
    else:
        require(isinstance(value, str) and TOKEN.fullmatch(value) is not None,
                f"{context}: expected token_id:<integer>, got {value!r}")
        number = int(value.split(":", 1)[1])
    require(0 <= number < vocabulary, f"{context}: token {number} is outside the checkpoint vocabulary")
    return number


def logprob(value, context: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value),
            f"{context}: expected a finite log probability, got {value!r}")
    require(value <= LOGPROB_EPSILON, f"{context}: a log probability cannot exceed zero: {value}")
    return float(value)


def count(value, context: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{context}: expected an integer >= {minimum}, got {value!r}")
    return value


def distribution(mapping, context: str, *, top_k: int, vocabulary: int) -> dict[int, float]:
    """A reported top-k map with at least ``top_k`` distinct valid entries and mass at most one."""
    require(isinstance(mapping, Mapping), f"{context}: missing top-{top_k} distribution")
    result: dict[int, float] = {}
    for key, value in mapping.items():
        number = token_id(key, f"{context}/top", vocabulary=vocabulary)
        require(number not in result, f"{context}: duplicate top entry {number}")
        result[number] = logprob(value, f"{context}/{number}")
    require(len(result) >= top_k, f"{context}: incomplete top-{top_k} distribution ({len(result)} entries)")
    require(math.fsum(math.exp(v) for v in result.values()) <= 1 + 1e-5,
            f"{context}: top probabilities have mass greater than one")
    return result


def position(chosen: int, score: float, top: Mapping[int, float], context: str, *, greedy: bool) -> None:
    """The chosen score agrees with its entry; a greedy choice is a maximum within the tie allowance."""
    if chosen in top:
        require(math.isclose(score, top[chosen], rel_tol=0, abs_tol=ENTRY_ATOL),
                f"{context}: chosen-token score {score} disagrees with its top entry {top[chosen]}")
    if greedy:
        best = max(top.values())
        require(score >= best - TIE_ATOL,
                f"{context}: greedy choice {chosen} scores {score}, below the reported maximum {best}")


def among_top(token: int, top: Mapping[int, float], k: int = 5) -> bool:
    """Top-k membership, counting equal scores at the cutoff as members."""
    if token not in top:
        return False
    cutoff = sorted(top.values(), reverse=True)[k - 1]
    return top[token] >= cutoff - TIE_ATOL


def _usage(usage, context: str, *, completion_tokens: int, prompt_tokens: int | None) -> dict:
    require(isinstance(usage, Mapping), f"{context}: missing usage")
    prompt = count(usage.get("prompt_tokens"), f"{context}/usage.prompt_tokens", minimum=1)
    completion = count(usage.get("completion_tokens"), f"{context}/usage.completion_tokens")
    total = count(usage.get("total_tokens"), f"{context}/usage.total_tokens", minimum=1)
    require(completion == completion_tokens,
            f"{context}: usage reports {completion} completion tokens, {completion_tokens} observed")
    require(total == prompt + completion, f"{context}: usage total {total} != {prompt} + {completion}")
    if prompt_tokens is not None:
        require(prompt == prompt_tokens, f"{context}: usage reports {prompt} prompt tokens, {prompt_tokens} sent")
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


MISSING = object()


def _termination(context: str, *, n: int, requested: int, reason, ignore_eos: bool,
                 tokens: Sequence[int], eos_token_ids: Iterable[int], stop_reason, stops: Sequence) -> None:
    """Length and termination against the pinned engine's documented fields.

    ``finish_reason`` is ``length`` or ``stop``; with ``stop``, ``stop_reason`` is the witness: None
    for the primary end-of-sequence token, the token ID for one of the checkpoint's other
    end-of-sequence tokens (the engine stops on them as implicit stop tokens), otherwise the matched
    member of the request's ``stop`` list. The pinned engine emits the end-of-sequence token as the
    last generated token and counts it in usage, while chat content omits its text (measured on the
    reference); a stop before the requested length needs no token witness beyond ``stop_reason``."""
    require(n <= requested, f"{context}: {n} tokens exceed the requested {requested}")
    if ignore_eos:
        require(n == requested and reason == "length",
                f"{context}: a fixed-length draw requires {requested} tokens and length termination; got {n}/{reason!r}")
        return
    require(reason in ("length", "stop"), f"{context}: invalid finish reason {reason!r}")
    if reason == "length":
        require(n == requested, f"{context}: length termination after {n} of {requested} tokens")
        return
    require(stop_reason is not MISSING, f"{context}: stop termination without the documented stop_reason field")
    eos = set(checkpoint_eos() if eos_token_ids is None else eos_token_ids)
    if stop_reason is None or (type(stop_reason) is int and stop_reason in eos):
        require(n < requested or (bool(tokens) and tokens[-1] in eos),
                f"{context}: an end-of-sequence stop at the full requested length without its token")
        if stop_reason is not None:
            require(not tokens or tokens[-1] == stop_reason or n < requested,
                    f"{context}: stop_reason {stop_reason} differs from the last generated token")
    else:
        require(stop_reason in list(stops), f"{context}: stop_reason {stop_reason!r} is not a requested stop")


def normalize(context: str, tokens, scores, tops, *, requested: int, reason, usage,
              ignore_eos: bool, top_k: int, vocabulary: int, greedy: bool,
              prompt_tokens: int | None = None, eos_token_ids: Iterable[int] | None = None,
              stop_reason=MISSING, stops: Sequence = ()) -> dict:
    """Validate aligned token/score/top-k arrays and termination; return integer-keyed records."""
    require(type(requested) is int and requested > 0, f"{context}: requested length must be positive")
    require(type(top_k) is int and top_k > 0, f"{context}: top_k must be positive")
    arrays = (tokens, scores, tops)
    require(all(isinstance(a, list) for a in arrays), f"{context}: missing token/probability arrays")
    n = len(tokens)
    require(n > 0 and len(scores) == n and len(tops) == n,
            f"{context}: misaligned observations: lengths {[len(a) for a in arrays]}")
    ids = [token_id(t, f"{context}/{i}", vocabulary=vocabulary) for i, t in enumerate(tokens)]
    values = [logprob(s, f"{context}/{i}/chosen") for i, s in enumerate(scores)]
    maps = [distribution(t, f"{context}/{i}", top_k=top_k, vocabulary=vocabulary) for i, t in enumerate(tops)]
    for i in range(n):
        position(ids[i], values[i], maps[i], f"{context}/{i}", greedy=greedy)
    _termination(context, n=n, requested=requested, reason=reason, ignore_eos=ignore_eos, tokens=ids,
                 eos_token_ids=eos_token_ids, stop_reason=stop_reason, stops=stops)
    return {"id": context, "tokens": ids, "token_logprobs": values, "top_logprobs": maps,
            "finish_reason": reason, "stop_reason": None if stop_reason is MISSING else stop_reason,
            "usage": _usage(usage, context, completion_tokens=n, prompt_tokens=prompt_tokens)}


def completion(response, context: str, *, requested: int, ignore_eos: bool, vocabulary: int,
               top_k: int = 5, greedy: bool = True, prompt_tokens: int | None = None,
               eos_token_ids: Iterable[int] | None = None, stops: Sequence = ()) -> dict:
    """One ``/v1/completions`` response with exactly choice 0 and complete log probabilities."""
    require(isinstance(response, Mapping), f"{context}: expected a response object")
    require(not response.get("error"), f"{context}: request failed: {response.get('error')}")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], Mapping)
            and choices[0].get("index") == 0, f"{context}: expected exactly completion choice 0")
    choice = choices[0]
    lp = choice.get("logprobs")
    require(isinstance(lp, Mapping), f"{context}: missing log probabilities")
    return normalize(context, lp.get("tokens"), lp.get("token_logprobs"), lp.get("top_logprobs"),
                     requested=requested, reason=choice.get("finish_reason"), usage=response.get("usage"),
                     ignore_eos=ignore_eos, top_k=top_k, vocabulary=vocabulary, greedy=greedy,
                     prompt_tokens=prompt_tokens, eos_token_ids=eos_token_ids,
                     stop_reason=choice.get("stop_reason", MISSING), stops=stops)


def _chat_entries(content, context: str):
    require(isinstance(content, list), f"{context}: missing chat log-probability content")
    tokens, scores, tops = [], [], []
    for i, entry in enumerate(content):
        require(isinstance(entry, Mapping), f"{context}/{i}: chat log-probability entry is not an object")
        tokens.append(entry.get("token"))
        scores.append(entry.get("logprob"))
        top = entry.get("top_logprobs")
        require(isinstance(top, list), f"{context}/{i}: missing top log probabilities")
        mapping: dict = {}
        for item in top:
            require(isinstance(item, Mapping), f"{context}/{i}: top entry is not an object")
            key = item.get("token")
            require(key not in mapping, f"{context}/{i}: duplicate top entry {key!r}")
            mapping[key] = item.get("logprob")
        tops.append(mapping)
    return tokens, scores, tops


def chat(response, context: str, *, requested: int, ignore_eos: bool, vocabulary: int,
         top_k: int = 5, greedy: bool = True, eos_token_ids: Iterable[int] | None = None,
         stops: Sequence = ()) -> dict:
    """One ``/v1/chat/completions`` response with exactly choice 0 and complete log probabilities."""
    require(isinstance(response, Mapping), f"{context}: expected a response object")
    require(not response.get("error"), f"{context}: request failed: {response.get('error')}")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], Mapping)
            and choices[0].get("index") == 0, f"{context}: expected exactly chat choice 0")
    choice = choices[0]
    message = choice.get("message")
    require(isinstance(message, Mapping) and isinstance(message.get("content"), str),
            f"{context}: missing chat message content")
    lp = choice.get("logprobs")
    require(isinstance(lp, Mapping), f"{context}: missing chat log probabilities")
    tokens, scores, tops = _chat_entries(lp.get("content"), context)
    row = normalize(context, tokens, scores, tops, requested=requested, reason=choice.get("finish_reason"),
                    usage=response.get("usage"), ignore_eos=ignore_eos, top_k=top_k, vocabulary=vocabulary,
                    greedy=greedy, eos_token_ids=eos_token_ids,
                    stop_reason=choice.get("stop_reason", MISSING), stops=stops)
    row["text"] = message["content"]
    return row


def stream(payloads: Sequence[str], context: str, *, kind: str, requested: int, ignore_eos: bool,
           vocabulary: int, top_k: int = 5, greedy: bool = True, prompt_tokens: int | None = None,
           eos_token_ids: Iterable[int] | None = None, stops: Sequence = ()) -> dict:
    """Assemble a complete SSE response: every ``data:`` payload, a terminal ``[DONE]``, one finish
    record for choice 0, a final usage record, and aligned log probabilities for every token."""
    require(kind in ("completion", "chat"), f"{context}: unknown stream kind {kind!r}")
    require(isinstance(payloads, list) and payloads and payloads[-1] == "[DONE]",
            f"{context}: stream ended without its terminal record")
    tokens, scores, tops, text = [], [], [], []
    reason = usage = None
    stop_reason = MISSING
    for index, raw in enumerate(payloads[:-1]):
        require(isinstance(raw, str) and raw != "[DONE]", f"{context}: record after the terminal record")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise ObservationError(f"{context}: malformed stream record {index}") from exc
        require(isinstance(payload, Mapping) and not payload.get("error"),
                f"{context}: stream record {index} is an error: {payload}")
        if payload.get("usage") is not None:
            require(usage is None, f"{context}: duplicate usage record")
            usage = payload["usage"]
        choices = payload.get("choices")
        require(isinstance(choices, list), f"{context}: stream record {index} has no choices list")
        for choice in choices:
            require(isinstance(choice, Mapping) and choice.get("index") == 0,
                    f"{context}: unexpected streaming choice")
            require(reason is None, f"{context}: content after the finish record")
            lp = choice.get("logprobs")
            if kind == "completion":
                if choice.get("text"):
                    text.append(choice["text"])
                if lp is not None:
                    require(isinstance(lp, Mapping), f"{context}: malformed chunk log probabilities")
                    parts = [lp.get(key) for key in ("tokens", "token_logprobs", "top_logprobs")]
                    require(all(isinstance(p, list) for p in parts) and len({len(p) for p in parts}) == 1,
                            f"{context}: chunk omits token observations")
                    tokens.extend(parts[0]); scores.extend(parts[1]); tops.extend(parts[2])
            else:
                delta = choice.get("delta")
                require(isinstance(delta, Mapping), f"{context}: chat chunk without delta")
                if delta.get("content"):
                    text.append(delta["content"])
                if lp is not None:
                    require(isinstance(lp, Mapping), f"{context}: malformed chunk log probabilities")
                    t, s, m = _chat_entries(lp.get("content") or [], context)
                    tokens.extend(t); scores.extend(s); tops.extend(m)
            if choice.get("finish_reason") is not None:
                reason = choice["finish_reason"]
                stop_reason = choice.get("stop_reason", MISSING)
    require(reason is not None, f"{context}: stream has no finish record")
    require(usage is not None, f"{context}: stream has no final usage record (include_usage)")
    row = normalize(context, tokens, scores, tops, requested=requested, reason=reason, usage=usage,
                    ignore_eos=ignore_eos, top_k=top_k, vocabulary=vocabulary, greedy=greedy,
                    prompt_tokens=prompt_tokens, eos_token_ids=eos_token_ids, stop_reason=stop_reason, stops=stops)
    row["text"] = "".join(text)
    return row


def prompt_scores(response, token_ids: Sequence[int], context: str, *, vocabulary: int) -> list[float]:
    """Own-token scores for positions 1..n-1 of a ``prompt_logprobs=0`` completion.

    The engine's convention is an initial null followed by exactly one entry per scored position,
    keyed by that position's own token; a missing, truncated or extra entry is structural.
    """
    require(isinstance(token_ids, list) and len(token_ids) >= 2, f"{context}: needs at least two prompt tokens")
    for i, t in enumerate(token_ids):
        token_id(t, f"{context}/prompt/{i}", vocabulary=vocabulary)
    require(isinstance(response, Mapping) and not response.get("error"), f"{context}: request failed")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], Mapping)
            and choices[0].get("index") == 0, f"{context}: expected exactly one teacher-forced completion")
    entries = choices[0].get("prompt_logprobs")
    require(isinstance(entries, list) and len(entries) == len(token_ids),
            f"{context}: prompt log probabilities have {len(entries) if isinstance(entries, list) else None} "
            f"entries for {len(token_ids)} tokens")
    require(entries[0] is None, f"{context}: position 0 must be the documented null entry")
    result = []
    for i in range(1, len(token_ids)):
        entry = entries[i]
        require(isinstance(entry, Mapping) and len(entry) == 1,
                f"{context}/{i}: expected exactly one own-token score")
        (key, value), = entry.items()
        require(str(key) == str(token_ids[i]), f"{context}/{i}: entry scores token {key}, not {token_ids[i]}")
        score = value.get("logprob") if isinstance(value, Mapping) else None
        result.append(logprob(score, f"{context}/{i}"))
    return result


def batch(rows: Sequence, expected_ids: Sequence[str], context: str = "batch") -> dict[str, Mapping]:
    """The exact expected unique case set; any failed, missing or extra member is structural."""
    expected = list(expected_ids)
    require(expected and all(isinstance(i, str) and i for i in expected) and len(expected) == len(set(expected)),
            f"{context}: expected case IDs must be nonempty and unique")
    require(isinstance(rows, list) and len(rows) == len(expected),
            f"{context}: {len(rows) if isinstance(rows, list) else None} records for {len(expected)} cases")
    out: dict[str, Mapping] = {}
    for row in rows:
        require(isinstance(row, Mapping), f"{context}: case is not an object")
        cid = row.get("id")
        require(isinstance(cid, str) and cid in expected and cid not in out,
                f"{context}: unexpected or duplicate case ID {cid!r}")
        require(not row.get("error"), f"{context}/{cid}: request failed: {row.get('error')}")
        out[cid] = row
    require(set(out) == set(expected), f"{context}: incomplete case inventory")
    return out


def revalidate(row: Mapping, requested: int, *, vocabulary: int, ignore_eos: bool = True,
               top_k: int = 5, greedy: bool = True) -> dict:
    """Re-check a stored normalized record (integer tokens) exactly as when it was observed, and return
    it with its arrays as validated: integer tokens and integer-keyed top-k maps (JSON object keys are
    strings, so a record read back from a file has string keys until it is revalidated)."""
    require(isinstance(row, Mapping), "record is not an object")
    tokens, tops = row.get("tokens"), row.get("top_logprobs")
    require(isinstance(tokens, list) and isinstance(tops, list), f"{row.get('id')}: missing arrays")
    require(all(isinstance(m, Mapping) for m in tops), f"{row.get('id')}: missing top-k maps")
    checked = normalize(row.get("id", "record"),
                        [token_id(t, f"{row.get('id')}/token", vocabulary=vocabulary) for t in tokens],
                        row.get("token_logprobs"),
                        [{int(k): v for k, v in m.items()} for m in tops],
                        requested=requested, reason=row.get("finish_reason"), usage=row.get("usage"),
                        ignore_eos=ignore_eos, top_k=top_k, vocabulary=vocabulary, greedy=greedy)
    return {**row, **{key: checked[key] for key in ("tokens", "token_logprobs", "top_logprobs")}}


def validate_series(series: Mapping, expected_ids: Sequence[str], requested: int, *,
                    vocabulary: int) -> dict[str, Mapping]:
    """A fixed-length greedy series: its protocol fields and every member revalidated; the members are
    returned as revalidated, so every comparison reads one representation."""
    require(isinstance(series, Mapping), "series: expected an object")
    require(series.get("max_tokens") == requested and series.get("ignore_eos") is True,
            "series: wrong fixed-length request protocol")
    rows = batch(series.get("cases"), expected_ids, "series")
    return {cid: revalidate(row, requested, vocabulary=vocabulary) for cid, row in rows.items()}


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """The first position where two equal-length continuations differ; later positions have
    different histories and are never compared."""
    require(len(a) == len(b), "continuations of different lengths are not comparable")
    return next((i for i in range(len(a)) if a[i] != b[i]), None)


def near_tie(a: Mapping, b: Mapping, index: int) -> bool:
    """At the first divergence, each side's token is among the other's top five."""
    return (among_top(a["tokens"][index], b["top_logprobs"][index])
            and among_top(b["tokens"][index], a["top_logprobs"][index]))


def compare(a: Mapping, b: Mapping, expected_ids: Sequence[str], requested: int, *,
            vocabulary: int, recipes=("ordinary", "mtp3")) -> list[dict]:
    """Ordinary against MTP3 series: identical evaluation/source/workload/tokenizer identity, then the
    first divergence only."""
    left = validate_series(a, expected_ids, requested, vocabulary=vocabulary)
    right = validate_series(b, expected_ids, requested, vocabulary=vocabulary)
    for field in ("evaluation_id", "source_identity", "workload_identity", "tokenizer_identity"):
        require(isinstance(a.get(field), str) and bool(a[field]) and a[field] == b.get(field),
                f"series: missing or mismatched {field}")
    require((a.get("recipe"), b.get("recipe")) == tuple(recipes),
            f"series: comparison needs {recipes[0]} then {recipes[1]} observations")
    out = []
    for cid in expected_ids:
        ca, cb = left[cid], right[cid]
        div = first_divergence(ca["tokens"], cb["tokens"])
        row = {"id": cid, "first_divergence": div, "identical": div is None, "rule_holds": div is None}
        if div is not None:
            row.update(token_a=ca["tokens"][div], token_b=cb["tokens"][div], rule_holds=near_tie(ca, cb, div))
        out.append(row)
    return out


def k3(log_ratio: float) -> dict:
    """``expm1(r) - r``; exponent overflow is an explicit numerical infinity."""
    require(type(log_ratio) in (int, float) and math.isfinite(log_ratio),
            f"k3 needs a finite log ratio, got {log_ratio!r}")
    try:
        return {"value": math.expm1(log_ratio) - log_ratio, "overflow": False}
    except OverflowError:
        return {"value": None, "overflow": True}


def k3_value(result: Mapping) -> float:
    """The numerical value of a k3 record, ``inf`` for an overflow verdict."""
    return math.inf if result["overflow"] else result["value"]


def k3_mean(reference: Sequence[float], served: Sequence[float], context: str) -> dict:
    """Mean k3 over aligned positions of two complete score arrays (reference minus served)."""
    require(len(reference) == len(served) and len(served) > 0,
            f"{context}: {len(served)} served scores for {len(reference)} reference positions")
    values = [k3(logprob(r, f"{context}/reference/{i}") - logprob(s, f"{context}/served/{i}"))
              for i, (r, s) in enumerate(zip(reference, served))]
    overflow = [i for i, v in enumerate(values) if v["overflow"]]
    finite = [v["value"] for v in values if not v["overflow"]]
    mean = None if overflow else math.fsum(v / len(values) for v in finite)
    return {"positions": len(values), "overflow": bool(overflow), "overflow_positions": overflow[:64],
            "mean": mean, "max": None if overflow else max(finite), "values": values}


def json_number(value: float) -> dict:
    """Standard-JSON serialization of a possibly infinite verdict."""
    return {"value": None, "overflow": True} if math.isinf(value) else {"value": value, "overflow": False}


COUNTERS = {
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
}


def parse_spec_counters(text: str) -> dict[str, float]:
    """Required counters, summed over label sets; absent, nonfinite or malformed samples fail."""
    require(isinstance(text, str), "metrics text is missing")
    totals = dict.fromkeys(COUNTERS.values(), 0.0)
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([^\s{]+)(?:\{.*\})?\s+(\S+)(?:\s+\S+)?$", line)
        if not match or match[1] not in COUNTERS:
            require(not any(line.startswith(name) for name in COUNTERS), "malformed speculative counter")
            continue
        key = COUNTERS[match[1]]
        try:
            value = float(match[2])
        except ValueError as exc:
            raise ObservationError(f"{key}: invalid counter value") from exc
        require(math.isfinite(value) and value >= 0 and value.is_integer(), f"{key}: invalid count {value}")
        totals[key] += value
        seen.add(key)
    require(seen == set(totals), f"missing speculative counters: {sorted(set(totals) - seen)}")
    return totals


def counter_delta(before: Mapping, after: Mapping, *, completion_tokens: int, draft_length: int = 3) -> dict:
    """An isolated interval's draft counters: finite, monotone, positive activity and consistent with
    the configured draft length. Validity does not depend on the acceptance rate; an unavailable
    counter is a structural failure, never a zero."""
    delta = {}
    for key in COUNTERS.values():
        for values in (before, after):
            v = values.get(key) if isinstance(values, Mapping) else None
            require(type(v) in (int, float) and math.isfinite(v) and v >= 0 and int(v) == v,
                    f"{key}: missing or invalid counter")
        delta[key] = after[key] - before[key]
        require(delta[key] >= 0, f"{key}: counter decreased during the interval")
    require(type(completion_tokens) is int and completion_tokens > 0, "no complete observed output")
    require(delta["drafts"] > 0 and delta["draft_tokens"] > 0, "no draft activity")
    require(delta["drafts"] <= delta["draft_tokens"] <= draft_length * delta["drafts"],
            "draft counts are inconsistent with the declared draft length")
    require(delta["accepted_tokens"] <= delta["draft_tokens"], "more accepted tokens than proposed tokens")
    return {**delta, "before": dict(before), "after": dict(after), "completion_tokens": completion_tokens,
            "acceptance_rate": delta["accepted_tokens"] / delta["draft_tokens"],
            "mean_acceptance_length": 1 + delta["accepted_tokens"] / delta["drafts"]}
