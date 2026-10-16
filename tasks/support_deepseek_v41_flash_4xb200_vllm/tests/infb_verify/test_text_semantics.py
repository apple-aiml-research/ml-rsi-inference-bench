#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Served text semantics through the ordinary TP4 server: persistent text-history fidelity, and chat, streaming and
request isolation. Both run against the session's one server (`conftest.py`), through the
engine's existing request and log-probability interfaces, with the grading copies of the public protocol code.

**`test_text_history_fidelity`.** Four private records (`/tests/hidden/text_history.json`): one prose and one code
text at each total input length 8,192 and 20,480 tokens, whose last 512 tokens are a continuation the model-author
implementation generated, with the author's teacher-forced log-probability of each continuation token (provenance in
the file). For each record, three modes, each compared at the 512 continuation positions with the existing per-sequence
discrepancy (the mean of k3 = exp(lr) - 1 - lr over the positions):

  * `fresh`: the whole record teacher-forced (prompt log-probabilities; such a request never reads the prefix cache),
    against the author's log-probabilities;
  * `after_unrelated`: the same request again, immediately after an unrelated generation request on the same server;
  * `self_consistency`: 512 tokens generated natively from the record's prompt (greedy, ignoring EOS), their emitted
    log-probabilities against the server's own teacher-forced rescoring of exactly that generated sequence -- decode
    against prefill, not new author truth.

Every reply is parsed structurally first (the one typed parser of `fidelity/check.py`; the generation's token ids and
log-probabilities by count and type), and a structurally invalid reply fails the node before any statistic. Each mode's
median over the four records must be at most the 20,480-token band's existing fidelity bar (config.json
`infb.bars.fidelity_T_per_band["20480"]`). No retry, no re-draw. The record of every reading is
written before the verdict.

**`test_chat_streaming_and_isolation`.** Ordinary chat; the same request streamed; a second, tool-free conversation
turn; and two teacher-forced completion requests sent concurrently, then each alone, at deterministic sampling. It
checks API, error and finish framing and requested-token log-probabilities with the grading copy of
`tests/api_examples.py`, that the stream assembles to the non-streamed reply of the same deterministic request, that
the second turn's prompt carries the first turn, and that each concurrent request's prompt log-probabilities agree
with its solo reading within the existing fidelity allowance of its length band (the mean k3 of solo against
concurrent). Exact wording is never required, and a nonempty reply alone is never taken as isolation.

Concurrency is established inside the engine, not only on the wire. A teacher-forced reading is one prefill step, and
an idle engine starts a step on the first request to arrive, so two readings posted together to it almost never share
a step (measured against a defect that corrupts exactly that case: 0 of 8, and 0 of 8 as one request
carrying both prompts); while another request is running they are taken into one step 7 times in 8. So the pair is
sent PAIR_ROUNDS times while a companion runs: a streamed completion generating a fixed number of tokens (greedy, EOS
ignored) from a fixed private prompt. The engine schedules a running request in every step, so a companion that had
streamed events before the first round and still streams events after the last was part of every step in between,
each reading's prefill included. The solo readings are taken first, with nothing else running; each round's readings
must agree with them within the existing allowances. A round whose reading is over its band's allowance is drawn once
more, whole, under the same companion, and
fails only if its re-draw is over too; a structurally invalid reading is terminal, never re-drawn. A companion that
fails as a request is a candidate failure. If the companion did not
demonstrably run through the rounds, the node has not exercised the property it claims: that is the verifier's fault,
never a pass and never a candidate failure, unless a check already failed for the candidate.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from infb_verify import _trusted, serve

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
X = json.loads((TESTS_DIR / "config.json").read_text())["infb"]
BARS = {int(b): float(v) for b, v in X["bars"]["fidelity_T_per_band"].items()}
SEM = X["text_semantics"]
RECORD_DIR = Path(os.environ.get("INFB_SEMANTICS_RECORD_DIR", "/logs/verifier/semantics"))

_FIDELITY = _trusted.load("fidelity/check.py")
_API = _trusted.load("tests/api_examples.py")

# The concurrent pair's companion: a streamed completion of COMPANION_TOKENS tokens from the first COMPANION_PROMPT ids
# of the first concurrent record. The pair is sent PAIR_ROUNDS times while it runs (a round over its allowance once
# more); co-residency is shown when it streamed LEAD_EVENTS events before the first round was sent and LEAD_EVENTS
# after the last round, a re-draw included, returned.
COMPANION_PROMPT, COMPANION_TOKENS, LEAD_EVENTS, PAIR_ROUNDS = 256, 1024, 3, 4


def _persist(name: str, record: dict) -> None:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    (RECORD_DIR / f"{name}.json").write_text(json.dumps(record, indent=1, default=str))


def _completions(server) -> str:
    return server.url_for("v1/completions")


def _teacher_forced(server, token_ids: list[int], positions: range) -> dict:
    """One teacher-forced reading of `token_ids`, parsed by the one typed parser; the served log-probability at each
    position in `positions` (indices into `token_ids`) or the structural errors."""
    t0 = time.monotonic()
    status, body, transport = _FIDELITY.request_prompt_logprobs(_completions(server), X["model_id"], token_ids)
    errors = [f"the request failed: {transport}"] if transport else ([] if status == 200 else [f"HTTP {status}"])
    values = None
    if not errors:
        values, parse_errors, _ = _FIDELITY.parse_prompt_logprobs(body, token_ids)
        errors += parse_errors
    served = [values[p - 1] for p in positions] if values is not None else None
    return {"errors": errors, "served": served, "seconds": round(time.monotonic() - t0, 3),
            "transport_failed": bool(transport)}


def _generate(server, prompt_ids: list[int], n: int) -> dict:
    """`n` tokens generated natively from `prompt_ids` (greedy, EOS ignored) with each emitted token's log-probability."""
    body = {"model": X["model_id"], "prompt": prompt_ids, "max_tokens": n, "temperature": 0.0, "seed": 0,
            "ignore_eos": True, "logprobs": 0, "return_token_ids": True, "skip_special_tokens": False}
    status, reply, transport = _API.post(server.url_root, "/v1/completions", body)
    if transport:
        return {"errors": [f"the request failed: {transport}"], "transport_failed": True}
    errors, facts = _API.check_completion(status, reply, logprobs=True, token_ids=True)
    ids, lps = facts.get("token_ids"), facts.get("token_logprobs")
    if not errors and (len(ids) != n or facts.get("finish_reason") != "length"):
        errors.append(f"{len(ids)} tokens generated with finish {facts.get('finish_reason')!r}, not {n} by length")
    return {"errors": errors, "token_ids": ids, "logprobs": lps, "transport_failed": False}


def _k3_mean(reference: list[float], served: list[float]) -> float:
    return _FIDELITY.score(reference, served)["mean"]


def test_text_history_fidelity(server):
    """Each of the three history modes' median k3 over the four records is at most the 20,480-token band's bar."""
    cfg = SEM["history"]
    data = json.loads((TESTS_DIR / cfg["data"]).read_text())
    records = data["records"]
    bar = BARS[int(cfg["bar_band"])]
    record: dict = {"node": "test_text_history_fidelity", "bar": bar, "bar_band": cfg["bar_band"],
                    "provenance": data.get("provenance"), "records": []}
    faults = [f"{r.get('id')}: {', '.join(e)}" for r in records if (e := _validate_history_record(r))]
    if faults or len(records) != 4:
        record.update(status="reference_fault", errors=faults or [f"{len(records)} records, not 4"])
        _persist("test_text_history_fidelity", record)
        pytest.fail("reference_fault: " + "; ".join(record["errors"]))
    if why := serve.down(server):
        record.update(status="blocked_by:server_unreachable", blocked=why)
        _persist("test_text_history_fidelity", record)
        pytest.fail(serve.blocked_message(why))
    unrelated = cfg["unrelated_request"]
    for r in records:
        ids, p0 = r["token_ids"], r["prompt_len"]
        positions = range(p0, len(ids))
        row: dict = {"id": r["id"], "family": r["family"], "total": len(ids), "prompt_len": p0, "modes": {}}
        fresh = _teacher_forced(server, ids, positions)
        row["modes"]["fresh"] = fresh
        u_status, _, u_transport = _API.post(server.url_root, "/v1/completions", dict(unrelated, model=X["model_id"]))
        row["unrelated"] = {"status": u_status, "transport": u_transport}
        again = _teacher_forced(server, ids, positions)
        row["modes"]["after_unrelated"] = again
        gen = _generate(server, ids[:p0], int(cfg["generate_tokens"]))
        consistency: dict = {"generation_errors": gen["errors"], "errors": list(gen["errors"]),
                             "transport_failed": gen.get("transport_failed", False)}
        if not gen["errors"]:
            rescored = _teacher_forced(server, ids[:p0] + gen["token_ids"], range(p0, p0 + len(gen["token_ids"])))
            consistency.update(errors=rescored["errors"], rescored=rescored["served"], emitted=gen["logprobs"],
                               generated_ids=gen["token_ids"], transport_failed=rescored["transport_failed"])
        row["modes"]["self_consistency"] = consistency
        for mode, obs in row["modes"].items():
            if obs["errors"]:
                obs["k3_mean"] = None
            elif mode == "self_consistency":
                obs["k3_mean"] = _k3_mean(obs["rescored"], obs["emitted"])  # prefill as reference, decode as served
            else:
                obs["k3_mean"] = _k3_mean(r["reference_logprobs"], obs["served"])
        record["records"].append(row)
    invalid = [f"{row['id']} {mode}: {'; '.join(obs['errors'])}" for row in record["records"]
               for mode, obs in row["modes"].items() if obs["errors"]]
    medians = {}
    if not invalid:
        for mode in ("fresh", "after_unrelated", "self_consistency"):
            medians[mode] = statistics.median(row["modes"][mode]["k3_mean"] for row in record["records"])
    failing = [m for m, v in medians.items() if not v <= bar]  # the passing direction: NaN fails
    status = "pass" if not invalid and not failing else "candidate_fail"
    if invalid and all(obs.get("transport_failed") for row in record["records"] for obs in row["modes"].values()
                       if obs["errors"]):
        record["primary_cause"] = "the server stopped answering during this node"
    record.update(medians=medians, invalid=invalid, failing_modes=failing, status=status)
    _persist("test_text_history_fidelity", record)
    print("INFB_TEXT_HISTORY " + json.dumps({"medians": medians, "bar": bar, "invalid": invalid,
                                             "per_record": {row["id"]: {m: o["k3_mean"] for m, o in row["modes"].items()}
                                                            for row in record["records"]}}))
    assert not invalid, "structurally invalid readings (never read as a discrepancy): " + " | ".join(invalid)
    assert not failing, (
        f"the median k3 over the four history records is over the bar {bar:.6g} in mode(s) "
        + ", ".join(f"{m} ({medians[m]:.6g})" for m in failing)
        + "; per record: " + ", ".join(f"{row['id']} " + "/".join(f"{row['modes'][m]['k3_mean']:.4g}" for m in medians)
                                       for row in record["records"])
    )


def _validate_history_record(r) -> list[str]:
    """A history record's own structure: identity, token ids, the prompt length inside the record, and one finite
    author log-probability per continuation position."""
    if not isinstance(r, dict):
        return ["the record is not an object"]
    errors = [f"no {k!r}" for k in ("id", "family", "token_ids", "prompt_len", "reference_logprobs") if k not in r]
    ids, p0, ref = r.get("token_ids"), r.get("prompt_len"), r.get("reference_logprobs")
    if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0
                                                           for t in ids):
        return errors + ["token_ids is not a list of non-negative integers"]
    if not isinstance(p0, int) or isinstance(p0, bool) or not 0 < p0 < len(ids):
        return errors + ["prompt_len is not inside the record"]
    if not isinstance(ref, list) or len(ref) != len(ids) - p0 or not all(
            _FIDELITY.is_number(v) and v == v and abs(v) != float("inf") for v in ref):
        errors.append(f"reference_logprobs is not one finite number per continuation position ({len(ids) - p0})")
    return errors


def _validate_chat_inputs(inputs) -> list[str]:
    """The private chat data's own structure: two non-empty questions, and two concurrent readings, each an id, a
    length band with a fidelity allowance and at least two non-negative token ids. A fault here is the verifier's."""
    if not isinstance(inputs, dict):
        return ["the chat inputs are not an object"]
    errors = []
    chat = inputs.get("chat")
    if not (isinstance(chat, list) and len(chat) >= 2 and all(isinstance(q, str) and q.strip() for q in chat)):
        errors.append("chat is not two or more non-empty questions")
    pair = inputs.get("concurrent")
    if not (isinstance(pair, list) and len(pair) == 2):
        return errors + ["concurrent is not two readings"]
    for k, seq in enumerate(pair):
        ids = seq.get("token_ids") if isinstance(seq, dict) else None
        if not isinstance(seq, dict) or not seq.get("id") or seq.get("band") not in {str(b) for b in BARS} | set(BARS):
            errors.append(f"concurrent[{k}] has no id or no band with an allowance")
        if not (isinstance(ids, list) and len(ids) >= 2 and all(
                isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in ids)):
            errors.append(f"concurrent[{k}].token_ids is not a list of non-negative integers")
    return errors


def _chat_body(messages: list[dict], cfg: dict) -> dict:
    return {"model": X["model_id"], "messages": messages, "temperature": 0.0, "seed": 0,
            "max_tokens": int(cfg["max_tokens"]), "logprobs": True}


def _parsed(seq: dict, reply, name: str) -> dict:
    """One teacher-forced reading of a concurrent record, parsed by the one typed parser: its values, or its errors."""
    st, body, transport = reply
    if transport or st != 200:
        return {"values": None,
                "errors": [f"{name}: " + (f"the request failed: {transport}" if transport else f"HTTP {st}")]}
    values, errs, _ = _FIDELITY.parse_prompt_logprobs(body, seq["token_ids"])
    return {"values": None if errs else values, "errors": [f"{name}: {e}" for e in errs]}


def _pair_round(pair: list[dict], fetch, solo: list[dict], name: str) -> list[dict]:
    """The pair sent together: each reading's errors and its mean k3 against its solo reading (None when either
    reading is invalid)."""
    with concurrent.futures.ThreadPoolExecutor(len(pair)) as ex:
        replies = list(ex.map(fetch, pair))
    out = []
    for seq, reply, alone in zip(pair, replies, solo):
        reading = _parsed(seq, reply, name)
        k3 = (_k3_mean(alone["values"], reading["values"])
              if alone["values"] is not None and reading["values"] is not None else None)
        out.append({"errors": reading["errors"], "k3": k3})
    return out


class _Companion(threading.Thread):
    """A streamed completion of `n` tokens (greedy, EOS ignored) from `prompt_ids`, run beside the concurrent pair.
    Records the arrival time of every data event, how the stream ended, and its request errors; never raises."""

    def __init__(self, server, prompt_ids: list[int], n: int):
        super().__init__(daemon=True)
        self.body = {"model": X["model_id"], "prompt": prompt_ids, "max_tokens": n, "temperature": 0.0, "seed": 0,
                     "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        self.url, self.n = _completions(server), n
        self.events: list[float] = []
        self.finish, self.done, self.usage, self.errors = None, False, None, []

    def run(self) -> None:
        req = urllib.request.Request(self.url, data=json.dumps(self.body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=_API.REQUEST_TIMEOUT_S) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        self.done = True
                        continue
                    if not line.startswith("data: "):
                        self.errors.append(f"a line that is not a data event: {line[:80]!r}")
                        continue
                    self.events.append(time.monotonic())
                    try:
                        chunk = json.loads(line[len("data: "):])
                    except ValueError:
                        self.errors.append(f"an event that is not JSON: {line[6:86]!r}")
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    self.usage = chunk.get("usage") or self.usage
                    for c in chunk.get("choices") or []:
                        self.finish = c.get("finish_reason") or self.finish
        except urllib.error.HTTPError as exc:
            self.errors.append(f"HTTP {exc.code}")
        except (urllib.error.URLError, OSError) as exc:
            self.errors.append(f"the request failed: {type(exc).__name__}: {exc}"[:300])

    def ended(self, timeout: float) -> list[str]:
        """Wait for the stream's end; the companion's own request errors (a candidate's), [] when it ended by length
        with the requested number of tokens."""
        self.join(timeout)
        if self.is_alive():
            return [f"the companion generation of {self.n} tokens did not end within {timeout:.0f} s"]
        if self.errors:
            return list(self.errors[:3])
        if not self.done or self.finish != "length":
            return [f"the companion stream ended with finish {self.finish!r} and "
                    f"{'a' if self.done else 'no'} terminal [DONE], not by length"]
        got = (self.usage or {}).get("completion_tokens") if isinstance(self.usage, dict) else None
        if got != self.n:
            return [f"the companion stream reports {got!r} completion tokens, not the {self.n} requested"]
        return []


def test_chat_streaming_and_isolation(server):
    """Chat, streamed chat, a second tool-free turn and two concurrent completions repeated alone: framing, stream
    assembly, requested-token log-probabilities and solo/concurrent agreement within the existing allowance."""
    cfg = SEM["chat"]
    inputs = json.loads((TESTS_DIR / cfg["data"]).read_text())
    if faults := _validate_chat_inputs(inputs):
        _persist("test_chat_streaming_and_isolation", {"node": "test_chat_streaming_and_isolation",
                                                       "status": "reference_fault", "errors": faults})
        pytest.fail("reference_fault: " + "; ".join(faults))
    if why := serve.down(server):
        _persist("test_chat_streaming_and_isolation", {"node": "test_chat_streaming_and_isolation",
                                                       "status": "blocked_by:server_unreachable", "blocked": why})
        pytest.fail(serve.blocked_message(why))
    base = server.url_root
    checks: dict = {}

    first = inputs["chat"][0]
    messages = [{"role": "user", "content": first}]
    errors, reply = _API.check_chat(*_API.post(base, "/v1/chat/completions", _chat_body(messages, cfg))[:2], logprobs=True)
    checks["chat"] = {"errors": errors, "reply": reply}

    s = _API.stream(base, "/v1/chat/completions", dict(_chat_body(messages, cfg), stream_options={"include_usage": True}))
    errors, streamed = _API.check_chat_stream(s, logprobs=True)
    if not errors and not checks["chat"]["errors"]:
        if streamed["content"] != reply["content"]:
            errors.append("the streamed pieces do not assemble to the non-streamed reply of the same deterministic request")
        if streamed["tokens"] != reply.get("tokens"):
            errors.append("the streamed tokens are not the non-streamed reply's tokens")
    checks["stream"] = {"errors": errors, "events": streamed["events"], "finish": streamed["finish"]}

    turn2 = messages + [{"role": "assistant", "content": reply.get("content") or ""},
                        {"role": "user", "content": inputs["chat"][1]}]
    errors, second = _API.check_chat(*_API.post(base, "/v1/chat/completions", _chat_body(turn2, cfg))[:2], logprobs=True)
    u1, u2 = reply.get("usage") or {}, second.get("usage") or {}
    if not errors and not checks["chat"]["errors"] and not (
            u2.get("prompt_tokens", 0) >= u1.get("prompt_tokens", 0) + u1.get("completion_tokens", 0)):
        errors.append(f"the second turn's prompt ({u2.get('prompt_tokens')} tokens) does not carry the first turn "
                      f"({u1.get('prompt_tokens')} + {u1.get('completion_tokens')})")
    checks["second_turn"] = {"errors": errors, "reply": second}

    pair = inputs["concurrent"]
    fetch = lambda seq: _FIDELITY.request_prompt_logprobs(_completions(server), X["model_id"], seq["token_ids"])  # noqa: E731
    solo = [_parsed(seq, fetch(seq), "solo") for seq in pair]  # first, with nothing else running
    lead = LEAD_EVENTS
    companion = _Companion(server, pair[0]["token_ids"][:COMPANION_PROMPT], COMPANION_TOKENS)
    companion.start()
    waited = time.monotonic()
    while len(companion.events) < lead and companion.is_alive() and time.monotonic() - waited < _API.REQUEST_TIMEOUT_S:
        time.sleep(0.005)
    sent, before = time.monotonic(), len(companion.events)
    rounds = []  # the pair, PAIR_ROUNDS times while the one companion runs; a round over its allowance once more
    for r in range(PAIR_ROUNDS):
        first = _pair_round(pair, fetch, solo, f"concurrent round {r + 1}")
        valid = not any(reading["errors"] for reading in first)
        over = any(reading["k3"] is not None and not reading["k3"] <= BARS[int(seq["band"])]
                   for reading, seq in zip(first, pair))
        again = _pair_round(pair, fetch, solo, f"concurrent round {r + 1} re-draw") if valid and over else None
        rounds.append((first, again))
    returned = time.monotonic()
    companion_errors = companion.ended(_API.REQUEST_TIMEOUT_S)
    after = sum(1 for t in companion.events if t > returned)
    co_residency = {"events_before_pair": before, "events_after_readings": after, "events": len(companion.events),
                    "lead_events": lead, "rounds": PAIR_ROUNDS, "redraws": sum(again is not None for _, again in rounds),
                    "finish": companion.finish, "pairs_seconds": round(returned - sent, 3),
                    "shown": not companion_errors and before >= lead and after >= lead}
    rows = []
    for k, seq in enumerate(pair):
        allowance = BARS[int(seq["band"])]
        row = {"id": seq["id"], "band": seq["band"], "allowance": allowance, "errors": list(solo[k]["errors"]),
               "k3_by_round": [], "redraw_by_round": []}
        for r, (first, again) in enumerate(rounds):
            got, redo = first[k], (again[k] if again is not None else None)
            row["errors"] += got["errors"] + (redo["errors"] if redo is not None else [])
            row["k3_by_round"].append(got["k3"])
            row["redraw_by_round"].append(redo["k3"] if redo is not None else None)
            if got["k3"] is None or got["k3"] <= allowance or redo is None:
                continue  # agrees; or invalid, or over beside an invalid reading: those failures are terminal
            if redo["k3"] is not None and not redo["k3"] <= allowance:
                row["errors"].append(f"solo and concurrent readings disagree: mean k3 {got['k3']:.6g} over the "
                                     f"{seq['band']}-token band's allowance {allowance:.6g} (round {r + 1}), and "
                                     f"{redo['k3']:.6g} on its re-draw")
        counted = [g if g is None or g <= allowance or d is None else d
                   for g, d in zip(row["k3_by_round"], row["redraw_by_round"])]
        row["k3_mean"] = max((v for v in counted if v is not None), default=None)
        rows.append(row)
    checks["concurrent"] = {"errors": [f"companion: {e}" for e in companion_errors]
                            + [e for r in rows for e in r["errors"]], "rows": rows, "co_residency": co_residency}

    failures = {k: v["errors"] for k, v in checks.items() if v["errors"]}
    unshown = None if failures or co_residency["shown"] else (
        f"verifier_fault: the concurrent readings were not shown to share the engine with a running request (the "
        f"companion streamed {before} events before the first round was sent and {after} after the last returned; "
        f"{lead} needed on each side)")
    _persist("test_chat_streaming_and_isolation", {"node": "test_chat_streaming_and_isolation", "checks": checks,
                                                   "status": "candidate_fail" if failures else
                                                   "verifier_fault" if unshown else "pass"})
    print("INFB_CHAT " + json.dumps({**{k: {"errors": v["errors"]} for k, v in checks.items()},
                                     "co_residency": co_residency}))
    assert not failures, "; ".join(f"{k}: {' | '.join(v)}" for k, v in failures.items())
    if unshown:
        pytest.fail(unshown)
