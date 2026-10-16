#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""R-CHAT: the chat-completions protocols on the class's live server, after its earlier workloads (one node per class)."""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from glm53_protocols.fidelity import _k3_term  # noqa: E402 - the fidelity statistic's own term
from glm53_protocols.limits import HTTP_S  # noqa: E402
from infb_verify.records import compact, write_record  # noqa: E402

MODEL = os.environ.get("INFB_MODEL_DIR", "/model/zai-org/GLM-5.3-Flash")
MAX_TOKENS = 160
TOP_LOGPROBS = 2
CHOSEN_TOLERANCE = 1e-4

CONVERSATIONS = {
    "ordinary": [{"role": "user", "content": "In a few sentences, explain why a B-tree keeps its nodes wide."}],
    "turn": [{"role": "user", "content": "What does a write-ahead log protect against?"},
             {"role": "assistant", "content": "It protects committed changes against crashes: a change is recorded "
                                              "durably in the log before the data files are updated, so recovery can "
                                              "replay or undo it."},
             {"role": "user", "content": "And why does that make group commit cheaper?"}],
    "pair_a": [{"role": "user", "content": "Summarize the idea behind copy-on-write file systems."}],
    "pair_b": [{"role": "user", "content": "Write a short limerick about a patient compiler."}],
}


def _template_ids(tok, messages) -> list[int]:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _entries_problems(entries, n_tokens: int, top_logprobs: int | None = TOP_LOGPROBS) -> list[str]:
    """One finite entry per generated token; with `top_logprobs`, that many alternatives at every position and the
    chosen token the most probable one (the non-streamed answer; the engine's streamed alternatives are not aligned
    position by position at the pinned base, so a stream is checked without them)."""
    out = []
    if not isinstance(entries, list) or len(entries) != n_tokens:
        return [f"{len(entries) if isinstance(entries, list) else type(entries).__name__} log-probability entries for "
                f"{n_tokens} generated tokens"]
    for i, e in enumerate(entries):
        top = e.get("top_logprobs") if isinstance(e, dict) else None
        if not isinstance(e, dict) or not isinstance(e.get("token"), str) or not _finite(e.get("logprob")):
            out.append(f"position {i}: malformed entry {compact(e, 200)}")
        elif top_logprobs is None:
            pass
        elif not isinstance(top, list) or len(top) != TOP_LOGPROBS or not all(_finite(t.get("logprob")) for t in top):
            out.append(f"position {i}: {len(top) if isinstance(top, list) else top!r} top log-probabilities for "
                       f"{top_logprobs} requested")
        elif e["logprob"] < max(t["logprob"] for t in top) - CHOSEN_TOLERANCE:
            out.append(f"position {i}: the chosen token's log-probability {e['logprob']} is below its position's "
                       f"largest {max(t['logprob'] for t in top)} at deterministic sampling")
        if len(out) >= 6:
            break
    return out


def _message_problems(msg) -> list[str]:
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return [f"the message is not an assistant message: {compact(msg, 300)}"]
    content, reasoning = msg.get("content"), msg.get("reasoning_content")
    out = [f"{k} is {type(v).__name__}, not a string" for k, v in (("content", content), ("reasoning_content", reasoning))
           if v is not None and not isinstance(v, str)]
    if not (content or reasoning):
        out.append("neither content nor reasoning_content carries text")
    return out


def _teacher_forced(base_url: str, prompt_ids: list[int], out_ids: list[int]) -> tuple[list[float] | None, str | None]:
    ids = list(prompt_ids) + list(out_ids)
    try:
        r = requests.post(base_url + "/generate", json={
            "input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            "return_logprob": True, "logprob_start_len": 0}, timeout=HTTP_S)
        r.raise_for_status()
        ent = r.json()["meta_info"]["input_token_logprobs"]
    except Exception as exc:  # noqa: BLE001
        return None, f"teacher-forced request failed: {type(exc).__name__}: {str(exc)[:300]}"
    if len(ent) != len(ids) or any(e[1] != t for e, t in zip(ent, ids)):
        return None, "the teacher-forced answer does not cover the requested positions and token ids"
    vals = [e[0] for e in ent[len(prompt_ids):]]
    if not all(_finite(v) for v in vals):
        return None, "the teacher-forced answer holds a value that is not a finite number"
    return vals, None


def _k3(served: list[float], ref: list[float]) -> float:
    return sum(_k3_term(r - s) for r, s in zip(ref, served)) / len(ref)


def _nonstream(base_url: str, messages) -> dict:
    body = {"model": MODEL, "messages": messages, "temperature": 0.0, "max_tokens": MAX_TOKENS, "logprobs": True,
            "top_logprobs": TOP_LOGPROBS, "return_token_ids": True, "return_prompt_token_ids": True}
    t0 = time.time()
    try:
        r = requests.post(base_url + "/v1/chat/completions", json=body, timeout=HTTP_S)
        out = {"http_status": r.status_code, "elapsed_s": round(time.time() - t0, 3)}
        out["body"] = r.json()
    except Exception as exc:  # noqa: BLE001
        out = {"http_status": None, "elapsed_s": round(time.time() - t0, 3),
               "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    return out


def _check_nonstream(base_url: str, tok, messages, reply: dict, bar: float) -> dict:
    problems: list[str] = []
    body = reply.get("body") if isinstance(reply.get("body"), dict) else None
    if reply.get("http_status") != 200 or body is None:
        return {"problems": [f"HTTP {reply.get('http_status')}: {reply.get('error') or compact(reply.get('body'), 400)}"]}
    choices = body.get("choices")
    if body.get("object") != "chat.completion" or not isinstance(choices, list) or len(choices) != 1:
        return {"problems": [f"not one chat.completion choice: object {body.get('object')!r}, choices "
                             f"{len(choices) if isinstance(choices, list) else choices!r}"]}
    ch = choices[0]
    problems += _message_problems(ch.get("message"))
    if ch.get("index") != 0 or ch.get("finish_reason") not in ("stop", "length"):
        problems.append(f"choice index {ch.get('index')!r}, finish reason {ch.get('finish_reason')!r}")
    expected_prompt = _template_ids(tok, messages)
    if ch.get("prompt_token_ids") != expected_prompt:
        problems.append(f"the served prompt ({len(ch.get('prompt_token_ids') or [])} ids) is not the model's chat "
                        f"template of these messages ({len(expected_prompt)} ids)")
    out_ids = ch.get("response_token_ids")
    usage = body.get("usage") or {}
    if not isinstance(out_ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in out_ids) \
            or not out_ids or usage.get("completion_tokens") != len(out_ids):
        problems.append(f"response_token_ids {compact(out_ids, 120)} against usage.completion_tokens "
                        f"{usage.get('completion_tokens')!r}")
        return {"problems": problems}
    entries = ((ch.get("logprobs") or {}).get("content"))
    problems += _entries_problems(entries, len(out_ids))
    if not problems and tok.decode(out_ids) != "".join(e["token"] for e in entries):
        problems.append("the generated token ids and the log-probability entries' tokens differ")
    result: dict = {"problems": problems, "n_prompt": len(expected_prompt), "n_generated": len(out_ids),
                    "finish_reason": ch.get("finish_reason")}
    if problems:
        return result
    tf, err = _teacher_forced(base_url, expected_prompt, out_ids)
    if err:
        result["problems"] = [err]
        return result
    served = [e["logprob"] for e in entries]
    result["k3_vs_teacher_forced"] = _k3(served, tf)
    result["max_abs_diff"] = max(abs(a - b) for a, b in zip(served, tf))
    if not result["k3_vs_teacher_forced"] < bar:
        result["problems"].append(f"generation-time log-probabilities disagree with the same tokens' solo "
                                  f"teacher-forced ones: k3 {result['k3_vs_teacher_forced']:.6g}, not below the bar")
    return result


def _stream(base_url: str, tok, messages, bar: float) -> dict:
    body = {"model": MODEL, "messages": messages, "temperature": 0.0, "max_tokens": MAX_TOKENS, "stream": True,
            "logprobs": True, "stream_options": {"include_usage": True},
            "return_output_ids_in_sglext": True, "return_input_ids_in_sglext": True}
    t0 = time.time()
    problems: list[str] = []
    try:
        r = requests.post(base_url + "/v1/chat/completions", json=body, timeout=HTTP_S, stream=True)
        status = r.status_code
        lines = list(r.iter_lines(decode_unicode=True))
    except Exception as exc:  # noqa: BLE001
        return {"problems": [f"stream failed: {type(exc).__name__}: {str(exc)[:300]}"],
                "elapsed_s": round(time.time() - t0, 3)}
    rec: dict = {"http_status": status, "n_lines": len(lines), "elapsed_s": round(time.time() - t0, 3)}
    if status != 200:
        rec["problems"] = [f"HTTP {status}: {compact(lines[:3], 400)}"]
        return rec
    data = [ln for ln in lines if ln]
    bad = [ln for ln in data if not ln.startswith("data: ")]
    if bad or not data or data[-1] != "data: [DONE]":
        rec["problems"] = [f"SSE framing: {len(bad)} non-data lines, last line {data[-1][:60] if data else None!r}"]
        rec["raw_head"] = compact(lines[:6], 1200)
        return rec
    try:
        chunks = [json.loads(ln[6:]) for ln in data[:-1]]
    except ValueError as exc:
        return {**rec, "problems": [f"a stream chunk is not JSON: {exc}"]}
    ids = {c.get("id") for c in chunks}
    choice_chunks = [c for c in chunks if c.get("choices")]
    if any(c.get("object") != "chat.completion.chunk" for c in chunks) or len(ids) != 1 or not choice_chunks:
        problems.append(f"chunk objects/ids: {sorted({c.get('object') for c in chunks})}, {len(ids)} ids")
    first = (choice_chunks[0]["choices"][0].get("delta") or {}) if choice_chunks else {}
    if first.get("role") != "assistant":
        problems.append(f"the first delta carries role {first.get('role')!r}")
    finishes = [c["choices"][0].get("finish_reason") for c in choice_chunks]
    if not finishes or finishes[-1] not in ("stop", "length") or any(f for f in finishes[:-1]):
        problems.append(f"finish reasons across choice chunks: {[f for f in finishes if f]} (last {finishes[-1:]})")
    usage = [c.get("usage") for c in chunks if c.get("usage")]
    entries = [e for c in choice_chunks for e in (((c["choices"][0].get("logprobs") or {}).get("content")) or [])]
    sg = [c.get("sglext") for c in chunks if c.get("sglext")]
    out_ids = (sg[-1].get("output_ids") or [[]])[0] if sg else None
    in_ids = sg[-1].get("input_ids") if sg else None
    text = "".join(((c["choices"][0].get("delta") or {}).get("content") or "")
                   + ((c["choices"][0].get("delta") or {}).get("reasoning_content") or "") for c in choice_chunks)
    if not usage or usage[-1].get("completion_tokens") != len(entries):
        problems.append(f"usage {compact(usage[-1:], 200)} against {len(entries)} streamed entries")
    if not text:
        problems.append("the stream carried no content or reasoning text")
    expected_prompt = _template_ids(tok, messages)
    if in_ids != expected_prompt:
        problems.append("the streamed request's prompt ids are not the model's chat template of these messages")
    if not isinstance(out_ids, list) or len(out_ids) != len(entries):
        problems.append(f"{len(out_ids) if isinstance(out_ids, list) else out_ids!r} streamed output ids for "
                        f"{len(entries)} entries")
    rec.update(n_chunks=len(chunks), n_generated=len(entries), finish_reason=finishes[-1] if finishes else None)
    if problems:
        return {**rec, "problems": problems}
    problems += _entries_problems(entries, len(out_ids), top_logprobs=None)
    if not problems and tok.decode(out_ids) != "".join(e["token"] for e in entries):
        problems.append("the streamed token ids and the log-probability entries' tokens differ")
    if problems:
        return {**rec, "problems": problems}
    tf, err = _teacher_forced(base_url, expected_prompt, out_ids)
    if err:
        return {**rec, "problems": [err]}
    rec["k3_vs_teacher_forced"] = _k3([e["logprob"] for e in entries], tf)
    if not rec["k3_vs_teacher_forced"] < bar:
        problems.append(f"streamed log-probabilities disagree with the same tokens' solo teacher-forced ones: k3 "
                        f"{rec['k3_vs_teacher_forced']:.6g}, not below the bar")
    return {**rec, "problems": problems}


def chat_protocols(base_url: str, label: str, bar: float) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    steps: dict = {}

    def run(name: str, fn):
        t0 = time.time()
        res = fn()
        res["elapsed_s"] = round(time.time() - t0, 3)
        res["status"] = "pass" if not res.get("problems") else "candidate_fail"
        res["record"] = write_record("chat", label, name, {**res, "messages": None})
        steps[name] = res
        return res

    run("ordinary", lambda: _check_nonstream(base_url, tok, CONVERSATIONS["ordinary"],
                                             _nonstream(base_url, CONVERSATIONS["ordinary"]), bar))
    run("stream", lambda: _stream(base_url, tok, CONVERSATIONS["ordinary"], bar))
    run("turn", lambda: _check_nonstream(base_url, tok, CONVERSATIONS["turn"],
                                         _nonstream(base_url, CONVERSATIONS["turn"]), bar))
    with cf.ThreadPoolExecutor(2) as ex:
        fa = ex.submit(_nonstream, base_url, CONVERSATIONS["pair_a"])
        fb = ex.submit(_nonstream, base_url, CONVERSATIONS["pair_b"])
        conc = {"pair_a": fa.result(), "pair_b": fb.result()}
    for name in ("pair_a", "pair_b"):
        run(f"{name}_concurrent", lambda n=name: _check_nonstream(base_url, tok, CONVERSATIONS[n], conc[n], bar))
    for name in ("pair_a", "pair_b"):
        run(f"{name}_solo", lambda n=name: _check_nonstream(base_url, tok, CONVERSATIONS[n],
                                                            _nonstream(base_url, CONVERSATIONS[n]), bar))
    failing = [(n, s["problems"]) for n, s in steps.items() if s["status"] != "pass"]
    return {"status": "pass" if not failing else "candidate_fail",
            "reason": "; ".join(f"{n}: {p[0]}" for n, p in failing)[:1500] if failing else None,
            "steps": {n: {k: v for k, v in s.items() if k in ("status", "problems", "k3_vs_teacher_forced",
                                                            "max_abs_diff", "n_generated", "finish_reason",
                                                            "elapsed_s", "record")} for n, s in steps.items()}}
