#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""State-sensitive serving observations on aligned logical histories.

A warm generation reads the prefix cache without requesting prompt logprobs.
A separate cold teacher-forced request scores that generation's *actual* token
history. Thus every compared position has the same history, even when two
independently generated continuations would diverge. Local windows retain
boundary errors which an average over an entire long prompt could conceal.
Inputs, numerical limits and verdict policy belong to the caller.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

from functools import lru_cache

from qwen38.measurement import observations
from qwen38.serving import fidelity, mtp


@lru_cache(maxsize=1)
def vocabulary() -> int:
    return observations.vocabulary_size("/model")

from qwen38.serving.reuse import HITS, QUERIES, cache_delta, cache_reading, counter  # noqa: E402,F401


def generate(base, model, ids, n, *, case_id="generation", salt=None, timeout=600):
    """A validated greedy fixed-length continuation (token IDs, scores, top-five maps)."""
    row = mtp.fixed_length(base, model, ids, n, case_id=case_id, vocabulary=vocabulary(), salt=salt,
                           timeout=timeout)
    row["token_ids"] = list(row["tokens"])
    return row


def local_statistics(reference, served, *, windows=(1, 2, 4, 8, 9, 10)):
    """Per-position k3 of aligned scores and local window means; an exponent overflow is a numerical
    +infinity (recorded as a null value with an overflow flag), never a clipped value."""
    if not reference or len(reference) != len(served):
        raise observations.ObservationError("aligned comparison requires every generated position")
    ratios = [observations.logprob(r, "reference") - observations.logprob(v, "served")
              for r, v in zip(reference, served)]
    ks = [observations.k3(value) for value in ratios]
    values = [observations.k3_value(k) for k in ks]
    local = {}
    for width in windows:
        if type(width) is not int or width < 1:
            raise ValueError("local windows must have positive integer widths")
        if width > len(values):
            continue
        draws = [math.fsum(v / width for v in values[start:start + width]) if all(math.isfinite(v) for v in
                 values[start:start + width]) else math.inf for start in range(len(values) - width + 1)]
        worst = max(range(len(draws)), key=draws.__getitem__)
        local[str(width)] = {"max_mean": observations.json_number(draws[worst]), "start": worst,
                             "means": [observations.json_number(x) for x in draws]}
    mean = math.fsum(v / len(values) for v in values) if all(math.isfinite(v) for v in values) else math.inf
    return {"positions": len(values), "log_ratios": ratios, "k3": ks, "mean": observations.json_number(mean),
            "windows": local}


def statistic(value: dict) -> float:
    """A serialized statistic as a number (infinity for an overflow verdict)."""
    return observations.k3_value(value)


TURNOVER_FAMILY = ("joining-0", "joining-1", "joining-2", "joining-3", "replacement")


def turnover_diagnostic(aligned, *, family=TURNOVER_FAMILY):
    """The mixed-turnover family's pooled statistic, recorded and never graded.

    S_mt is the mean |served - teacher| over the pooled positions of the family's joining and
    replacement records. Pooling bounds each position's weight, which the per-record k3 mean does
    not. It is recorded so that a reading of a run, or a later revision, can use it; nothing asserts
    on it.
    """
    rows, pooled = [], []
    for record in aligned:
        if record["id"] not in family:
            continue
        ratios = record["statistics"]["log_ratios"]
        values = [abs(x) for x in ratios]
        pooled += values
        mean = math.fsum(v / len(values) for v in values) if values and all(math.isfinite(v) for v in values) \
            else math.inf
        rows.append({"id": record["id"], "positions": len(values), "mean_abs": observations.json_number(mean)})
    s_mt = math.fsum(v / len(pooled) for v in pooled) if pooled and all(math.isfinite(v) for v in pooled) \
        else math.inf
    return {"statistic": "mean |served - teacher| over the pooled positions of the joining and replacement records",
            "graded": False, "family": list(family), "records": rows, "positions": len(pooled),
            "s_mt": observations.json_number(s_mt)}


def aligned_continuation(base, model, ids, n, *, case_id, salt=None, require_hit=False, recipe="ordinary"):
    before = cache_reading(base)
    spec_before = observations.parse_spec_counters(mtp.request(base, "/metrics", timeout=60)) if recipe == "mtp3" else None
    generated = generate(base, model, ids, n, case_id=case_id, salt=salt)
    spec_after = observations.parse_spec_counters(mtp.request(base, "/metrics", timeout=60)) if recipe == "mtp3" else None
    cache = cache_delta(before, cache_reading(base))
    if require_hit and cache["hits"] <= 0:
        raise observations.ObservationError(f"{case_id}: the observed continuation did not restore a cached prefix")
    teacher_before = cache_reading(base)
    history = ids + generated["token_ids"]
    all_scores = fidelity.prompt_logprobs(base, model, history, vocabulary=vocabulary())
    teacher_cache = cache_delta(teacher_before, cache_reading(base))
    if teacher_cache["hits"] != 0:
        raise observations.ObservationError(f"{case_id}: the independent cold recomputation read cached state")
    reference = all_scores[len(ids) - 1:]
    report = {"id": case_id, "recipe": recipe, "prompt_token_ids": ids,
              "generation": generated, "cache": cache, "teacher_cache": teacher_cache,
              "teacher_logprobs": reference,
              "statistics": local_statistics(reference, generated["token_logprobs"])}
    if spec_before is not None:
        report["speculation"] = observations.counter_delta(spec_before, spec_after, completion_tokens=n)
    return report


def discover_boundary(base, model, tokens, *, salt, lengths=(4096, 8192), repeats=2):
    """Bounded cache preparation, observing the actual served cache geometry."""
    if repeats < 1 or repeats > 3:
        raise ValueError("cache preparation permits one to three repeats")
    probes = []
    for size in lengths:
        if size > len(tokens): continue
        ids = tokens[:size]
        generate(base, model, ids, 1, case_id="prepare", salt=salt)
        for attempt in range(repeats):
            before = cache_reading(base)
            generate(base, model, ids, 1, case_id="boundary", salt=salt)
            measured = cache_delta(before, cache_reading(base))
            probes.append({"tokens": size, "attempt": attempt, **measured})
            hit = measured["hits"]
            if hit > 0:
                if hit >= size or not hit.is_integer():
                    raise observations.ObservationError(f"invalid prefix boundary {hit} for {size} tokens")
                return {"boundary_tokens": int(hit), "warmed_prompt_tokens": size, "probes": probes}
    raise observations.ObservationError(f"no prefix restoration in the bounded preparation: {probes}")


def restored_branches(base, model, tokens, suffixes, n, *, case_id, recipe):
    """Warm distinct suffixes, revisit them, and score each actual continuation."""
    if len(suffixes) < 2 or any(not suffix for suffix in suffixes):
        raise ValueError("at least two nonempty branch suffixes are required")
    salt = f"qwen38-branches-{uuid.uuid4().hex}"
    discovered = discover_boundary(base, model, tokens, salt=salt)
    # Extend the prompt that actually witnessed reuse. Truncating it to the
    # reported hit count changes the cache request, and last-state caching may
    # then have no eligible state at that shorter length. The observed hit
    # boundary is still retained and every checked branch must witness reuse.
    prefix = tokens[:discovered["warmed_prompt_tokens"]]
    records = []
    for visit in range(2):
        for branch, suffix in enumerate(suffixes):
            prompt = prefix + suffix
            records.append(aligned_continuation(base, model, prompt, n,
                case_id=f"{case_id}/visit-{visit}/branch-{branch}", salt=salt,
                require_hit=True, recipe=recipe))
    return {"id": case_id, "boundary": discovered, "branches": records}


def turnover(base, model, prompts, *, n=16):
    """Finish unrelated requests before a fresh request consumes released slots."""
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(generate, base, model, ids, n, case_id=f"turnover-{i}",
                               salt=f"qwen38-turnover-{uuid.uuid4().hex}")
                   for i, ids in enumerate(prompts)]
        return [future.result() for future in futures]


ITERATION = re.compile(r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
                       r"(\d+) generation requests, (\d+) generation tokens,")


def scheduler_coverage(log_text):
    """Parse the engine's iteration observations for an isolated workload interval."""
    rows = [{"iteration": int(m[1]), "prefill_requests": int(m[2]), "prefill_tokens": int(m[3]),
             "decode_requests": int(m[4]), "decode_tokens": int(m[5])}
            for line in log_text.splitlines() if (m := ITERATION.search(line))]
    mixed = [row for row in rows if row["prefill_requests"] > 0 and row["decode_requests"] > 0]
    return {"iterations": rows, "mixed_iterations": mixed,
            "prefill_observed": any(row["prefill_tokens"] for row in rows),
            "decode_observed": any(row["decode_tokens"] for row in rows)}


def stream_generation(base, model, ids, n, *, case_id, first_token, timeout=600):
    """Complete SSE observation, including token arrival times and final usage."""
    body = {"model": model, "prompt": ids, "max_tokens": n, "temperature": 0., "seed": 0,
            "ignore_eos": True, "skip_special_tokens": False, "return_tokens_as_token_ids": True,
            "logprobs": 5, "stream": True, "stream_options": {"include_usage": True},
            "cache_salt": f"qwen38-mixed-{uuid.uuid4().hex}"}
    started = time.monotonic()
    row = {"id": case_id, "tokens": [], "token_logprobs": [], "top_logprobs": [],
           "finish_reason": None, "completion_tokens": None, "arrivals": []}
    request = urllib.request.Request(base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    complete = False
    payloads = []
    failure = None
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
            for line in response:
                line = line.decode().strip()
                if not line or line.startswith(":"): continue
                if not line.startswith("data: "):
                    raise observations.ObservationError("malformed SSE observation")
                data = line[6:]
                if data == "[DONE]": complete = True; break
                payloads.append(data)
                payload = json.loads(data)
                if payload.get("error"):
                    raise observations.ObservationError(str(payload["error"]))
                if payload.get("usage") is not None:
                    row["completion_tokens"] = payload["usage"]["completion_tokens"]
                for choice in payload.get("choices", []):
                    if choice.get("index") != 0:
                        raise observations.ObservationError("unexpected streaming choice")
                    lp = choice.get("logprobs")
                    if lp:
                        if any(not isinstance(lp.get(key), list) for key in ("tokens", "token_logprobs", "top_logprobs")):
                            raise observations.ObservationError("streaming chunk has incomplete log probabilities")
                        size = len(lp["tokens"])
                        if len(lp["token_logprobs"]) != size or len(lp["top_logprobs"]) != size:
                            raise observations.ObservationError("streaming chunk omits token observations")
                        for key in ("tokens", "token_logprobs", "top_logprobs"): row[key].extend(lp[key])
                        row["arrivals"].extend([time.monotonic()] * size)
                        if size: first_token.set()
                    if choice.get("finish_reason") is not None:
                        if row["finish_reason"] is not None:
                            raise observations.ObservationError("duplicate finish observation")
                        row["finish_reason"] = choice["finish_reason"]
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        row.update(started=started, finished=time.monotonic(), seconds=time.monotonic() - started)
        mtp.raw_record('stream', {'request': body, 'payloads': payloads, 'complete': complete,
                                  'observation': row, 'transport_error': failure})
    if not complete:
        raise observations.ObservationError("stream ended without its terminal record")
    arrivals = row["arrivals"]
    checked = observations.stream(payloads + ["[DONE]"], case_id, kind="completion", requested=n, ignore_eos=True,
                                  vocabulary=vocabulary(), prompt_tokens=len(ids))
    checked.update(arrivals=arrivals, started=row["started"], finished=row["finished"], seconds=row["seconds"],
                   token_ids=list(checked["tokens"]))
    return checked


def mixed_turnover(base, model, background_prompts, joining_prompts, fresh_ids, *, launch_log,
                   background_lengths=(256, 512, 768, 1024), n=64):
    """Unequal prefills join live decode; a completion is replaced by a fresh request.

    Streaming establishes the arrival/completion ordering. The native engine's
    iteration log must independently show a mixed prefill/decode iteration.
    A workload which finishes too early or never mixes is missing coverage.
    Every request is scored against its cold teacher, the background decodes
    included: they are the requests that decode through the mixed iterations.
    """
    from pathlib import Path
    if len(background_prompts) != len(background_lengths) or len(background_prompts) < 2:
        raise ValueError("background lengths must cover at least two prompts")
    if not joining_prompts or len({len(ids) for ids in joining_prompts}) < 2:
        raise ValueError("joining prefills must have unequal lengths")
    log_path = Path(launch_log)
    log_offset = log_path.stat().st_size
    events = [threading.Event() for _ in background_prompts]
    joined_events = [threading.Event() for _ in joining_prompts]
    with ThreadPoolExecutor(max_workers=len(events) + len(joined_events) + 1) as pool:
        backgrounds = [pool.submit(stream_generation, base, model, ids, length,
            case_id=f"background-{i}", first_token=events[i])
            for i, (ids, length) in enumerate(zip(background_prompts, background_lengths))]
        for event in events:
            if not event.wait(timeout=60):
                raise observations.ObservationError("background produced no first token within 60 seconds")
        joining = [pool.submit(stream_generation, base, model, ids, n,
            case_id=f"joining-{i}", first_token=joined_events[i]) for i, ids in enumerate(joining_prompts)]
        # The shortest background releases a real slot while longer decodes continue.
        first_finished = backgrounds[0].result(timeout=600)
        replacement = stream_generation(base, model, fresh_ids, n, case_id="replacement", first_token=threading.Event())
        background_rows = [first_finished] + [future.result(timeout=600) for future in backgrounds[1:]]
        joining_rows = [future.result(timeout=600) for future in joining]
    if not any(row["finished"] > replacement["started"] for row in background_rows[1:]):
        raise observations.ObservationError("no live decode remained when the fresh request replaced a completion")
    if not all(any(bg["arrivals"][0] < row["started"] < bg["finished"] for bg in background_rows)
               for row in joining_rows):
        raise observations.ObservationError("joining prefill did not arrive during live decode")
    with log_path.open("rb") as log:
        log.seek(log_offset); text = log.read().decode(errors="replace")
    coverage = scheduler_coverage(text)
    if not coverage["mixed_iterations"]:
        raise observations.ObservationError("native scheduler recorded no mixed prefill/decode iteration")
    # Cold scoring is after all background work, so its cache witness is isolated.
    scored = []
    for ids, row in [*zip(background_prompts, background_rows), *zip(joining_prompts, joining_rows),
                     (fresh_ids, replacement)]:
        before = cache_reading(base)
        scores = fidelity.prompt_logprobs(base, model, ids + row["token_ids"], vocabulary=vocabulary())[len(ids) - 1:]
        delta = cache_delta(before, cache_reading(base))
        if delta["hits"]:
            raise observations.ObservationError("mixed-state teacher request read cached state")
        scored.append({"id": row["id"], "prompt_token_ids": ids, "teacher_logprobs": scores,
                       "statistics": local_statistics(scores, row["token_logprobs"]), "teacher_cache": delta})
    return {"schema": "qwen38-mixed-state/1", "background": background_rows,
            "joining": joining_rows, "replacement": replacement, "aligned": scored, "scheduler": coverage}
