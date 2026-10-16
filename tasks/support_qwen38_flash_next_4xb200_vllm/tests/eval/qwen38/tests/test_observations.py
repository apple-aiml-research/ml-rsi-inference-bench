#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Public examples of the observation rules (CPU): what is valid, and what fails structurally."""
import json
import math

import pytest

from qwen38.measurement import observations as obs

V = 248077


def completion(tokens, scores, *, finish="length", stop_reason=None, usage=None, extra_top=None):
    tops = [{f"token_id:{t}": s, **(extra_top or {}), **{f"token_id:{100 + k}": s - 5 - k for k in range(4)}}
            for t, s in zip(tokens, scores)]
    return {"choices": [{"index": 0, "text": "", "finish_reason": finish, "stop_reason": stop_reason,
                         "logprobs": {"tokens": [f"token_id:{t}" for t in tokens], "token_logprobs": scores,
                                      "top_logprobs": tops}}],
            "usage": usage or {"prompt_tokens": 3, "completion_tokens": len(tokens),
                               "total_tokens": 3 + len(tokens)}}


def test_a_complete_fixed_length_record_is_valid():
    row = obs.completion(completion([5, 6, 7], [-0.1, -0.2, -0.3]), "ok", requested=3, ignore_eos=True,
                         vocabulary=V, prompt_tokens=3)
    assert row["tokens"] == [5, 6, 7] and row["finish_reason"] == "length"


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["choices"][0]["logprobs"]["token_logprobs"].pop(), "misaligned"),
    (lambda r: r["choices"][0]["logprobs"]["tokens"].__setitem__(0, f"token_id:{V}"), "outside"),
    (lambda r: r["choices"][0]["logprobs"]["token_logprobs"].__setitem__(1, 0.5), "cannot exceed zero"),
    (lambda r: r["usage"].__setitem__("completion_tokens", 2), "usage"),
    (lambda r: r["choices"][0].__setitem__("finish_reason", "stop"), "fixed-length"),
    (lambda r: r["choices"].append(dict(r["choices"][0], index=1)), "choice 0"),
])
def test_malformed_records_fail_structurally(mutate, message):
    response = completion([5, 6, 7], [-0.1, -0.2, -0.3])
    mutate(response)
    with pytest.raises(obs.ObservationError, match=message):
        obs.completion(response, "bad", requested=3, ignore_eos=True, vocabulary=V, prompt_tokens=3)


def test_an_end_of_sequence_stop_is_witnessed_by_stop_reason():
    response = completion([5, 248046], [-0.1, -0.1], finish="stop", stop_reason=None)
    row = obs.completion(response, "eos", requested=8, ignore_eos=False, vocabulary=V, eos_token_ids=[248046])
    assert row["finish_reason"] == "stop"
    # the checkpoint's other end-of-sequence token is reported by its ID
    other = completion([5, 248044], [-0.1, -0.1], finish="stop", stop_reason=248044)
    obs.completion(other, "eos-2", requested=8, ignore_eos=False, vocabulary=V, eos_token_ids=[248046, 248044])
    response["choices"][0]["stop_reason"] = "###"
    with pytest.raises(obs.ObservationError, match="not a requested stop"):
        obs.completion(response, "eos", requested=8, ignore_eos=False, vocabulary=V, eos_token_ids=[248046])


def test_a_greedy_tie_outside_a_truncated_top_k_is_valid():
    # the chosen token ties the maximum but the five-entry map lists five other tokens at that score
    response = completion([5], [-2.0])
    response["choices"][0]["logprobs"]["top_logprobs"] = [{f"token_id:{900 + k}": -2.0 for k in range(5)}]
    obs.completion(response, "tie", requested=1, ignore_eos=True, vocabulary=V)


def test_a_non_greedy_choice_is_structural():
    response = completion([5], [-2.0], extra_top={"token_id:9": -0.5})
    with pytest.raises(obs.ObservationError, match="greedy"):
        obs.completion(response, "argmax", requested=1, ignore_eos=True, vocabulary=V)


def test_k3_overflow_is_a_numerical_infinity():
    result = obs.k3(800.0)
    assert result == {"value": None, "overflow": True} and math.isinf(obs.k3_value(result))
    json.dumps(result, allow_nan=False)
    assert obs.k3(0.0)["value"] == 0.0


def test_a_stream_needs_its_terminal_records():
    chunk = {"choices": [{"index": 0, "text": "a", "finish_reason": None,
                          "logprobs": {"tokens": ["token_id:5"], "token_logprobs": [-0.1],
                                       "top_logprobs": [{"token_id:5": -0.1, **{f"token_id:{6 + k}": -3.0 - k
                                                                                for k in range(4)}}]}}]}
    final = {"choices": [{"index": 0, "text": "", "finish_reason": "length", "stop_reason": None,
                          "logprobs": {"tokens": [], "token_logprobs": [], "top_logprobs": []}}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    payloads = [json.dumps(chunk), json.dumps(final), json.dumps(usage), "[DONE]"]
    obs.stream(payloads, "s", kind="completion", requested=1, ignore_eos=True, vocabulary=V)
    for broken in (payloads[:-1], [payloads[0], payloads[2], "[DONE]"], [payloads[0], payloads[1], "[DONE]"]):
        with pytest.raises(obs.ObservationError):
            obs.stream(broken, "s", kind="completion", requested=1, ignore_eos=True, vocabulary=V)
