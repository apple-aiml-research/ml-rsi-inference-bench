#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Edits the multi-turn bench client needs, applied to the ENGINE TREE at run time."""

from __future__ import annotations

import re

from pathlib import Path

MARKER = "# infb-pd-client-edits"

# Log-spaced ITL buckets at 2**(1/32) ~ 2.2% per bucket. THE GEOMETRY IS DUPLICATED in
# pd_turn_metrics.pooled_itl_percentile and the two must not drift: this side writes the bucket
# indices and that side turns them back into milliseconds. It is duplicated rather than imported
# because this text is injected into a file in the engine tree, which cannot import a task's
# verifier library.
_HELPERS = MARKER + """
_INFB_ITL_FACTOR = 2.0 ** (1.0 / 32.0)
_INFB_ITL_MIN_MS = 0.1


def _infb_itl_summary(gaps_ns):
    if not gaps_ns:
        return (-1.0, -1.0, -1.0, ())
    import math
    ms = sorted(g / 1e6 for g in gaps_ns)

    def _q(pp):
        return ms[min(len(ms) - 1, int(round(pp * (len(ms) - 1))))]

    hist = {}
    for v in ms:
        b = 0 if v <= _INFB_ITL_MIN_MS else int(math.log(v / _INFB_ITL_MIN_MS, _INFB_ITL_FACTOR)) + 1
        hist[b] = hist.get(b, 0) + 1
    return (_q(0.50), _q(0.99), ms[-1], tuple(x for b in sorted(hist) for x in (b, hist[b])))


"""

_ITL_FIELDS = (
    "    itl_p50_ms: float = -1.0\n"
    "    itl_p99_ms: float = -1.0\n"
    "    itl_max_ms: float = -1.0\n"
    "    itl_hist: tuple = ()\n"
)

def _tokenizer_load(indent: str, call_args: str) -> str:
    inner = indent + "    "
    return (
        f"{indent}try:\n"
        f"{inner}tokenizer = AutoTokenizer.from_pretrained({call_args})\n"
        f"{indent}except Exception as infb_tok_exc:  # noqa: BLE001 -- ANY config failure means the same here\n"
        f"{inner}from transformers import PreTrainedTokenizerFast  # type: ignore\n"
        f"{inner}logger.warning(\n"
        f"{inner}    'AutoTokenizer could not resolve this checkpoint through its config (%s: %s); loading '\n"
        f"{inner}    'the tokenizer files directly. This client needs a tokenizer, not a model config.',\n"
        f"{inner}    type(infb_tok_exc).__name__, infb_tok_exc)\n"
        f"{inner}tokenizer = PreTrainedTokenizerFast.from_pretrained({call_args})"
    )


def _require(text: str, needle: str, what: str) -> None:
    if needle not in text:
        raise SystemExit(
            f"FATAL: the multi-turn client's {what} anchor is gone, so the edit cannot be placed. An "
            f"unpatched client measures unequal work between the two arms and reports nothing about it."
        )


def patch_multi_turn_client(client: Path) -> str:
    """Apply the edits in place. Returns what was done, for the run log."""
    notes: list[str] = []
    text = client.read_text(encoding="utf-8")
    if MARKER in text:
        return "already"
    before = text

    eos_anchor = '    if max_tokens is not None:\n        payload["max_tokens"] = max_tokens\n'
    _require(text, eos_anchor, "max_tokens payload")
    text = text.replace(eos_anchor, eos_anchor + '        payload["ignore_eos"] = True\n', 1)

    # (1b) load the tokenizer WITHOUT going through AutoConfig when AutoConfig cannot read the checkpoint.
    #
    # `AutoTokenizer.from_pretrained(args.model)` loads the model CONFIG on the way to resolving the
    # tokenizer class. DeepSeek-V4-Flash declares `model_type: deepseek_v4`, which transformers does not
    # know and the checkpoint does not carry an `auto_map` for, so BOTH doors are shut:
    #
    #     KeyError: 'deepseek_v4'                          # CONFIG_MAPPING has no such entry
    #     ValueError: The checkpoint you are trying to load has model type `deepseek_v4` but Transformers
    #                 does not recognize this architecture
    #     ...falling back to PreTrainedConfig, which then trips on the same config:
    #     AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'
    #
    # The client does not need the config. It needs a TOKENIZER -- `get_token_count` and the chat-template
    # render, nothing else -- and `PreTrainedTokenizerFast.from_pretrained` reads `tokenizer_config.json`
    # and `tokenizer.json` directly without consulting AutoConfig at all. So the fallback is not a
    # workaround for a broken checkpoint; it is asking for the thing actually wanted.
    #
    # Kept as a FALLBACK rather than a replacement so a checkpoint that resolves normally keeps the
    # tokenizer class its config names, and the substitution is announced in the log rather than silent.
    tok_call = re.search(
        r"^(?P<indent>[ \t]*)tokenizer\s*=\s*AutoTokenizer\.from_pretrained\((?P<args>[^()]*)\)[ \t]*$",
        text, re.M)
    if tok_call is None:
        notes.append("tokenizer-load: no `AutoTokenizer.from_pretrained(...)` assignment found, left alone")
    else:
        text = text[:tok_call.start()] + _tokenizer_load(
            tok_call.group("indent"), tok_call.group("args")) + text[tok_call.end():]
        notes.append("tokenizer-load: wrapped with a config-free fallback")

    _require(text, "class ServerResponse(NamedTuple):", "ServerResponse class")
    text = text.replace("class ServerResponse(NamedTuple):", _HELPERS + "class ServerResponse(NamedTuple):", 1)
    for cls, last_field in (("class ServerResponse(NamedTuple):", "    num_chunks: int\n"),
                            ("class RequestStats(NamedTuple):", "    client_id: int\n")):
        _require(text, cls, cls.split("(")[0])
        start = text.index(cls)
        _require(text[start:], last_field, f"{cls.split('(')[0]} last field")
        end = text.index(last_field, start) + len(last_field)
        text = text[:end] + _ITL_FIELDS + text[end:]

    reduce_anchor = "    num_chunks: int = len(chunk_delay)\n"
    _require(text, reduce_anchor, "chunk_delay reduction")
    text = text.replace(reduce_anchor, reduce_anchor + "    _itl = _infb_itl_summary(chunk_delay)\n", 1)

    sr_anchor = "        num_chunks=num_chunks,\n    )"
    _require(text, sr_anchor, "ServerResponse construction")
    text = text.replace(sr_anchor, "        num_chunks=num_chunks,\n"
                                   "        itl_p50_ms=_itl[0],\n"
                                   "        itl_p99_ms=_itl[1],\n"
                                   "        itl_max_ms=_itl[2],\n"
                                   "        itl_hist=_itl[3],\n    )", 1)
    rs_anchor = "        client_id=client_id,\n    )"
    _require(text, rs_anchor, "RequestStats construction")
    text = text.replace(rs_anchor, "        client_id=client_id,\n"
                                   "        itl_p50_ms=response.itl_p50_ms,\n"
                                   "        itl_p99_ms=response.itl_p99_ms,\n"
                                   "        itl_max_ms=response.itl_max_ms,\n"
                                   "        itl_hist=response.itl_hist,\n    )", 1)

    if text == before:
        raise SystemExit("FATAL: the client edits changed nothing; a no-op patch measures the wrong thing quietly")
    compile(text, str(client), "exec")
    client.write_text(text, encoding="utf-8")
    return "patched" + (" [" + "; ".join(notes) + "]" if notes else "")
