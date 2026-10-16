#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The E2 arm: the relative-attention entry point as the landing's score-mod on plain FlashAttention-4.

Expected verdicts, stated first: it passes every private case, and it is the comparator of the two
component-efficiency nodes (ratio = median_E2 / median_candidate >= 0.8).

It runs over the reference tree (the verifier's snapshot, on PYTHONPATH), in its own process, with
`SGLANG_INKLING_FA4_USE_PIP=1` set before `sglang.jit_kernel.flash_attention` is imported, and passes
`score_mod = get_inkling_relative_attention_score_mod(E)` with `aux_tensors=[rel_bias]` and no `rel_bias` argument:
the calls the landing's FA4 backend makes with `SGLANG_OPT_USE_INKLING_SHEARED_BIAS=0` under the pip escape. At import
it asserts that the imported `_flash_attn_varlen_func` is `flash_attn.cute.interface`'s from the site-packages 4.0.0b15
distribution. `INFB_E2_FALLBACK=1` leaves the pip escape unset: the fork's generic FA4, same score-mod.
"""
from __future__ import annotations

import os

FALLBACK = os.environ.get("INFB_E2_FALLBACK") == "1"
if not FALLBACK:
    os.environ["SGLANG_INKLING_FA4_USE_PIP"] = "1"
os.environ["SGLANG_OPT_USE_INKLING_SHEARED_BIAS"] = "0"

import importlib.metadata as md  # noqa: E402

import torch  # noqa: E402

import sglang.jit_kernel.flash_attention_v4 as _v4  # noqa: E402
from sglang.jit_kernel.flash_attention import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: E402
from sglang.srt.models.inkling_common.attn import get_inkling_relative_attention_score_mod  # noqa: E402

_impl = getattr(_v4, "_flash_attn_varlen_func", None)
ARM = {"fallback": FALLBACK, "impl_module": getattr(_impl, "__module__", None),
       "flash_attn_4": md.version("flash-attn-4")}
if not FALLBACK:
    assert ARM["impl_module"] == "flash_attn.cute.interface", ARM
    assert ARM["flash_attn_4"] == "4.0.0b15", ARM


def _window(window_size) -> tuple[int, int]:
    left, right = window_size
    return (-1, -1) if left is None and right is None else (int(left), int(right))


def relative_attention(q, k, v, rel_bias, cu_seqlens_q, max_seqlen_q, *, cu_seqlens_k=None, max_seqlen_k=None,
                       page_table=None, seqused_k=None, causal=True, window_size=(None, None), softmax_scale,
                       out=None) -> torch.Tensor:
    if not causal:
        raise ValueError("relative_attention: the contract is causal")
    score_mod = get_inkling_relative_attention_score_mod(int(rel_bias.shape[-1]))
    common = dict(softmax_scale=softmax_scale, causal=True, window_size=_window(window_size), num_splits=0,
                  score_mod=score_mod, aux_tensors=[rel_bias], ver=4)
    if page_table is not None:
        result = flash_attn_with_kvcache(q=q, k_cache=k, v_cache=v, page_table=page_table, cache_seqlens=seqused_k,
                                         cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, **common)
    else:
        result = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, **common)
    if isinstance(result, tuple):
        result = result[0]
    if out is not None:
        out.copy_(result)
        return out
    return result
