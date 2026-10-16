#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The typed protocol of the sparse-indexer contract (README.md): shapes, dtypes and strides of one step, the output
domain, and the loading of the implementation under test. It validates and loads; it computes nothing.

PyTorch and the engine are imported inside the functions, so the module imports with the standard library alone.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

NUM_CONSUMERS = 4
NUM_HEADS = 32
HEAD_DIM = 128
PACKED_BYTES = 64
NUM_CANDIDATES = 2048
CANDIDATE_BLOCK = 8
SPARSE_BLOCK_KV = 8
TOPK = 512
MAX_VISIBLE = 32768
PAGE_RECORD_BYTES = 68
PAGE_STRIDE_ALIGNMENT = 512
MAX_PACKED_ROWS = 512
MAX_PAGED_ROWS = 64
MAX_ROWS_PER_REQUEST = 6

DEPENDENCY = "vllm.third_party.deep_gemm_v41_sparse"
DEPENDENCY_API = (
    "get_sparse_mqa_logits_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "fp8_fp4_sparse_mqa_logits",
    "fp8_fp4_paged_sparse_mqa_logits",
)
ENTRY_POINTS = "vllm.model_executor.layers.sparse_indexer_eval"


@dataclass
class PackedStep:
    """One packed prefill step of the four consumers (README.md, "The component-measurement interface")."""

    q: Any  # int8 [4, R, 32, 64]
    q_scale: Any  # int32 [4, R, 32]
    k_quant: Any  # int8 [N, 64]
    k_scale: Any  # int32 [N]
    weights: Any  # bf16 [4, R, 32]
    k_start: Any  # int32 [R]
    k_end: Any  # int32 [R]
    candidate_blocks: Any  # int32 [R, 2048], rows contiguous

    @property
    def rows(self) -> int:
        return int(self.q.shape[1])


@dataclass
class PagedStep:
    """One paged decode/verification step of the four consumers."""

    q: Any  # int8 [4, R, 1, 32, 64]
    q_scale: Any  # int32 [4, R, 1, 32]
    kv_cache: Any  # uint8 [pages, page_kv, 1, 68], page stride a multiple of 512
    weights: Any  # bf16 [4, R, 32]
    context_lens: Any  # int32 [R]
    block_table: Any  # int32 [R, max_pages], rows contiguous
    row_indices: Any  # int32 [R]
    candidate_blocks: Any  # int32 [R, 2048], rows contiguous

    @property
    def rows(self) -> int:
        return int(self.q.shape[1])


def _check(problems: list[str], cond: bool, what: str) -> None:
    if not cond:
        problems.append(what)


def packed_problems(s: PackedStep) -> list[str]:
    """Every way `s` falls outside the packed interface."""
    import torch

    p: list[str] = []
    r = s.rows
    _check(p, s.q.dtype == torch.int8 and tuple(s.q.shape) == (NUM_CONSUMERS, r, NUM_HEADS, PACKED_BYTES)
           and s.q.is_contiguous(), "q must be contiguous int8 [4, R, 32, 64]")
    _check(p, s.q_scale.dtype == torch.int32 and tuple(s.q_scale.shape) == (NUM_CONSUMERS, r, NUM_HEADS)
           and s.q_scale.is_contiguous(), "q_scale must be contiguous int32 [4, R, 32]")
    _check(p, s.k_quant.dtype == torch.int8 and s.k_quant.dim() == 2 and s.k_quant.shape[1] == PACKED_BYTES
           and s.k_quant.is_contiguous(), "k_quant must be contiguous int8 [N, 64]")
    _check(p, s.k_scale.dtype == torch.int32 and tuple(s.k_scale.shape) == (s.k_quant.shape[0],)
           and s.k_scale.is_contiguous(), "k_scale must be contiguous int32 [N]")
    _check(p, s.weights.dtype == torch.bfloat16 and tuple(s.weights.shape) == (NUM_CONSUMERS, r, NUM_HEADS)
           and s.weights.is_contiguous(), "weights must be contiguous bf16 [4, R, 32]")
    for name in ("k_start", "k_end"):
        t = getattr(s, name)
        _check(p, t.dtype == torch.int32 and tuple(t.shape) == (r,), f"{name} must be int32 [R]")
    c = s.candidate_blocks
    _check(p, c.dtype == torch.int32 and tuple(c.shape) == (r, NUM_CANDIDATES) and c.stride(1) == 1,
           "candidate_blocks must be int32 [R, 2048] with contiguous rows")
    _check(p, r <= MAX_PACKED_ROWS, f"at most {MAX_PACKED_ROWS} packed rows")
    return p


def paged_problems(s: PagedStep) -> list[str]:
    """Every way `s` falls outside the paged interface."""
    import torch

    p: list[str] = []
    r = s.rows
    _check(p, s.q.dtype == torch.int8 and tuple(s.q.shape) == (NUM_CONSUMERS, r, 1, NUM_HEADS, PACKED_BYTES)
           and s.q.is_contiguous(), "q must be contiguous int8 [4, R, 1, 32, 64]")
    _check(p, s.q_scale.dtype == torch.int32 and tuple(s.q_scale.shape) == (NUM_CONSUMERS, r, 1, NUM_HEADS)
           and s.q_scale.is_contiguous(), "q_scale must be contiguous int32 [4, R, 1, 32]")
    kv = s.kv_cache
    _check(p, kv.dtype == torch.uint8 and kv.dim() == 4 and kv.shape[2] == 1 and kv.shape[3] == PAGE_RECORD_BYTES
           and kv.shape[1] % SPARSE_BLOCK_KV == 0 and kv.stride(0) % PAGE_STRIDE_ALIGNMENT == 0
           and kv.stride(0) >= kv.shape[1] * PAGE_RECORD_BYTES,
           "kv_cache must be uint8 [pages, page_kv, 1, 68] with a page stride that is a multiple of 512")
    _check(p, s.weights.dtype == torch.bfloat16 and tuple(s.weights.shape) == (NUM_CONSUMERS, r, NUM_HEADS)
           and s.weights.is_contiguous(), "weights must be contiguous bf16 [4, R, 32]")
    for name in ("context_lens", "row_indices"):
        t = getattr(s, name)
        _check(p, t.dtype == torch.int32 and tuple(t.shape) == (r,), f"{name} must be int32 [R]")
    bt = s.block_table
    _check(p, bt.dtype == torch.int32 and bt.dim() == 2 and bt.shape[0] == r and bt.stride(1) == 1,
           "block_table must be int32 [R, pages] with contiguous rows")
    c = s.candidate_blocks
    _check(p, c.dtype == torch.int32 and tuple(c.shape) == (r, NUM_CANDIDATES) and c.stride(1) == 1,
           "candidate_blocks must be int32 [R, 2048] with contiguous rows")
    _check(p, r <= MAX_PAGED_ROWS, f"at most {MAX_PAGED_ROWS} paged rows")
    return p


def output_buffer(rows: int, device: Any):
    """A fresh int32 [4, R, 512] output buffer, filled with a value no valid output contains."""
    import torch

    return torch.full((NUM_CONSUMERS, rows, TOPK), -7, dtype=torch.int32, device=device)


def output_problems(out: Any, rows: int) -> list[str]:
    import torch

    p: list[str] = []
    _check(p, out.dtype == torch.int32 and tuple(out.shape) == (NUM_CONSUMERS, rows, TOPK),
           "out must remain int32 [4, R, 512]")
    return p


def load_dependency() -> dict[str, Any]:
    """The four operations of the supplied dependency, as the engine resolves them."""
    module = importlib.import_module(DEPENDENCY)
    return {name: getattr(module, name) for name in DEPENDENCY_API}


def load_entry_points():
    """`(prefill, decode)` of the component-measurement interface."""
    module = importlib.import_module(ENTRY_POINTS)
    return module.prefill, module.decode


def run_prefill(prefill, s: PackedStep, out, workspace=None):
    return prefill(s.q, s.q_scale, s.k_quant, s.k_scale, s.weights, s.k_start, s.k_end, s.candidate_blocks, out,
                   workspace=workspace)


def run_decode(decode, s: PagedStep, out, workspace=None):
    return decode(s.q, s.q_scale, s.kv_cache, s.weights, s.context_lens, s.block_table, s.row_indices,
                  s.candidate_blocks, out, workspace=workspace)
