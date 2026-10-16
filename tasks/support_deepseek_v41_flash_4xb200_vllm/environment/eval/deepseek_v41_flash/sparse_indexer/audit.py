#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The read-only schema of the optional binding description (README.md, "Optional binding description").

An implementation that reorganizes the model's indexer or attention boundaries may provide
``vllm.model_executor.layers.sparse_indexer_audit.get_bindings(model) -> list[Binding]`` to describe, for the
evaluation's observer, where the equivalent computations happen. Each mapping receives the actual call's arguments
and result and returns views or coordinate descriptions of that computation; it may not compute a substitute
score, selection or attention result, replay a layer, clone state, mutate its inputs or write anything. The
description is optional: without it, evaluation uses its own knowledge of the engine's conventions.

What the observer does with a description: it observes ``call`` (a module's forward, or a function or bound method
where the model looks it up), and after each call reads ``views(args, kwargs, result)``; it observes the native
FlashMLA calls (``flash_mla_with_kvcache`` for decode, ``flash_mla_sparse_fwd`` for prefill) made while ``attention``
runs, and reads ``to_attention(views, leaf, leaf_args)`` for each of them. It copies what it needs from the tensors
these return and checks it against the calls themselves; a view is never trusted to connect two tensors.

Views (``views``), every tensor with the call's rows as its first dimension (``T`` rows):
  * ``rows``: int64 ``[T, 3]``: an integer naming the row's request (the same for all of a request's rows in a step and
    unique among live requests), the row's visible compressed length, and 1 for a decode or verification row, 0 for a
    prefill row;
  * ``q`` int8 ``[T, 32, 64]``, ``q_scale`` int32 ``[T, 32]``, ``weights`` ``[T, 32]``: the call's actual inputs;
  * ``k_cache``: the paged index-key cache the call reads, uint8 ``[pages, page_kv, 68]`` in the layout of this
    README's paged interface (each page's values, then its scales), and ``k_table``: int32 ``[T, P]``, each row's page
    table into it, or ``(table, index)`` with ``table`` int32 ``[N, P]`` and ``index`` int64 ``[T]`` (row -> table row);
  * ``candidate_pool`` int32 ``[T, 2048]``: the pool the call wrote (the source) or read (a consumer);
  * ``selection`` int32 ``[T, 512]`` (a consumer): the request-local positions the call selected;
  * optional, for the consumers' BF16 weights: ``projection`` ``[T, 32]``, the values the weights were computed from,
    and ``weight_scale``, the float they were multiplied by (the observer otherwise reads them at the supplied fused
    query call).

Attention coordinates (``to_attention(views, leaf, leaf_args)``, ``leaf`` the FlashMLA function's name, ``leaf_args``
its arguments by name), every tensor with the leaf call's rows as its first dimension (``R`` rows):
  * ``rows`` int64 ``[R]``: the producer call's row each leaf row serves;
  * decode: ``block_table`` int32 ``[R, P]`` and ``block_size``: the compressed cache's page table through which the
    row's selected positions become ``extra_indices_in_kvcache``;
  * prefill: ``region`` int64 ``[R]``, the index in ``indices`` of the row's compressed position 0; ``window_base``
    int64 ``[R]``, the index of absolute position 0 of the row's sliding window; ``positions`` int64 ``[R]``, the row's
    absolute positions; ``window`` and ``compress_ratio``.

The module imports with the standard library alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

SCHEMA_VERSION = 2

#: What a producer mapping returns for one call, as views of the actual tensors (keys a mapping may omit are those
#: the role has no use for).
SOURCE_VIEWS = ("rows", "q", "q_scale", "weights", "k_cache", "k_table", "candidate_pool")
CONSUMER_VIEWS = ("rows", "q", "q_scale", "weights", "k_cache", "k_table", "candidate_pool", "selection",
                  "projection", "weight_scale")
ATTENTION_COORDINATES = ("rows", "block_table", "block_size", "region", "window_base", "positions", "window",
                         "compress_ratio")


@dataclass(frozen=True)
class Binding:
    """One observed boundary of one layer.

    ``call`` is the object whose calls are observed (a module or a callable reachable from ``model``);
    ``views(args, kwargs, result)`` maps one actual call to a dict whose keys are among the role's view names above.
    For a consumer, ``attention`` names the call site during which its native FlashMLA calls run, and
    ``to_attention(views, leaf, leaf_args)`` gives, for one such FlashMLA call, the coordinates above: how the
    request-local selected positions appear there (prefill indices, or physical decode slots through the page table)
    without computing them. ``output_use`` names the module or callable that reads the attention output.
    """

    layer_id: int
    role: Literal["source", "consumer"]
    call: Any
    views: Callable[[tuple, dict, Any], dict]
    attention: Any = None
    to_attention: Callable[[dict, str, dict], dict] | None = None
    output_use: Any = None
    notes: dict = field(default_factory=dict)


def validate(bindings: list[Binding]) -> list[str]:
    """Structural problems of a binding description (types and roles only; evaluation checks the views against the
    intercepted calls)."""
    problems = []
    if not isinstance(bindings, list):
        return ["get_bindings must return a list of Binding"]
    seen = set()
    for b in bindings:
        # a copy of the dataclass serves: the fields, not the class, are the schema
        if not all(hasattr(b, k) for k in ("layer_id", "role", "call", "views", "attention", "to_attention")):
            problems.append(f"not a Binding: {type(b).__name__}")
            continue
        if b.role not in ("source", "consumer"):
            problems.append(f"layer {b.layer_id}: role {b.role!r}")
        if not callable(b.views):
            problems.append(f"layer {b.layer_id}: views is not callable")
        if b.role == "consumer" and (b.attention is None or not callable(b.to_attention)):
            problems.append(f"layer {b.layer_id}: a consumer binding names no attention call site or coordinates")
        if (b.layer_id, b.role) in seen:
            problems.append(f"layer {b.layer_id}: {b.role} described twice")
        seen.add((b.layer_id, b.role))
    return problems
