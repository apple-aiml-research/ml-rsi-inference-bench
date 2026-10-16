#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The fixed text the fidelity sequences are built from. Build-time input, never read at grading."""
from __future__ import annotations


P_KV_CACHE = """\
A serving engine keeps the attention keys and values of every token it has already processed, because
recomputing them for each new token would cost the same work again at every step. The store is called
the KV cache, and its size is the product of the layer count, the number of key-value heads, the head
dimension, two tensors, the bytes per element and the number of tokens resident. On a model with
thirty-six layers and eight key-value heads of sixty-four channels, one token of context costs a few
tens of kilobytes, so a hundred thousand tokens cost gigabytes. The cache is therefore the scarce
resource, not the parameters: weights are loaded once and shared by every request, while the cache
grows with each token of each sequence. Engines respond by paging the cache into fixed-size blocks,
sharing identical prefixes between requests that begin the same way, and evicting the least useful
blocks when a new request cannot be admitted. None of these mechanisms changes the arithmetic of
attention; they change which arithmetic is repeated and when the memory for it is reserved."""

P_MOE = """\
A mixture-of-experts layer replaces one large feed-forward network with many smaller ones and a
router that chooses a handful per token. The parameter count grows with the number of experts while
the work per token grows only with the number selected, which is why a model can carry a hundred and
twenty billion parameters and still decode at the speed of a much smaller dense network. The router
is a linear map from the hidden state to one score per expert, followed by a top-k selection and a
normalisation of the surviving scores into weights. Everything interesting follows from that
selection being discrete: the set of experts a token visits depends on the token, so batches become
ragged, the work per expert is data-dependent, and an implementation that shards experts across
devices has to move activations rather than weights. Quantising the experts compounds the problem,
because the dequantisation has to happen inside the same kernel that gathers the ragged batch, or the
bandwidth saved by the narrow format is spent again on writing wide intermediates to memory."""

P_QUANT = """\
Four-bit floating point is not one format but a family, and the differences decide whether two
implementations agree. A block of values shares one exponent scale, so the stored quantity is a
signed four-bit mantissa pattern indexed into a lookup table of sixteen representable values,
multiplied by two raised to the block's scale. Whether the scale is stored biased or unbiased, whether
the bias is one hundred and twenty-seven, whether the block spans thirty-two or sixty-four elements
along the reduction axis, and whether the accumulation happens in half or single precision are all
choices, and each of them shifts the result by more than the difference between correct answers. A
reference implementation that expands the blocks to sixteen-bit floats before multiplying will not
produce bit-identical results to a kernel that multiplies in the narrow format, and it should not: the
comparison worth making is whether the disagreement stays at the scale of arithmetic rounding or
grows to the scale of a wrong answer."""

P_SINKS = """\
Attention sinks are a small addition with a large effect. The softmax over a row of attention scores
must sum to one, so a query that has nothing useful to attend to still distributes its whole weight
over the available keys, amplifying noise. A sink is an extra learned logit added to the denominator
and to nothing else: the row's weights now sum to less than one, and a query with nothing to say can
say nothing. In implementation terms it is one term in one sum, which is exactly why it is easy to
drop by accident and hard to notice. A model that loses its sinks still loads, still serves, still
produces fluent text of the usual length, and still answers every question with confidence. What
changes is which answer, and only a measurement that scores the answers will see it."""

P_WINDOW = """\
Alternating attention patterns let a model pay for long context only where it needs it. In a
thirty-six layer stack, eighteen layers attend to the entire prefix and eighteen attend only to the
last one hundred and twenty-eight tokens, interleaved so that every second layer is cheap. The full
layers carry long-range structure; the windowed layers refine local detail. Because the pattern is a
property of the layer rather than of the request, an implementation has to plumb it from the
configuration through the memory pool to the kernel, and each hop is a place to lose it. Losing it in
the permissive direction -- treating a windowed layer as a full one -- costs memory and time and
produces text that still reads correctly, which makes it the more interesting failure of the two."""


P_NARRATIVE = """\
The harbour master kept two ledgers, and for eleven years nobody asked why. The first recorded what
the ships declared: crates of salt, bolts of cloth, a crated piano once, bound for a house that was
never built. The second recorded what the ships drew in the water when they arrived and when they
left, a column of numbers in a hand that grew smaller as the pages filled. He was not a suspicious
man. He simply believed that a thing which floats will tell you its weight whether or not its owner
will, and that a record costs almost nothing to keep and can only become more valuable. When the
inspector finally came, she spent a morning with the first ledger and an afternoon with the second,
and it was the column of numbers that told her which of the declarations had been written by someone
who had never seen the cargo."""

P_INSTRUCTION = """\
To laminate dough, begin with a rectangle of chilled butter enclosed in a rectangle of dough twice its
length. Fold the dough over the butter so the seams meet in the middle, then turn the parcel a quarter
turn and roll it out to three times its width. Fold it in thirds, chill it for half an hour, and
repeat. Each fold triples the number of layers, so four folds give eighty-one, which is as many as the
gluten will tolerate before the sheets begin to tear and the butter finds its way out. The chilling is
not optional and it is not about temperature for its own sake: warm butter absorbs into the dough
rather than separating it, and a laminated dough whose butter has been absorbed is simply a rich
bread. Work quickly, keep the corners square, and accept that the first attempt will be uneven in a
way that baking will make obvious."""

P_FORMAL = """\
The party of the first part shall maintain, for a period of not less than seven years from the date of
each transaction, records sufficient to establish the quantity, grade and origin of all materials
supplied hereunder, and shall make such records available for inspection upon fourteen days' written
notice. Failure to produce records within the notice period shall not of itself constitute a breach,
provided that the party demonstrates to the reasonable satisfaction of the requesting party that the
records exist and that the delay arises from circumstances outside its control. Nothing in this
clause shall be construed to require the disclosure of pricing information relating to transactions
with third parties, save where such information is necessary to establish origin and cannot
reasonably be redacted."""

P_ABSTRACT = """\
We study the extent to which the behaviour of a quantised language model can be attributed to its
quantisation rather than to the implementation of the kernels that consume it. Using teacher-forced
log-probabilities over a fixed set of sequences, we compare two independent implementations of the
same weights: one that expands the narrow format to sixteen-bit floats and multiplies in a general
matrix routine, and one that multiplies in the narrow format inside a fused kernel. We find that the
per-token discrepancy is concentrated in positions where the router's top-k selection is nearly tied,
that it does not accumulate along the sequence, and that a single dropped term in the attention
denominator produces a discrepancy two orders of magnitude larger than the arithmetic baseline. We
argue that a fidelity threshold calibrated on the agreement of two correct implementations is a
usable acceptance criterion for a third."""


P_RECALL_A = """\
Before anything else, three conventions. The station identifier for the northern site is BRAMBLE-7,
the calibration constant is nineteen point four, and the reference date is the second Tuesday of
March. Write them down; they will not be repeated when they are needed."""

P_RECALL_B = """\
Now, having established the conventions above, we can state the procedure. Readings are taken hourly
and normalised by the calibration constant given at the start of this document, which is nineteen
point four. Any reading that deviates by more than one part in forty from the running mean is
attributed to the northern site, whose identifier -- stated in the first paragraph and not since -- is
BRAMBLE-7, and is stamped with the reference date, the second Tuesday of March. A reader who has lost
the first paragraph cannot complete any of these three steps, which is the point of the exercise: the
dependency is real, it spans the whole passage, and no amount of local context will substitute for
it."""


P_CODE = '''\
def paged_attention(query, key_cache, value_cache, block_table, context_lens, scale, sinks=None):
    """One decode step of attention over a paged key-value cache."""
    batch, heads, head_dim = query.shape
    out = query.new_zeros(batch, heads, head_dim)
    for b in range(batch):
        length = int(context_lens[b])
        keys = gather_blocks(key_cache, block_table[b], length)
        values = gather_blocks(value_cache, block_table[b], length)
        scores = (query[b].unsqueeze(1) * keys).sum(-1) * scale
        if sinks is not None:
            m = torch.maximum(scores.max(-1).values, sinks)
            denom = (scores - m.unsqueeze(-1)).exp().sum(-1) + (sinks - m).exp()
        else:
            m = scores.max(-1).values
            denom = (scores - m.unsqueeze(-1)).exp().sum(-1)
        weights = (scores - m.unsqueeze(-1)).exp() / denom.unsqueeze(-1)
        out[b] = (weights.unsqueeze(-1) * values).sum(1)
    return out
'''

P_CONFIG = """\
[server]
host = "127.0.0.1"
port = 8000
tensor_parallel_size = 4
mem_fraction_static = 0.93
attention_backend = "triton"
cuda_graph_max_bs = 200

[model]
path = "/model"
dtype = "auto"
quantization = "mxfp4"
context_length = 131072
sliding_window = 128

[limits]
max_prefill_tokens = 16384
chunked_prefill_size = 8192
max_running_requests = 0
"""

P_JSON = """\
{"layer": 0, "kind": "sliding_attention", "window": 128, "heads": 64, "kv_heads": 8, "sinks": true}
{"layer": 1, "kind": "full_attention", "window": null, "heads": 64, "kv_heads": 8, "sinks": true}
{"layer": 2, "kind": "sliding_attention", "window": 128, "heads": 64, "kv_heads": 8, "sinks": true}
{"layer": 3, "kind": "full_attention", "window": null, "heads": 64, "kv_heads": 8, "sinks": true}
{"experts": 128, "experts_per_token": 4, "intermediate": 2880, "block": 32, "format": "mxfp4"}
{"tp": 4, "blocks_per_rank_floor": 22, "blocks_per_rank_ceil": 23, "columns_missed_floor": 64}
"""

P_ARITHMETIC = """\
2880 divided by 32 is 90. 90 divided by 4, rounded down, is 22. 22 times 32 is 704. 704 times 4 is
2816. 2880 minus 2816 is 64. 64 divided by 2880 is 0.0222, which is 2.22 percent. 90 divided by 4,
rounded up, is 23. 23 times 32 is 736. 2880 divided by 4 is 720. 736 is greater than 720 by 16. 736
times 4 is 2944, which exceeds 2880 by 64, so the last rank is clamped and covers 672. 672 plus 736
plus 736 plus 736 is 2880."""

P_REPEAT = """\
The token appears, the token appears again, the token appears a third time, and the count is three.
The token appears, the token appears again, the token appears a third time, and the count is six. The
token appears, the token appears again, the token appears a third time, and the count is nine. The
token appears, the token appears again, the token appears a third time, and the count is twelve. The
token appears, the token appears again, the token appears a third time, and the count is fifteen. The
token appears, the token appears again, the token appears a third time, and the count is eighteen."""

P_MARKDOWN = """\
## Acceptance, in three layers

| Layer | What it proves | What it cannot prove |
|---|---|---|
| L0 | the tree installs and the registry resolves the architecture | that a single answer is right |
| L2 | the served model reaches the quality bar upstream registered | that the arithmetic matches a reference |
| L3 | the kernels the change touches still pass their own tests | that the whole model is right |

A candidate that passes all three has done the work upstream's maintainers did. A candidate that
passes L0 and L3 and fails L2 has written something that runs and is wrong -- which is the common
failure and the reason the quality layer exists.
"""

PASSAGES = {
    "kv_cache": P_KV_CACHE,
    "moe": P_MOE,
    "quant": P_QUANT,
    "sinks": P_SINKS,
    "window": P_WINDOW,
    "narrative": P_NARRATIVE,
    "instruction": P_INSTRUCTION,
    "formal": P_FORMAL,
    "abstract": P_ABSTRACT,
    "recall_a": P_RECALL_A,
    "recall_b": P_RECALL_B,
    "code": P_CODE,
    "config": P_CONFIG,
    "json": P_JSON,
    "arithmetic": P_ARITHMETIC,
    "repeat": P_REPEAT,
    "markdown": P_MARKDOWN,
}

CONVERSATIONS = [
    ("low", "Explain why a serving engine caches keys and values.", "kv_cache", "abstract"),
    ("medium", "What decides whether two four-bit implementations agree?", "quant", "sinks"),
    ("high", "Why is a mixture-of-experts layer harder to shard than a dense one?", "moe", "window"),
    ("low", "Summarise the laminating procedure.", "instruction", "narrative"),
    ("medium", "What does an attention sink actually change?", "sinks", "quant"),
    ("high", "Walk through the sharding arithmetic at four-way tensor parallelism.", "arithmetic", "moe"),
    ("low", "Read this configuration back to me.", "config", "kv_cache"),
    ("medium", "What are the three conventions, and where were they stated?", "recall_a", "recall_b"),
    ("high", "Review this attention kernel for a dropped term.", "code", "sinks"),
    ("low", "Describe the record-keeping clause in plain words.", "formal", "narrative"),
    ("medium", "Which layers are windowed and which are full?", "json", "window"),
    ("high", "What can each acceptance layer not prove?", "markdown", "abstract"),
    ("low", "Count the repetitions.", "repeat", "arithmetic"),
    ("medium", "Why did the second ledger settle the question?", "narrative", "formal"),
    ("high", "Relate the window to the memory pool and the kernel.", "window", "code"),
    ("low", "State the calibration constant and the station identifier.", "recall_b", "recall_a"),
]

SEQUENCES = [
    ("plain_01", "hidden", 512, ["kv_cache", "sinks"]),
    ("plain_02", "hidden", 512, ["narrative", "instruction"]),
    ("plain_03", "hidden", 768, ["quant", "arithmetic", "moe"]),
    ("plain_04", "hidden", 768, ["code", "config"]),
    ("plain_05", "hidden", 1024, ["recall_a", "window", "moe", "kv_cache", "recall_b"]),
    ("plain_06", "hidden", 1024, ["formal", "abstract", "markdown"]),
    ("plain_07", "hidden", 1280, ["json", "arithmetic", "code", "config"]),
    ("plain_08", "hidden", 1280, ["repeat", "narrative", "repeat", "instruction"]),
    ("plain_09", "hidden", 1536, ["abstract", "quant", "sinks", "window", "moe"]),
    ("plain_10", "hidden", 1536, ["instruction", "formal", "narrative", "markdown"]),
    ("plain_11", "hidden", 1536, ["recall_a", "kv_cache", "moe", "quant", "window", "code", "recall_b"]),
    ("plain_12", "hidden", 1536, ["markdown", "json", "config", "code", "arithmetic", "repeat"]),
    ("broad_01", "hidden", 1536, ["recall_a", "kv_cache", "moe", "quant", "sinks", "window", "abstract",
                                  "formal", "narrative", "instruction", "markdown", "recall_b"]),
    ("broad_02", "hidden", 1536, ["code", "config", "json", "arithmetic", "repeat", "code", "config",
                                  "json", "arithmetic", "repeat"]),
    ("broad_03", "hidden", 1536, ["narrative", "instruction", "formal", "abstract", "narrative",
                                  "instruction", "formal", "abstract"]),
    ("broad_04", "hidden", 1536, ["repeat", "arithmetic", "repeat", "arithmetic", "recall_a", "repeat",
                                  "arithmetic", "repeat", "recall_b"]),
    ("public_01", "public", 512, ["kv_cache", "moe"]),
    ("public_02", "public", 768, ["code", "arithmetic"]),
    ("public_03", "public", 1024, ["narrative", "formal", "abstract"]),
    ("public_04", "public", 1536, ["recall_a", "window", "quant", "markdown", "recall_b"]),
]
SEPARATOR = "\n\n"
