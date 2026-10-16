# Native component accuracy and resources

The packed experts used by serving must implement the released checkpoint's
operation, defined by `expert_math.py`, including the complete TP4 result.
`attention.py` defines the corresponding attention operation. The component
requirements below apply in addition to correct full-model serving.

## Numerical results

`numerical-policy.json` is the accuracy contract. It defines two operations:

| Operation | Maximum normalized error | RMS normalized error |
|---|---:|---:|
| Complete TP4 expert output, for checkpoint or generated weights | 0.08 | 0.008 |
| Complete attention output | 0.02 | 0.004 |

For a reference tensor `r` and candidate tensor `a`, normalized error is
`abs(a - r) / max(RMS(r), abs(r), 1e-30)` elementwise, as implemented by
`numerics.error`. Both statistics must satisfy their limits, and compared values
must be finite and have the expected shape. These are approximation limits for
the mathematical result, not requirements to reproduce an optimized reference's
layout, fusion or accumulation dtype. Full-model numerical comparisons use the
separate log-probability protocols in `fidelity.py`.

## Admitted operation domain

The expert operation is admitted on the released dimensions -- hidden and
intermediate size 2,880, 128 experts, four distinct routed experts per token,
tensor parallelism four -- for 1 to 4,096 tokens per invocation. Native padding,
sharding and fusion are free within the resource policy below.

- Packed E2M1 values use all sixteen nibble codes, lower nibble first, with one
  E8M0 scale per 32 logical values; scale byte 255 is invalid. The FP32 decoding
  and every value cast at a declared BF16 boundary are finite.
- Activations are finite BF16. Biases hold the released checkpoint's
  BF16-representable values, which the mathematical reference keeps in FP32.
- Route IDs are valid and distinct within a token. Route probabilities are
  finite, nonnegative and sum to one within FP32 rounding; zero probabilities
  are allowed.
- The operation is the complete `expert_math.py` computation: scale decoding,
  both affine biases, the gate clamped at a maximum of 7, the up projection
  clamped to [-7, 7], `gate * sigmoid(1.702 * gate) * (up + 1)`, the weighted
  combination of the selected experts, every TP tail and the complete output.
  Every intermediate and result of the reference is finite.

The attention operation is admitted on the released TP4 dimensions (16 query
heads, two KV heads, head width 64) with already rotary-encoded finite BF16
queries, keys and values at consecutive causal positions, full attention or the
released 128-token sliding window, and finite FP32 sink scores: each head's sink
enters the softmax denominator and contributes a zero value. One invocation has
1 to 512 query tokens and at most 16,384 positions of history per sequence, the
query included.

`expert_math.admit` and `attention.admit` state these checks. Inputs outside the
domain are an evaluation-data fault, not an accuracy test of an implementation.

## Resident storage

Persistent expert storage is bounded by **576 MiB per layer per TP rank**, and
**36 × 576 MiB per rank** for the full model. This budget includes packed
values/scales, padding, helper tensors, and retained source or expanded copies.
The full-model allowance additionally provides **4 GiB per rank** for ordinary
model weights and support allocations. Request/KV pools are accounted separately,
and the complete serving deployment must fit each GPU.

These budgets require a compact resident representation while allowing native
layout and padding choices.

## Component workspace

The workspace budget per rank depends on the number of tokens in one complete
expert operation:

| Tokens in the operation | Workspace ceiling |
|---|---:|
| 1–64 | 64 MiB |
| 65–256 | 128 MiB |
| 257–1,024 | 256 MiB |
| 1,025–4,096 | 768 MiB |

Temporary conversion must fit this budget; it does not exempt retained expert
copies from the resident-storage limit.

## Accounting

Unique live storage ranges count once, including packed values and scales,
native padding, helper data and any retained source or expanded copies. Aliased
views do not multiply storage, and a second retained allocation does not
disappear because it holds the same contents. Live allocations outside the
framework allocator count: device use that grows beyond the allocator's
reservation is an external allocation. Reserved but unused allocator capacity is
not resident memory. Ordinary, expert, request/KV and workspace allocations are
disjoint categories.

Baselines are synchronized readings taken after distributed and CUDA
initialization. Initialization is complete first: communicators establish their
buffers lazily, so the default and tensor-parallel groups' collectives are
exercised once at the operation's smallest and largest messages, eagerly and
under graph capture (`native_experts.complete_collectives`). Each operation's
baseline is read after releasing the allocator's cached blocks. In graph mode one
complete invocation is one captured graph: the timing rotates eight captured graphs
(one per input slot) only to rotate inputs, so the invocation's setup peak,
retained scratch and device use outside the allocator (including the graph's own
instantiation) are read after the first capture, and the rotation's totals are
recorded beside them. For one complete operation, `setup_peak_workspace` is the peak
of setup (warmup, compilation and graph capture) above the persistent
allocations, `retained_workspace` is the scratch held between calls, and
`steady_peak_workspace` is the additional live scratch above that retained
baseline during the operation. The accepted peak is
`max(setup_peak_workspace, retained_workspace + steady_peak_workspace)` plus
external allocations held over the operation. A negative delta, or device use
below the allocator's reservation growth, is invalid accounting, never zero
usage. Eager and graph execution, all four ranks and the actual production
storage are measured.

`expert_timing.py` reports component timing and these storage and workspace
observations with their baselines. Setup and steady-state time are separate
measurements. Numerical and resource requirements continue to apply while
measuring efficiency; measurements alone do not establish task acceptance.
