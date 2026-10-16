# KDA mathematical resources

Kernel authoring is part of this task. These benchmark-supplied resources expose
forward mathematics and a serial reference, not an optimized implementation.
The implementation must preserve these semantics during prefill and decode and
be used by native model serving. The directory includes benchmark-supplied
loaded-layer inspection, sample and timing utilities (`client.py`, through the
worker extension `worker.py` that `launch.py` loads) and their interpretation
(`assessment.py`).

## The mathematics

The reference is a serial PyTorch delta recurrence (`recurrence.py`, the one function
extracted from flash-linear-attention; see `PROVENANCE.md`). It supplies the mathematical
operation, not an optimized chunked implementation. Inputs at the TP4 boundary use eight
heads, key and value dimensions 128, BF16 activations, and FP32 recurrent state in
`[sequence, head, key, value]` order. `mathematics.py` independently implements the
enclosing layer's projections, width-four depthwise convolution and its three BF16
histories, gates, recurrence and output normalization. Its projected output is reduced
independently across the four ranks.

For one token, with per-key-channel decay `g`, update the state as follows:

```
q = normalize(q, epsilon=1e-6) / sqrt(128)
k = normalize(k, epsilon=1e-6)
g = -exp(A_log) * softplus(raw_gate + dt_bias, beta=1, threshold=20)
decayed = exp(g)[..., None] * state
delta = beta * (v - k @ decayed)
next_state = decayed + outer(k, delta)
attention_output = q @ next_state
```

Q/K normalization accumulates in FP32, then rounds to the input dtype; keeping the normalized
Q/K in FP32 inside a fused kernel, without that rounding, is an accepted implementation, and
the accepted errors below allow for the difference. The recurrence keeps FP32 state; its
attention output rounds to BF16. `beta` is a sigmoid. The output RMS norm
uses the checkpoint's epsilon and weight and multiplies by the sigmoid of the output gate.
Each packed sequence has its own initial/final state. A valid continuation consumes the
prior state; a fresh prefill starts from zero.

## Numerical accuracy

`numerical-policy.json` defines the accepted approximation to the supplied
mathematics. Expected and returned tensors must have the same shape and finite
values. Relative RMSE is `RMS(actual - expected) / (RMS(expected) + 1e-8)`, with
statistics accumulated in FP32 as in `mathematics.error`.

| Result | Maximum relative RMSE |
|---|---:|
| Core attention output | 0.005 |
| Persistent state | 0.005 |
| Complete layer output, including output projection and TP reduction | 0.008 |

A maximum absolute error of at most `1e-6` also satisfies a comparison. These
requirements define accepted numerical results; they do not prescribe parameter
storage, fusion or accumulation strategy beyond the mathematical interface.

The supplied tools apply this policy and keep three results apart:
`evidence_complete` describes the measurement record; `sample_correctness`
(sample records) or `numerical_checks` (timing records) is `pass`, `fail` or
`unavailable` under the policy; `task_acceptance` is always `not_determined`,
because correct samples alone do not establish model support or efficiency.
Their exit status is 0 when the evidence is complete and every comparison meets
the policy, 1 when a comparison fails it, and 2 when the evidence is incomplete.
`assessment.py --mode sample|timing --input <record>` reports a saved record the
same way.

## Loaded-layer sample

With the ordinary server started by `/app/eval/kimi_linear/launch.py`, this
command runs four small cases on the first discovered native KDA layer of your
server and compares every step with the mathematics above:

```sh
python /app/eval/kimi_linear/kda/client.py --url http://127.0.0.1:8000 \
  --mode sample --output /tmp/kda-sample.json
```

The cases are those of `protocol.json` `sample`: a fresh eight-token prefill, a
continued eight-token prefill over two steps, a packed prefill of sequences of
5 and 11 tokens over two steps, and repeated decode of two sequences. The record
contains, for all four TP ranks, the discovered layers' identities and sources,
the input, output and state shapes, and per step the complete-output (including
output projection and TP reduction), core-attention and four state errors and
whether unowned state slots stayed bit-identical. It invokes your loaded layer
directly; it is not evidence that ordinary serving consumes that layer's output
and state, and it is not a comprehensive correctness suite.

## Standalone operator example

The standalone harness requires ordinary PyTorch and one GPU, with no FLA or engine import
on its reference side:

```sh
cd /app/eval/kimi_linear/kda
python operator_check.py --output /tmp/kda-reference.json
PYTHONPATH=/app python operator_check.py --adapter my_kda_adapter:evaluate --output /tmp/kda-operator.json
```

The first command exercises only the serial specification. For the second, provide your own
`evaluate` callable taking keyword arguments `mode`, `q`, `k`, `v`, `g`, `beta`, `states`,
`state_indices`, `has_initial_state`, `cu_seqlens`. Q/K/V and per-channel log decay are
`[1,total_tokens,8,128]`; beta is `[1,total_tokens,8]`. Q/K are unnormalized BF16 inputs;
the operation normalizes them with the equation above. Decay and beta are FP32. `states`
is the complete FP32 `[slots,8,128,128]` buffer. Prefill initializes selected slots to zero
where `has_initial_state` is false; valid decode uses the selected prior state. Return
`(output, final_states)` in the input activation dtype and complete state-buffer shape.
The examples include an eight-token fresh prefill, continued prefill, two small
packed sequences of lengths 5 and 11, and repeated two-sequence decode. One spare
state slot illustrates preservation of unowned state. They report core-attention
and state errors under the numerical policy above, with nonzero exit for invalid
or numerically failing samples. They do not assess the enclosing layer's complete
output or constitute a comprehensive correctness suite.

This optional development adapter does not constrain your kernel API and is not a graded
entry point. The grader instead calls the actual loaded layers and requires serving to
consume their output and state, through the native-layer mapping below.

## Measure native KDA performance

The supplied timing client measures the native layers loaded by your server,
using the same measurement code as evaluation. With the ordinary deployment
started by `/app/eval/kimi_linear/launch.py`, this command measures one public
prefill workload:

```sh
python /app/eval/kimi_linear/kda/client.py --url http://127.0.0.1:8000 \
  --mode timing --family 0 --output /tmp/kda-timing.json
```

Every family below is required for KDA efficiency. `--family` selects a public
measurement workload, not a correctness-test category:

| Index | Workload |
|---|---|
| 0 | One 128-token prefill |
| 1 | One 512-token prefill |
| 2 | One 2,048-token prefill |
| 3 | Four packed 2,048-token prefills |
| 4 | Decode batch 1 |
| 5 | Decode batch 3 |
| 6 | Decode batch 16 |
| 7 | Decode batch 64 |

Timing includes the complete loaded layer: projections, convolution, gates,
normalization, recurrence, persistent-state work, output projection and TP
reduction. MLA and MoE are outside this component measurement. Without
`--family`, all eight families are timed in order. The record contains the
measured layer timings on each TP rank and, for every family, numerical checks
of layers 0 and 19 before and after its timing; the report states whether those
checks meet the numerical policy. It reports speed without an efficiency
verdict: no speed bar is supplied. Correctness is defined by the mathematics and
numerical policy above and the task's required serving behavior; a timing
record does not establish an efficiency pass.

Each invocation -- one inspection, sample or timing-family RPC across the four
ranks -- has the published component-invocation limit of 1,800 seconds
(`../operation-limits.json`), which includes its setup, warmup, compilation,
repetitions, synchronization and readback.

## Alternative native layouts

The default mapping in `layer_adapter.py` follows this checkpoint's native layer layout.
Different names, layouts or fusion are allowed through a committed `/app/kda-verification.json`:

```json
{
  "layers": ["model.layers.0.self_attn"],
  "entry": "forward",
  "hidden_argument": "hidden_states",
  "output_argument": "output",
  "positions_argument": "positions",
  "metadata_key": "prefix",
  "epsilon": "o_norm.eps",
  "attention_probe": "o_norm",
  "weights": {"q_proj": {"path": "qkv.weight", "slices": [[0, 0, 1024]]}},
  "states": ["kv_cache.0.0", "kv_cache.0.1", "kv_cache.0.2", "kv_cache.0.3"],
  "lifecycle": {
    "consume_entry": "_forward",
    "state_indices": "non_spec_state_indices_tensor",
    "query_start_loc": "non_spec_query_start_loc",
    "valid_state": "has_initial_state",
    "consume_output_argument": "core_attn_out",
    "consume_token_axis": 1,
    "consume_returns_output": false
  }
}
```

The illustration abbreviates `layers`: supply all 20 distinct native KDA module paths.
Unspecified weight selectors use `DEFAULT_WEIGHTS` and unspecified `lifecycle` keys
`DEFAULT_LIFECYCLE`. A selector is an attribute/index path or
an object with `path`, optional `[dimension,start,end]` slices and an optional two-dimension
`transpose`. State selectors express the canonical orientation as views of the actual
persistent caches -- a fused q/k/v convolution state is three views of one tensor. They must
not allocate a substitute state.

`entry` is the **complete layer**: the call that takes the layer's hidden states (and, when
`positions_argument` is not null, the token positions) and writes the layer's result --
projections, convolution, gates, normalization, recurrence, persistent-state work, output
projection and TP reduction. For a return-value entry, set `returns_output: true`; a layer
without a positions argument sets `positions_argument: null`. Set `attention_probe: null`
for a fused enclosing operation. Controlled calls build the entry's attention metadata with
the loaded runner's own metadata builder for that layer, from the call's facts.

`lifecycle` names where the layer consumes its persistent state during serving, relative to
the mapped layer: `consume_entry` is the call inside the layer that reads and writes the state
(by default the inner operation the layer's opaque custom op runs; it may be the entry itself),
and `consume_output_argument`, `consume_token_axis` and `consume_returns_output` say where its
output is and which axis holds tokens. The consume output is the tensor that holds the consume
entry's result when it returns, which the layer reads after the call; a buffer the consume entry
itself consumes before returning is not its output, and an entry that writes the layer's final
result names that result. `state_indices`, `query_start_loc` and `valid_state`
are paths in the layer's native attention metadata: each row's state slot, the rows' query
offsets, and whether a row continues earlier state. `state_indices` and `query_start_loc` are
required; `valid_state` may be `null` when the layout has no such field. Serving observation
reads the actual state tensors and the actual output at this boundary; these selectors cannot
introduce candidate callables into the mathematical reference.

With the ordinary server started by `/app/eval/kimi_linear/launch.py`, the existing
inspection command resolves this mapping:

```sh
python /app/eval/kimi_linear/kda/client.py --url http://127.0.0.1:8000 \
  --mode inspect --output /tmp/kda-inspect.json
```

The result contains all four TP ranks' discovered layer identities, source
paths, state shapes and state strides. Missing ranks or discovery errors are
reported by the command. This
operation invokes no layer and no numerical comparison, and does not establish
that ordinary serving consumes every mapped output and state.

`PROVENANCE.md` and `source-manifest.json` identify the serial source and local
adaptations. No optimized FLA KDA implementation is supplied. Evaluation restores
its protected copy of these resources before checking the native implementation.
