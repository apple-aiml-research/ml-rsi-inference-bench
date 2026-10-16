# Attention numerics

This note states how the output of a complete DSA attention sublayer of GLM-5.3-Flash is compared with the model
author's computation of the same sublayer. It defines a numerical boundary and one allowance. It does not say which
layers, inputs or positions an evaluation compares, and a comparison you run yourself is not a task verdict.

## The compared quantity

For a sequence and a DSA attention layer (a layer whose `layer_types` entry in `config.json` is
`deepseek_sparse_attention`):

- **input**: the sublayer's input hidden states for every position of the sequence, as the layer hands them to its
  attention (after the layer's hyper-connection pre-mix and input RMSNorm), with the positions and causal history;
- **output**: the sublayer's result at the compared positions, after the output projection and the tensor-parallel
  reduction, before the residual and hyper-connection combination.

An implementation may hand the attention's result to the rest of the layer either as each rank's tensor-parallel partial
(reduced afterwards) or already reduced on every rank; both are compared as the reduced sublayer output. Where every
rank reports the same result at a position -- each rank's within `2^-6` of the first rank's in RMS relative to its
scale, four BF16 units -- the result is taken as already reduced and one copy is compared; otherwise the ranks' results
are partials and their sum is compared. A sequence whose positions mix the two layouts is read position by position.

The author computation is `Glm5NextTextAttention` with its `Glm5NextTextIndexer` in
`reference/modeling_glm5_next.py`, evaluated on the same input history with causal masking and the release's
selection semantics, using the checkpoint's own weights.

## Representation boundaries of the author computation

The author computation accumulates in FP32 and keeps these representation boundaries of the declared deployment on
four B200 GPUs:

- the FP8 projections (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj`) as the engine's FP8 GEMM path encodes
  them on B200: the checkpoint's e4m3 128 x 128 weight blocks with power-of-two (UE8M0) block scales, and each
  activation quantized per token in groups of 128 channels to e4m3 with a power-of-two scale;
- the BF16 parameters (`kv_b_proj`, the indexer's projections, the norms) in BF16, with results rounded to BF16 where
  the release rounds them;
- the attention latent that the FP8 KV cache holds in e4m3 at unit scale, for every key and value position.

## The metric and the allowance

For a compared output tensor `c` (positions x hidden size) and the author's `a`:

    E = RMS(c - a) / (RMS(a) + 1e-8)

The representation's own effect on the same tensor is `E_rep = E(a, a_exact)`, where `a_exact` is the author
computation on the same input history in exact arithmetic on the checkpoint's stored parameters: its e4m3 128 x 128
weight blocks dequantized with their float32 block scales, the activations unquantized (FP32 products of the BF16
inputs) and a BF16 latent, everything else as above. The tensor is within the policy when

    E <= sqrt(2) * max(E_rep, 2^-8)      or      max |c - a| <= 1e-6

sqrt(2): an implementation of the same mathematics under the same representation boundaries departs from exact
arithmetic by one draw of the representation's error, as the author computation does, and two such draws differ by at
most sqrt(2) times one. 2^-8, BF16's unit roundoff, is the floor: both outputs are BF16, and a difference below it is
not attributable to the FP8 boundaries. The constants are fixed by this policy, not measured. Every compared tensor
must be within the policy on its own, with its own `E_rep`; there is no averaging across tensors, layers or requests.
`glm53_protocols/attention.py` implements the metric, the allowance and the escape.

## The example

`fidelity/attention_example.json` holds one recording of the author implementation on the actual checkpoint:
the first DSA attention layer (index 3) on the first 32 tokens of the public record `code-5120`
(`fidelity/examples.json`). Each tensor is carried under `tensors` as its raw little-endian bytes, base64-encoded,
with its dtype, shape and the SHA-256 of the bytes.

| Tensor | Shape | Dtype |
|---|---|---|
| `input_hidden_states` | 32 x 4096 | bfloat16 |
| `output` | 32 x 4096 | bfloat16 |
| `input_ids` | 32 | int64 |
| `positions` | 32 | int64 |

The same file records the provenance: Transformers 5.17.0 `Glm5NextForConditionalGeneration` with
the checkpoint's fine-grained FP8 quantization on four B200 GPUs, SDPA attention, one teacher-forced forward without a
KV cache, with the input and output taken at the sublayer boundary above. For this tensor (layer 3, these 32 positions)
the author computation under the deployment encodings differs from exact arithmetic on the stored parameters by
`E_rep = 0.03552`, so its allowance is `sqrt(2) * 0.03552 = 0.05023`. The recording is a third reading -- it keeps the
checkpoint's float block scales, quantizes activations with float group scales and keeps a BF16 latent -- and differs
from the deployment-encoded author computation by `E = 0.04165` (largest absolute difference 0.0957), within this
tensor's allowance, and from exact arithmetic by `E = 0.02162`. For a sense of scale, the recorded output rounded to
e4m3 differs from itself unrounded by `E = 0.02656`.

```python
import base64, json, sys
import torch
sys.path.insert(0, "/app/eval/glm53_flash")
from glm53_protocols.attention import relative_rms_error, max_abs_difference, within_policy

record = json.load(open("/app/eval/glm53_flash/fidelity/attention_example.json"))
t = {name: torch.frombuffer(bytearray(base64.b64decode(v["base64"])), dtype=getattr(torch, v["dtype"])).reshape(v["shape"])
     for name, v in record["tensors"].items()}
rounded = t["output"].float().to(torch.float8_e4m3fn).float()   # an illustrative perturbation
e = relative_rms_error(rounded, t["output"])                     # 0.02656
within_policy(e, max_abs_difference(rounded, t["output"]), e_rep=0.03552)   # this tensor's E_rep: True
```

The example illustrates the boundary and the metric. It does not locate layers in an engine and it is not a
comparison harness.
