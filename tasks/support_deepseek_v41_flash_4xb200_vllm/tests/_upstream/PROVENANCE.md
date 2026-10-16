# Vendored upstream material, with provenance

## The maintainers' tests for this model (`tests/test.patch`)

`test.patch` is `git diff --binary e83d57d8ae14 e77daef89e18 -- <eleven paths>` in `vllm-project/vllm`:
the eleven test paths as the runtime-integration squash (e77daef89e18, #56214) left them, against the
base. It is shipped twice, byte-identical: to the verifier here, where test.sh applies it after restoring
`/app/tests` to the base, and to the agent at `/app/eval/deepseek_v41_flash/tests/upstream.patch`, with
`selection.json` beside it.

| path | what |
|---|---|
| `tests/models/registry.py` | the `DeepseekV41ForCausalLM` and `DSparkV41DraftModel` example entries (and, from an intervening main commit, one revision pin for another model) |
| `tests/models/test_registry.py` | one line |
| `tests/kernels/test_compressor_kv_cache.py` | new |
| `tests/kernels/test_engram.py` | new |
| `tests/kernels/test_fused_indexer_q_rope_quant.py` | extended |
| `tests/kernels/core/test_fused_q_kv_rmsnorm.py` | new |
| `tests/kernels/test_mhc_kernels.py` | extended |
| `tests/models/test_deepseek_v4_mega_moe.py` | extended (by both squashes) |
| `tests/models/test_dspark_mla.py` | new |
| `tests/quantization/test_fp8.py` | extended |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | extended |

The nine test files are exactly the integration squash's test files that import the model package
(`vllm.models.deepseek_v4_1`); its other fourteen test files do not. `tests/models/test_initialization.py`
is the base's own file, unchanged, reading the registry entry above.

## The efficiency gate

`tests/infb_efficiency/` is this task's own efficiency implementation. It began as the task's verbatim copy of the
benchmark's Day-0 efficiency implementation and keeps only what this task runs;
`tests/infb_efficiency/PROVENANCE.json` records the source's file digests, every local change and the current
digests. Its client is the agent's `/app/eval/deepseek_v41_flash/serving/client.py`, byte for byte.

## The evaluation materials

`tests/eval/deepseek_v41_flash/` is the grading copy of the agent's `/app/eval/deepseek_v41_flash/`
(`environment/eval/`, byte for byte): the hidden wrappers import their checking code from it
(`infb_verify/_trusted.py`).

## The hidden material

- `hidden/gsm8k_slice.json`: 200 indices into the public GSM8K test split, seed 20260924.
- `hidden/fidelity.json`: 32 token sequences from the hidden text pool with the model-author
  implementation's per-token log-probabilities; its `provenance` block says what ran.
