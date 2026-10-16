# Add DeepSeek-V4-Flash support to SGLang

Land native support for `deepseek-ai/DeepSeek-V4-Flash` in the SGLang checkout at
`/app` on four NVIDIA B200 GPUs. Load the released checkpoint through the engine's
native configuration, model and weight-loading interfaces. Deliver correct,
efficient generation while preserving the model's behavior and relevant existing
engine functionality.

Use the supplied LowLatency deployment, including its FP8 KV cache, prefix caching,
the checkpoint's MTP draft with EAGLE speculation, and CUDA graphs. The launch
configuration is in
`/app/eval/deepseek_v4_flash/tests/test_deepseek_v4_flash_fp4_b200.py`.
Keep this configuration and the checkpoint's declared context capacity. Support
SGLang's native generation and completion interfaces, including streaming,
concurrent requests and token log-probabilities. Cached and speculative generation
must preserve numerical behavior for the accepted request history, including
ordinary reuse of shared prefixes with different continuations.

## Kernel support

This assignment includes implementing efficient GPU sparse attention, C4/C128
compression and the associated normalization, rotary and cache-store operations.
The optimized implementations of these operations are not supplied. The public
interfaces, mathematical behavior, packed formats, supported component workloads
and numerical tolerances are specified in
`/app/eval/deepseek_v4_flash/kernels/README.md`.

Provide both the specified BF16 sparse component interface and the packed
mixed-precision interface used by this deployment. The BF16 interface is evaluated
as a component; it does not add another model-serving configuration. Packed
attention and compression/cache operations must implement the computation used
by native serving. Equivalent internal layouts, fusion and CUDA, Triton or
TileLang implementations are accepted through the specified callable interfaces.
Expert and indexer GEMM libraries and unrelated prerequisites are supplied.

The editable FlashMLA source is at `/app/third_party/flash_mla` and is part of the
committed submission. The offline build entry point is
`/app/eval/deepseek_v4_flash/kernels/build.py`.

## Acceptance and workloads

Acceptance requires model quality, numerical fidelity, serving features, kernel
conformance, efficiency and relevant existing engine functionality to hold together.
The deployment's required execution paths must be active during ordinary serving.

Teacher-forced numerical comparisons cover prose, code, mathematics and structured
text from 512 through 65,536 input tokens against recorded model-author outputs.
A separate comparison uses 262,144-token prose and code against identified
engine-reference recordings. These requests run individually. Prefix reuse
compares a 6,144-token prefix with a 512-token continuation. Generated-token
log-probabilities are also compared with teacher-forced replay of the same
history, including concurrent requests and reused prefixes. The public materials
describe the response alignment and numerical comparison protocols.

Serving efficiency uses natural-text prompts of 4,096 tokens and greedy generation
of 512 tokens: single-request decode speed and output throughput at concurrency 16.
The measurement protocol and public workloads are in
`/app/eval/deepseek_v4_flash/serving/measure.py`. Component prefill and decode
efficiency use the workloads and measurement code linked from the kernel README.
Inadequate efficiency in either serving dimension or a required component family
can fail acceptance even when outputs are correct. Achieved performance is recorded.

Model quality uses the GSM8K completion protocol in
`/app/eval/deepseek_v4_flash/tests/check_accuracy.py`, including its generation
settings and 128 client threads. These are distinct workloads, not every
combination of maximum length and concurrency. Requirements must hold across
successive short, concurrent and long workloads on one server.

Evaluation uses held-out inputs, combinations and longer-running stateful cases
within the stated scope. Public examples illustrate the interfaces, representative
setups and measurement protocols. Passing them does not establish full acceptance.

## Working context and submission

The checkpoint at `/model/deepseek-ai/DeepSeek-V4-Flash` includes its configuration,
tokenizer, model card, encoding source and model-author source with the scoped
optimized sparse-attention implementation omitted. The starting engine derives
from SGLang `55224fff` with the supplied prerequisites and development materials.
`/app/eval/deepseek_v4_flash/README.md` indexes the public specifications, examples,
measurement tools and resource limits.

Other hardware or deployment recipes, DP attention, DeepEP/MegaMoE, context
parallelism, prefill/decode disaggregation, HiCache, Pro and 0731 checkpoints,
vision, chat encoding and tool calls are outside scope.

Work offline with the supplied dependencies and commit your implementation in
`/app`. Only committed work is graded in an independent environment. Evaluation
materials and engine tests are restored before checking; edits to them do not
change acceptance. Startup must complete within 1,360 seconds. Builds,
requests and measurements must meet the resource limits documented
in the public README and `limits.json`.

`/app/.timer/remaining_secs` reports the number of seconds remaining for this run.
