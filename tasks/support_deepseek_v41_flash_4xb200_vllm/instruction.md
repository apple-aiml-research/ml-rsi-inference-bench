# Support DeepSeek-V4.1-Flash in vLLM

Land native support for `deepseek-ai/DeepSeek-V4.1-Flash` in the vLLM checkout at
`/app` on four NVIDIA B200 GPUs. Load the released checkpoint through vLLM's
native model and weight-loading interfaces, preserving the model's behavior and
relevant existing engine functionality.

The required serving configuration is defined in
`/app/eval/deepseek_v41_flash/recipe.py`: tensor parallelism four, native sparse-MLA
attention, the supplied CUDA-graph configuration and DSpark using
the checkpoint's draft weights. Keep this deployment working through the
standard completion and chat APIs, including streaming, concurrent use and
token log-probabilities. Preserve model quality and numerical fidelity to the
supplied model-author implementation.
The recipe documents both its launch arguments and the required resolved engine
configuration.

Implement the MXFP4 sparse-indexer operations in the supplied editable dependency
and integrate the model's two-level indexing into native prefill and decode,
including target verification. Consumer indexers must select positions using their
own scores within the model's shared candidate pool; native attention must use those
selections. The required library interfaces, numerical contract, supported inputs
and component measurement interface are defined in
`/app/eval/deepseek_v41_flash/sparse_indexer/README.md`. Correctness and component
efficiency are required separately for prefill and decode.

Efficient production serving is part of acceptance. The deployment's required
execution paths must be active during ordinary serving. Per-user decode speed
and concurrent output throughput are evaluated using the measurement protocol
in `/app/eval/deepseek_v41_flash/serving/client.py measure`; the README beside it
gives a command using the public workload. Inadequate performance on either
dimension can fail acceptance even when outputs are correct. Achieved
performance is recorded with the result.

Component efficiency measures the complete four-consumer indexer pipeline, including
candidate/metadata preparation, score computation, selection and remapping, from
quantized inputs through selected logical positions. Each consumer has its own
queries and weights over the model's shared index keys. The sparse-indexer README
gives the workload domain and measurement commands. Component measurements and
whole-serving measurements are separate requirements.

Acceptance requires the specified model behavior, serving features, efficiency
and relevant existing engine functionality to hold together. Evaluation covers
these requirements on held-out inputs, including longer requests and concurrent
use within the stated deployment. Public examples illustrate API use and
measurement protocols; passing them does not establish task acceptance.

## Workload scope

Teacher-forced numerical comparisons cover short inputs through 20,480 tokens.
The deployment's 32,768-token configured capacity is distinct from that numerical
scope. Greedy requests must be deterministic and concurrent requests independent.
Quality uses the supplied GSM8K protocol in
`/app/eval/deepseek_v41_flash/tests/quality.py`, which defines generation and
request scheduling. Serving efficiency uses natural-text prompts of 4,096 tokens
and 512 generated tokens, with single-request decode and throughput at
concurrency 16. These are distinct workloads, not every combination of maxima.

The sparse-indexer operator contract covers packed and paged inputs through 32,768
visible compressed positions, with the model geometry and input-shape limits
specified in its README. Native indexing must preserve each query's visible history
throughout the configured serving execution. Component limits do not require the
full model to serve every combination of operator maxima.

Serving requirements must hold across successive measurements and correctness
requests on one server. Efficiency may also be measured after a fresh launch at
the same configuration. Standalone compatibility checks run separately.

## Working context

`/model/deepseek-ai/DeepSeek-V4.1-Flash` contains the pinned checkpoint, model card,
configuration, tokenizer, `inference/` model-author source and `encoding/` sources.
The model-author implementation is also installed as a separate runtime:
`/opt/model-author` holds the released `inference/` code with its own Python
environment, and `/model-author/deepseek-v41-mp4` holds the checkpoint converted by
the author's own tooling for four-way model parallelism. It uses the same four
GPUs as the server, so it cannot run while a server holds them.
The prepared engine includes the existing frontends and prerequisite kernels. An
editable dependency under `/app/third_party/deepgemm_v41_sparse` supplies generic
runtime/build support and top-k dependencies; its targeted sparse-indexer kernels
and model integration remain to be implemented. This is an adapted historical
model-support assignment that includes that dependency work.

`/app/eval/deepseek_v41_flash/README.md` gives the launch and rebuild entry points,
the author runtime's commands and representative API examples. The engine's
existing tests that must keep passing and offline datasets are supplied for
development.

Image/video inference, AMD or SM12, data/expert parallel serving, prefill/decode
disaggregation, KV connectors, NVFP4 compressed KV cache, mega attention,
optimized SWA-bounded replay, prefix caching, Engram host offload or data-parallel
sharding, million-token contexts and new parser extensions are outside scope.

## Submission

Work offline in `/app` and commit the implementation. Only the final committed
tree is submitted; local caches and running processes are not. Evaluation
restores trusted tests and materials independently of development edits.
Correctness, compatibility and efficiency are all required.

Server startup is bounded at 7,000 seconds and a compiled-source rebuild at
4,580 seconds; the other operation limits are listed in
`/app/eval/deepseek_v41_flash/README.md`.

`/app/.timer/remaining_secs` reports the number of seconds remaining for this run.
