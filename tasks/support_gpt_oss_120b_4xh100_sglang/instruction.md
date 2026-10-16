# Support GPT-OSS-120B in SGLang

Land native support for `openai/gpt-oss-120b` in the prepared SGLang checkout at
`/app` on four NVIDIA H100 80GB GPUs. Load the released packed MXFP4 checkpoint
through the engine's native configuration, model and weight-loading interfaces.
Implement the missing packed-expert computation and model integration, preserving
the model's numerical behavior and relevant existing engine functionality.

The intended serving command is:

```sh
python -m sglang.launch_server --model-path openai/gpt-oss-120b --tp 4 \
  --cuda-graph-max-bs 200 --mem-fraction-static 0.93
```

Keep ordinary decoding, tensor parallelism four, CUDA graphs and the engine's
Triton attention for this architecture. No attention-backend or tool-parser flag
is supplied. Load the weights as released without replacing or pre-converting the
checkpoint on disk. Deliver efficient native execution with compact resident
expert storage, and fit model storage, working memory and serving within the
available hardware. `/app/eval/gpt_oss/component-contract.md` defines the
admitted component operation domain, the storage/workspace budgets and their
accounting; `/app/eval/gpt_oss/numerical-policy.json` defines the numerical
accuracy required of complete expert outputs, for checkpoint or generated
weights, and of attention outputs.

Serve `/generate` and `/v1/chat/completions`, including token log-probabilities,
streaming, reasoning-effort controls and standard function-call conversations
through the checkpoint's Harmony format. Support one function call per assistant
turn and subsequent tool-result turns; simultaneous multiple calls in one turn
are outside scope. Preserve model quality, numerical fidelity and normal API
semantics under concurrent use.

Efficient packed-expert execution and production serving are part of acceptance.
The deployment's required execution paths must be active during ordinary serving.
Per-user decode speed and concurrent output throughput are evaluated using
`/public/measure_serving.py`; `/app/eval/gpt_oss/README.md` defines component
measurement and gives commands for both protocols. Inadequate packed-expert or
serving performance can fail acceptance even when outputs are correct. Achieved
performance is recorded with the result.

Acceptance requires correct and efficient native packed-expert execution, the
specified model and Harmony serving behavior, and relevant existing engine
functionality to hold together. Evaluation covers these requirements on held-out
text, reasoning and tool conversations within the stated deployment. Public
examples illustrate model arithmetic, API use and measurement protocols;
passing them does not establish task acceptance.

## Workload scope

Teacher-forced numerical comparisons include text through 16,384 tokens, separate
from the configured 131,072-token capacity. Concurrent serving includes the
batches supported by the deployment's CUDA graphs. Decode and fresh-prefill
log-probabilities must agree for the same continuation under the supplied
comparison protocol.

Model quality uses GPQA diamond at low reasoning effort: up to 198 concurrent
questions, at most 4,096 generated tokens and temperature 0.1. The supplied
`/app/eval/gpt_oss/acceptance/run_eval.py` defines request formatting and answer
scoring and includes a small public example set.

Whole-serving efficiency uses natural-text prompts of 4,096 tokens and greedy
generation of 512 tokens, measuring single-request decode and throughput at
concurrency 64. Packed-expert efficiency covers the complete native TP4 operation
from decode to prefill counts of 1 through 4,096, with balanced and skewed routing;
`/app/eval/gpt_oss/expert_timing.py` defines the measured workloads and graph/eager
execution.
Both routing families and the declared measured sizes are required. These are
distinct workloads, not every combination of their maxima.

Serving requirements must hold across successive workloads at the specified
deployment. Instrumented consumption checks and standalone component measurements
are separate from ordinary whole-serving efficiency measurements.

## Working context

`/model` contains checkpoint revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`;
its repository ID resolves offline. Model-format resources are in
`/public/gpt_oss/` and `/public/harmony/`. `/app/eval/gpt_oss/README.md` introduces
the development materials, representative numerical examples and the optional adapter
interface for alternative native layouts. Optimized packed-expert computation
is part of this task; the supplied mathematics is not an optimized implementation.

The episode has 32 CPUs, 512 GiB RAM, 2 TiB storage and 64 GiB shared memory.
Network access is disabled. Server startup must complete within 900 seconds;
`/app/eval/gpt_oss/README.md` lists this and the other operation limits
evaluation applies.

## Submission

Commit your implementation in `/app`, including any changes to its editable
kernel package. Only committed changes from the prepared base are submitted.
Evaluation materials and upstream tests are restored independently; edits to
them, local caches and running processes do not change acceptance.

`/app/.timer/remaining_secs` reports the number of seconds remaining for this run.
