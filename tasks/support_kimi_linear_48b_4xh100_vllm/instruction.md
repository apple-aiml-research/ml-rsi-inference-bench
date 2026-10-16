# Support Kimi-Linear-48B-A3B-Instruct in vLLM

Implement efficient Kimi Delta Attention forward-inference kernels and land
native vLLM support for `moonshotai/Kimi-Linear-48B-A3B-Instruct`. Work in the
prepared engine checkout at `/app` and load the full checkpoint at `/model`
through vLLM's native model and weight-loading interfaces. Preserve model
behavior and relevant existing engine functionality, including scheduling and
cache management for already-supported models.

The environment provides eight NVIDIA H100 80GB GPUs. Support the TP4/BF16,
two-replica TP4 data-parallel, and TP4 prefix-caching deployments in
`/app/eval/kimi_linear/launch.py`, including its capacity-limited caching
variant. These use chunked prefill, compiled execution with full CUDA graphs
for ordinary decode, and FlashMLA. Reliable completion/chat serving,
concurrent requests, prompt/output token
log-probabilities and numerical fidelity are required. Data parallelism and
prefix caching are separate required deployments.

Preserve each request's model state and numerical behavior throughout its
lifetime, including concurrent serving, cancellation and resumption under
memory pressure. With prefix caching enabled, reuse completed common prefixes
across requests and independent continuations. Reusable prefixes include
previously generated tokens. Cache reuse, eviction and subsequent computation
must preserve the behavior of independently processing each request's complete
token history. Recomputing all repeated prefixes does not satisfy prefix-cache
support; valid recomputation when a reusable checkpoint is unavailable is allowed.

Preserve quality and fidelity to the supplied model and KDA mathematics.
`/app/eval/kimi_linear/kda/numerical-policy.json` specifies the accepted error
for attention, persistent state and complete-layer output. The state-lifecycle
README and policy specify continuation equivalence, reuse observations and
representative examples. The fidelity materials distinguish model-author
recordings from longer-context engine-reference recordings. Native serving
must consume the implementation delivered for prefill, decode and state reuse.

Efficient native KDA execution and production serving are required at the
cache-disabled TP4 deployment. Per-user decode speed and concurrent output
throughput use `/app/eval/kimi_linear/serving/client.py measure`. The KDA
README defines complete-layer measurement and gives a timing command.
Inadequate KDA or serving performance can fail acceptance even when outputs
are correct. Achieved performance is recorded with the result.

All required correctness, serving, efficiency and preservation criteria must
pass together. Evaluation uses held-out inputs within the stated scope.
Public examples illustrate the protocols; passing them does not establish
task acceptance.

## Workload scope

Numerical and isolation workloads include prompts from one through 12,000 tokens and
continuations through 1,024 tokens, plus a cache-disabled TP4 long-context
protocol at 64,000 input and 512 output tokens. The configured context
capacity is separate from realized workload lengths. State-lifecycle workloads
include repeated, branched, generated-history, mixed fresh/cached and resumed
requests. The launch documentation specifies the caching block size and the
capacity-limited variant used for turnover and preemption. These are declared
workloads, not every combination of maximum context, concurrency and deployment.

Quality uses the GSM8K protocol in
`/app/eval/kimi_linear/tests/acceptance.py`. Serving efficiency retains natural
prompts of 4,096 tokens and 512 generated tokens, at single-request decode and
throughput concurrency 64. KDA efficiency covers the eight complete-layer
families in its README. Ordinary requirements hold across successive workloads
on one server. Serving-consumption observations use separate instrumentation;
efficiency is measured without it.

## Working context and adaptation

This task includes kernel authoring and hybrid prefix-state support on the
prepared historical engine. Generic operations and dependencies are supplied;
optimized KDA, native Kimi integration and completed Kimi prefix-state support
are not. The KDA materials supply forward mathematics and a serial reference.
Their README documents semantic mappings for different native layer layouts.

The checkpoint revision is `e1df551a447157d4658b573f9a695d57658590e9`.
Model-author source in `/app/eval/kimi_linear/reference/` is available to
inspect, not runnable with the installed dependencies. `router-semantics.md`
is authoritative for the documented routing discrepancy. The main evaluation
README gives launch, rebuild and example entry points.

Quantization, speculation, combined DP2-plus-prefix-cache serving, other TP
sizes, pipeline/expert/context parallelism, other hardware, tool-call/reasoning
parsers and later engine API migrations are outside scope.

## Submission

Work offline and commit the implementation in `/app`. Only the final committed
tree is evaluated in an independent container. Compiled-source changes trigger
the supplied rebuild. Evaluation materials and upstream tests are restored;
edits to them, cached artifacts and running servers are not submitted.
Single operations are bounded by the published clocks and limits in
`/app/eval/kimi_linear/operation-limits.json`. `/app/.timer/remaining_secs`
reports the number of seconds remaining for this run.
