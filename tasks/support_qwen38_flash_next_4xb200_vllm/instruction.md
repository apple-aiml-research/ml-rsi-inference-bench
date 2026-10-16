# Support Qwen3.8-Flash-Next-FP8 in vLLM

Implement native support for `Qwen/Qwen3.8-Flash-Next-FP8` in `/app`, using the full
checkpoint at `/model` on four NVIDIA B200 GPUs. Support the ordinary and
three-token MTP deployments defined by `/app/eval/qwen38/serving/recipes.py`.

The starting tree supplies the engine dependencies, HC, low-latency GEMM and
GDN sigmoid helpers, and a generic block-FP8 repair. Model integration and
optimized QSA and PLE implementations are required work.

## Required support

- Load the released checkpoint and register its declared architecture through
  vLLM's native model interfaces. Preserve the recipes' parallel layout, FP8/MoE
  and GDN execution, KV precision, compilation and CUDA graphs, chunked prefill,
  aligned prefix caching and context configuration.
- Serve text and chat completions, streaming, greedy generation, token
  probabilities and prompt probabilities. Preserve each request's model state
  through continued serving and prefix reuse.
- Compute QSA and PLE according to the supplied model semantics, including
  persistent state through prefill, decode and speculative execution. Their
  measured implementations must be the native implementations used by the model.
- With MTP3, use the model's draft head, preserve the target model's generation
  semantics, and handle accepted and rejected proposals with correct state.
- Meet the model-quality requirements and preserve the applicable base-engine
  behavior. The evaluation protocol and recipe-to-check mapping are documented
  in `/app/eval/qwen38/README.md`.

Optimize complete serving performance in both deployments: per-user decode speed
at concurrency 1 and output throughput at concurrency 32. Optimize ordinary QSA
prefill, both deployments' QSA decode, and both deployments' PLE prefill/decode.
MTP3 QSA prefill remains subject to correctness and native-integration requirements;
it has no separate component timing gate. PLE auxiliary memory must remain within
the allowance in `/app/eval/qwen38/scope.json`.

Correctness, native integration, efficiency, memory and compatibility must hold
together for the same committed implementation. Equivalent algorithms, helper
names, fusion and cache layouts are allowed. The measurement interface in
`/app/eval/qwen38/measurement/README.md` identifies native regions and supports
enclosing fused regions; it does not require a particular internal cache format.

## Scope and evaluation resources

The recipes retain the checkpoint's native 262,144-token context configuration.
Independent fidelity evaluation covers sequences up to 18,432 tokens. Component
and stateful workloads have the finite ranges in `/app/eval/qwen38/scope.json`;
configured capacity and independently checked sequence lengths are distinct.
Serving efficiency uses natural-text prompts with 4,096 input and 512 output
tokens. Its token-counting, synchronization and aggregation methods are defined
by `/app/eval/qwen38/measurement/`.

`/model` contains the checkpoint configuration, tokenizer, chat template and model
card. `/app/eval/qwen38/reference/` contains the pinned model-author source and
the supplied component mathematics. The model-author source is provided for
reading; its newer Transformers version is not installed. The environment's
engine and dependency versions remain fixed.

`/app/eval/qwen38/run.py --help` lists the public examples and measurement commands.
Examples illustrate the evaluation protocols and representative setups. Grading
uses additional inputs and combinations within the stated scope. Example success
is not a whole-task acceptance result. Public measurements report achieved
performance and numerical observations; correctness alone is insufficient to pass.

Other modalities, checkpoints, hardware and parallel layouts, KV precisions,
CPU offload, speculative methods or draft lengths, KV connectors, disaggregated
serving and context extension are outside scope.

## Constraints and submission

The environment is offline apart from the coding harness. Compiled changes must
build through `/app/eval/qwen38/build.sh`. Each server launch has 1,800 seconds
to become healthy. Workload limits, including the accuracy evaluation's total
budget, are defined in `/app/eval/qwen38/budgets.json`. The episode's remaining
time is `/app/.timer/remaining_secs`.

Commit your changes in `/app`. Evaluation rebuilds and serves the committed final
HEAD in a fresh environment. Uncommitted files, development caches and running
servers are not submitted. Grading restores trusted evaluation resources and
applicable engine tests before using them; editing those resources does not
change acceptance.
