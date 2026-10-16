# Add GLM-5.3-Flash support to SGLang

Land native support for `zai-org/GLM-5.3-Flash` in the SGLang checkout at `/app`
on four NVIDIA B200 GPUs. Load the released checkpoint through SGLang's native
configuration, model and weight-loading interfaces, preserving its behavior
and relevant existing engine functionality.

Support both intended deployments, HighThroughput and LowLatency, defined in
`/app/eval/glm53_flash/tests/test_glm53_flash_b200.py`. Preserve their parallel
layout, attention and MoE backends and FP8 KV cache. LowLatency uses the
checkpoint's NextN draft and adaptive speculation.

Provide reliable text generation through `/generate`, `/v1/completions` and
`/v1/chat/completions`, with normal batching, concurrent use and token
log-probabilities. Preserve model quality and numerical fidelity to the supplied
Transformers model implementation.

Efficient production serving is part of acceptance in both deployments. Each
deployment's required execution paths must be active during ordinary serving.
Per-user decode speed and concurrent output throughput are evaluated using the
measurement protocol in `/app/eval/glm53_flash/serving/measure.py`; the working
materials README gives commands for each deployment. Inadequate performance on
either dimension in either deployment can fail acceptance even when outputs
are correct. Achieved performance is recorded with the result.

Acceptance requires the specified model behavior, serving features, efficiency
and relevant existing engine functionality to hold together in both deployments.
Evaluation covers these requirements on held-out inputs and sustained concurrent
use within the stated configurations. Public examples illustrate API use and
measurement protocols; passing them does not establish task acceptance.

## Workload scope

Support text inputs through 8,192 tokens, including supplied continuations in
input-logprob requests; batched logprob requests of up to 32 sequences; generation
of up to 2,048 new tokens; and quality workloads with up to 128 concurrent requests.
Quality uses the maintainers' GSM8K protocol in
`/app/eval/glm53_flash/tests/check_accuracy.py`. These are distinct workloads, not
every combination of their maxima or a requirement to numerically validate the
entire configured context capacity.

Serving efficiency uses natural-text prompts of 4,096 tokens and greedy
generation of 512 tokens. Both deployments measure single-request decode;
throughput uses concurrency 64 in HighThroughput and 16 in LowLatency.
Serving requirements must hold across successive workloads on one server per
deployment.

## Working context

`/model/zai-org/GLM-5.3-Flash` contains checkpoint revision `690b7052`, its
configuration, tokenizer, chat template and model card. The engine starts from
SGLang `a9944aec` and already supplies the KDA, k-pool DSA and mHC layernorm
kernels. Kernel authoring is not required; their wrappers may be adapted.

`/app/eval/glm53_flash/reference/` contains the pinned Transformers model source.
`/app/eval/glm53_flash/README.md` introduces the deployment entry points and a
set of representative numerical examples in `/app/eval/glm53_flash/fidelity/check.py`.
`/app/eval/glm53_flash/tests/api_examples.py` illustrates chat and a longer
completion request at either deployment.

Image/video inputs, external DFlash2 drafts, prefill/decode disaggregation,
context or pipeline parallelism, HiCache, other hardware and other quantizations
are outside scope.

## Submission

Work offline using the supplied dependencies and commit your implementation in
`/app`. Only committed implementation changes are graded. Evaluation materials
and engine tests are restored independently; edits to them, caches and running
processes do not change acceptance. Each serving configuration has 1,750 seconds
to become healthy; the working materials README lists the operation limits every
check and measurement runs under. `/app/eval/glm53_flash/attention-numerics.md`
states the numerical contract for the DSA attention sublayers.

`/app/.timer/remaining_secs` reports the number of seconds remaining for this run.
