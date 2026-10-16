# Kimi Linear working materials

The checkpoint, configuration, tokenizer, model card and chat template are at
`/model`. `reference/` links to the supplied model-author Python source for
inspection. `router-semantics.md` documents the authoritative routing definition
where that source differs.

Start a deployment with:

```sh
python /app/eval/kimi_linear/launch.py --port 8000                          # tp4, the recipe
python /app/eval/kimi_linear/launch.py --deployment dp2 --port 8000         # two TP4 replicas
python /app/eval/kimi_linear/launch.py --deployment apc --port 8000         # prefix caching
python /app/eval/kimi_linear/launch.py --deployment apc-pressure --port 8000
```

The environment has eight H100 80GB GPUs. `tp4` (prefix caching off), `apc` (prefix caching with 512-token state
checkpoints, `--mamba-block-size 512`) and `apc-pressure` (`apc` at a 16,384-token context with a 64-block cache, at
most 16 running requests and 8,192 batched tokens per step: the capacity-limited variant for cache turnover and
preemption) serve on the first four GPUs; `dp2` (two TP4 data-parallel replicas, prefix caching off) on all eight.
Data parallelism and prefix caching are not combined. `launch.py --deployment <name> --print` shows the command.

Use `/app/eval/kimi_linear/build.sh` after changing compiled sources or build
configuration. Python and Triton edits use the editable installation.

## Operation limits

Evaluation bounds single operations on the supplied hardware (eight H100 80GB
GPUs; the TP4 deployments of `launch.py` on four of them, `dp2` on all eight) by these wall-clock limits, which are also
in `operation-limits.json` with their clock definitions. They are resource
limits, not throughput objectives or per-token service targets; an operation
that exhausts its limit on a healthy host fails for lack of resources.

| Operation | Limit | Clock |
|---|---:|---|
| Server startup | 600 s | process spawn through a ready health response, including model load and startup compilation |
| Compiled-source rebuild, when required | 7,900 s | start of `build.sh` through its successful completion |
| One completions request, or one concurrently submitted batch, whose largest input is below 64,000 tokens | 900 s | client dispatch through the complete response(s), including queueing and first-use compilation |
| The same with any input of at least 64,000 tokens | 1,800 s | as above |
| One loaded-layer inspection, sample, timing or other instrumented runtime invocation | 1,800 s | dispatch through every rank's record, including setup, warmup, compilation, repetitions, synchronization and readback |
| One existing-engine unit-test file | 1,800 s | collection and all selected cases in that file, including compilation |
| One parser, configuration or identity action | 300 s | dispatch through its complete record |
| Teardown of a server or worker group | 180 s | TERM, then KILL after 10 s if needed, including reaping |

Kernel authoring is an explicit adaptation of this task. `kda/` supplies a serial
mathematical reference, not optimized KDA. Its README gives small fresh/continued/packed-state
operator examples, a loaded-layer sample of the same kinds on your server, the numerical
accuracy policy and the mapping interface for alternative native layer layouts.

For model-level numerical comparison, `fidelity/examples.json` contains ten
recorded model-author outputs: eight short inputs of 256 to 2,048 tokens and
two longer inputs of 9,250 and 11,550 tokens. Compare them with the server using:

```sh
python /app/eval/kimi_linear/fidelity/check.py --mode author --url http://127.0.0.1:8000
```

These examples preserve their original tokens, reference values and provenance.
They illustrate the same numerical-comparison protocol on short and longer
contexts. Evaluation uses separate inputs within the requested deployment;
neither these readings nor the operator example establish complete native support.

The longer-context continuation comparisons use identified engine-reference
recordings at the intended deployment. Model-author recordings define the
shorter numerical comparisons; fresh-prefill versus decode comparisons check
consistency within the candidate. The fidelity records identify their reference
and comparison method. These are distinct sources of evidence for the same
required model behavior.

Request-state lifecycle -- fresh and continued requests, data parallelism, prefix reuse, branching, generated
history, preemption and cancellation -- is specified in `state_lifecycle/README.md` with its continuation rule in
`state_lifecycle/policy.json`. `state_lifecycle/client.py sample` runs a deployment's representative example and
`compare` teacher-forces the recorded continuations on a cold `tp4` server; against your own server that comparison
is self-consistency, not agreement with the independent reference evaluation also uses.

For efficiency, `serving/README.md` gives a runnable command for measuring
per-user decode speed and concurrent output throughput. `kda/README.md` also
gives a timing example for the native layers loaded by your server. The
benchmark supplies these instruments to align performance feedback with
evaluation. They report
measurements on public workloads, not task acceptance.

Additional supplied entry points:

| Resource | Purpose |
|---|---|
| `tests/acceptance.py --mode accuracy` | GSM8K generation and scoring protocol |
| `tests/acceptance.py --mode api` / `--mode concurrency` | Ordinary API and request-independence feedback |
| `tests/acceptance.py --mode recipe` | Resolved serving configuration observations |
| `tests/native.py`, `tests/regressions.py` | Native integration and selected existing-engine feedback |
| `kda/client.py --mode inspect` | Discover the loaded native-layer mapping; see `kda/README.md` |
| `kda/client.py --mode sample` | Invoke the first loaded native layer on the sample cases under the numerical policy |
| `kda/client.py --mode timing`, `kda/assessment.py` | Complete-layer timing families and the interpretation of saved records; see `kda/README.md` |
| `kda/operator_check.py` | Standalone operator examples against your own adapter; see `kda/README.md` |
| `state_lifecycle/client.py sample` / `compare` | A deployment's representative state-lifecycle example and its continuation comparison; see `state_lifecycle/README.md` |

These resources include benchmark-written adaptations, not only original vLLM
tools. For example:

```sh
python /app/eval/kimi_linear/tests/acceptance.py --mode accuracy --url http://127.0.0.1:8000
```

The instruction distinguishes numerical inputs through 12,000 tokens from the
64,000+512 long-context protocol and configured capacity. Every declared KDA
performance family is required; measuring family 0 alone does not cover them. Offline GSM8K data is linked under
`datasets/gsm8k/`. Acceptance requires correct and efficient native KDA,
the specified model and serving behavior, and preservation of relevant existing
engine functionality, including on held-out inputs within the deployment.

Commit the implementation in `/app`. Evaluation restores its protected materials
and upstream tests and uses broader inputs than the examples.
