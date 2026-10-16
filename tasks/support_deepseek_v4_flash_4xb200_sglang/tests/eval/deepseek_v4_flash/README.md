# DeepSeek-V4-Flash working materials

The checkpoint and model-author sources are at
`/model/deepseek-ai/DeepSeek-V4-Flash`. SGLang's source and existing development
tools are in `/app`. The task supplies prerequisite JIT kernels; native model
support remains to be implemented, including the scoped GPU kernels whose optimized
implementations are not supplied: sparse attention, C4/C128 compression and the
cache transforms. `kernels/README.md` specifies their interfaces, mathematics,
packed formats, domain, numerical tolerances and component workloads; `kernels/check.py`
checks and measures them on fixed examples and workloads, and `kernels/build.py` is the
offline build of the engine tree and the vendored FlashMLA at `/app/third_party/flash_mla`.

The intended LowLatency launch is defined in
`tests/test_deepseek_v4_flash_fp4_b200.py`. `tests/serve.py --help` describes the
launch wrapper. This is the configuration required by the task.

The supplied efficiency client measures per-user decode speed and concurrent
output throughput in tokens per second: `serving/measure.py` runs
`serving/client.py`, the same candidate measurement code as evaluation. With the
intended server running:

```sh
python /app/eval/deepseek_v4_flash/serving/measure.py --base-url http://127.0.0.1:30000
```

The public workload illustrates the measurement
protocol; evaluation uses separate inputs within the requested deployment.
The output reports performance, not a task pass verdict.

Recorded public examples illustrate retrieving token log-probabilities and
comparing them with the model-author implementation:

```sh
python /app/eval/deepseek_v4_flash/fidelity/check.py --base-url http://127.0.0.1:30000
```

The command compares four independent model-author records:

| Input family | Tokens |
|---|---:|
| Structured text | 512 |
| Math | 2,304 |
| Code | 4,416 |
| Prose | 8,192 |

The prefix-reuse examples exercise another serving setup: a continuation with
and without its prefix cached. The two inputs use prose and code, each with a
6,144-token prefix and a 512-token continuation:

```sh
python /app/eval/deepseek_v4_flash/fidelity/prefix_cache.py --base-url http://127.0.0.1:30000
```

Two independent public input sequences also illustrate the long engine-reference
comparison: one code and one prose record, each with 262,144 input tokens.
Their reference outputs were recorded from the identified engine deployment,
not the model-author runtime:

```sh
python /app/eval/deepseek_v4_flash/fidelity/check.py --base-url http://127.0.0.1:30000 \
  --records /app/eval/deepseek_v4_flash/fidelity/examples/long_reach
```

These examples illustrate numerical comparison and cache reuse across varied
inputs. They preserve their original records;
`fidelity/examples/provenance.json` records their origins and each record's
comparison authority. The commands report measurements without a task pass
verdict. `instruction.md` distinguishes author-fidelity inputs through 65,536 tokens,
the 262,144-token engine-reference comparison and configured context capacity.
A per-record discrepancy and a median over a set of records describe different
observations; these measurements do not supply a universal per-record pass bar.

Model-author recordings are the numerical reference where that implementation
can run at the requested context. Beyond its practical range, evaluation uses
separately recorded engine-reference log-probabilities at the required deployment.
Both use the teacher-forced comparison implemented in `fidelity/fidelity.py`;
the reference provenance differs, not the required model semantics. Prefix reuse
compares the same continuation with a cold and a populated prefix cache.

Every numerical comparison first checks the served answer in full against the
requested span, as `fidelity/fidelity.py` documents: one entry per requested
position, each carrying the requested token id in order, no value at the start
position (where the engine scores no token) and a finite log-probability at every
other position. An answer that departs from this is reported as invalid, with the
position, and receives no number; evaluation treats it the same way. For the prefix
examples the cold and the warm answers are each checked against the continuation's
positions after its first token.

Generated-history replay compares a greedy generation's own output
log-probabilities, token by token, with cold teacher-forced replays of the same
history after a cache flush (`fidelity/generated_replay.py` documents the
alignment and the metrics). A serving engine is not bit-reproducible, so the
history is replayed **three times**, each pass after its own flush, and the
generation is judged against the spread those replays themselves show rather
than against any one of them. Its examples are one 2,048-token prompt and one
2,048-token prefix shared by two continuations, 64 generated tokens each:

```sh
python /app/eval/deepseek_v4_flash/fidelity/generated_replay.py --base-url http://127.0.0.1:30000 \
  --example fresh --output /tmp/replay.json
```

It applies no pass threshold; the comparison is between two engine paths over one
history, not independent model truth.

`tests/served_recipe.py` reports configuration fields and sampled profiling
observations; given the tree's component map (`kernels/check.py --profile-map`) it
also reports whether the served attention runs the tree's own kernels. A report with
no mismatches does not establish graph replay, effective speculation or full task
acceptance.

The measurement client is supplied by the benchmark to make performance feedback
consistent with evaluation. The following entry points describe their own arguments with `--help`:

| Resource | Purpose |
|---|---|
| `tests/check_accuracy.py` | GSM8K completion quality using the supplied offline data and generation protocol |
| `tests/check_native.py` | Native installation and model registration feedback |
| `tests/check_regressions.py` | Selected existing-engine regression feedback |
| `tests/served_recipe.py` | Resolved settings and sampled execution observations, with the limitations above |
| `kernels/check.py` | Kernel examples against their expected values; component workload timings; the component map |
| `kernels/build.py` | The offline build of the engine tree and the vendored FlashMLA |
| `fidelity/generated_replay.py` | Generated-token log-probabilities against three cold replays of the same history |
| `tests/check_speculation.py` | Measured speculative acceptance length; no task pass threshold |

These include benchmark-supplied adaptations, not only original engine tools. Acceptance
requires all behavior specified in `instruction.md`, including model fidelity,
the serving features, efficiency and preservation of relevant engine behavior.
Offline GSM8K data is in `tests/fixtures/gsm8k_test.jsonl`.

## Resource limits

Evaluation runs on the task's four B200 GPUs at the release recipe and bounds each
operation by the wall-clock limits below; `limits.json` carries the same values and
definitions, and the supplied scripts apply them. These are generous resource
limits, not throughput objectives; an operation that exceeds its limit fails.

| Operation | Limit | Clock |
|---|---:|---|
| Server startup | 1,360 s | Process spawn through the first ready health response, including model load and startup compilation |
| One HTTP request, or one explicitly submitted concurrent batch, with every input below 64,000 tokens | 900 s | Client dispatch through the complete response(s), including queueing and first-use compilation |
| The same, with any input of at least 64,000 tokens | 1,800 s | As above |
| One measurement or instrumented runtime invocation, including one component family measurement | 1,800 s | Dispatch through all ranks' records, including setup, warmup, compilation, the prescribed repetitions, synchronization and readback |
| One offline build of the engine tree and the vendored FlashMLA (`kernels/build.py`) | 3,600 s | Build dispatch through usable imports |
| One component correctness check of the kernel interfaces, all of its cases | 1,800 s | Process start through the last case's outputs, including setup, compilation and readback |
| One existing-engine unit-test file | 1,800 s | Collection and all selected cases of that file, including compilation |
| A non-inference action (for example a cache flush or a server-information read) | 300 s | Dispatch through the complete answer |
| Server teardown | 180 s | TERM, then KILL after 10 s if needed, including reaping |

Commit implementation changes in `/app`. Evaluation restores the protected
materials and engine tests before grading.
