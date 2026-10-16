# GPT-OSS working materials

The checkpoint is at `/model`; model-format resources are in `/public/gpt_oss/`
and `/public/harmony/`. The SGLang engine and editable kernel package belong to
the same `/app` repository. Use the serving command in the task instruction.

The task includes implementing native packed-expert computation.
`component-contract.md` specifies the admitted operation domain, the
resident-storage and workspace budgets and their accounting.
`numerical-policy.json` is the accuracy contract for complete expert and
attention outputs; `numerics.py` reads it and measures against it.
`expert_math.py` provides independent serial mathematics for the released
format, not an optimized implementation. Existing engine tests and the model
release resources are also supplied.

## Examples and feedback

Recorded public examples illustrate token log-probability comparison across
several input families:

```sh
PYTHONPATH=/app/eval:/app/python:/app/third_party/triton_kernels \
  python -m gpt_oss.fidelity --base-url http://127.0.0.1:30000 --output /tmp/fidelity.json
```

| Record | Input family | Tokens |
|---|---|---:|
| `fidelity_examples/example_01.json` | Prose | 512 |
| `fidelity_examples/example_02.json` | Math | 768 |
| `fidelity_examples/example_03.json` | Code | 1,024 |
| `fidelity_examples/example_04.json` | Structured text | 1,536 |

The command reports numerical measurements for all four sequences without a
task pass verdict. Their token IDs, independent model-author reference values
and provenance are unchanged from the original records. These examples
illustrate content diversity and the comparison protocol. They do not cover
every serving feature or the full context range required by the task.

Ordinary function-call conversations, including a returned tool result and
streamed responses, are illustrated by `tool_calls/examples.json`:

```sh
PYTHONPATH=/app/eval:/app/python:/app/third_party/triton_kernels \
  python -m gpt_oss.tool_calls.probe --base-url http://127.0.0.1:30000 \
  --examples /app/eval/gpt_oss/tool_calls/examples.json --output /tmp/tool-examples
```

The command reports outcomes for those examples; `tool_calls/protocol.py`
defines the JSON and streaming checks it applies. The released Harmony format
and normal API semantics define the required behavior beyond the sample
conversations; passing them is not a task acceptance verdict.

GPQA diamond supplies the quality protocol at low reasoning effort. The small
public sample uses the same request formatting and answer extraction:

```sh
PYTHONPATH=/app/eval:/app/python:/app/third_party/triton_kernels \
  python -m gpt_oss.acceptance.run_eval --base-url http://127.0.0.1:30000 \
  --output /tmp/gpqa-example
```

`acceptance/run_eval.py` defines the generation settings and scheduling; the full
quality workload has 198 questions. `instruction.md` distinguishes numerical
inputs through 16,384 tokens from configured capacity and serving concurrency.

The packed experts and attention have small component examples. With the four
GPUs available, this loads one released layer through the native
implementation, routes eight tokens with that layer's router and compares the
complete TP4 result, eager and replayed from a CUDA graph, with `expert_math.py`
under the published expert policy:

```sh
PYTHONPATH=/app/eval:/app/python:/app/third_party/triton_kernels \
  torchrun --standalone --nproc-per-node=4 -m gpt_oss.expert_probe --output /tmp/expert-example
```

and this compares the native attention on one history crossing the released
window, for full and sliding-window layers, with `attention.py`'s definition:

```sh
PYTHONPATH=/app/eval:/app/python:/app/third_party/triton_kernels \
  python -m gpt_oss.attention --output /tmp/attention.json
```

Both report their measurements and whether this sample is within the published
policy; neither is a task acceptance verdict.

| Resource | Purpose |
|---|---|
| `expert_math.py`, `component-contract.md`, `numerical-policy.json` | Packed format/mathematics, admitted domain, numerical and resource constraints |
| `numerics.py` | Normalized error and its standing against the published policy |
| `expert_probe.py`, `attention.py` | Checkpoint-backed expert example; attention definition and example |
| `expert_timing.py`, `timing_inputs.py`, `efficiency/client.py` | Component and whole-serving measurements and their workloads |
| `native_experts.py` | Native expert adapter interface and implementation identity |
| `fidelity.py`, `acceptance/run_eval.py`, `tool_calls/probe.py` | Recorded numerical, quality and ordinary conversation feedback |
| `served.py` | Served-configuration, reasoning-effort and decode-versus-prefill protocols |
| `decode_batch/probe.py` | Decode-versus-fresh-prefill agreement across one concurrent batch |
| `final_channel/probe.py`, `basic_tool_calls/probe.py` | Final-channel content and tool-call return checks, with public questions |
| `launch.py`, `recipes.json` | The declared serving recipe |

These include benchmark-supplied adaptations, not only original SGLang tools.
Acceptance requires correct and efficient native packed experts, the specified
model and Harmony serving behavior, and preservation of relevant engine
functionality on held-out inputs within the deployment.

## Measure efficiency

The benchmark supplies measurement code so your performance feedback uses the
same metrics as evaluation. With the task's serving deployment running:

```sh
python /public/measure_serving.py --engine sglang \
  --base-url http://127.0.0.1:30000 --tokenizer /model/tokenizer.json \
  --texts /app/eval/gpt_oss/efficiency/sample.jsonl.xz --concurrency 64
```

This reports per-user decode speed and concurrent output throughput in tokens
per second. Higher is better while the required model behavior and serving
configuration hold. The public sample illustrates the measurement protocol;
evaluation uses separate inputs within the requested deployment.

Packed-expert execution also has a component measurement client. Run this
standalone example with the four GPUs available; it loads a checkpoint layer
through your native implementation and measures one batch size:

```sh
PYTHONPATH=/app/eval:/app/python:/app/third_party/triton_kernels \
  torchrun --standalone --nproc-per-node=4 -m gpt_oss.expert_timing \
  --counts 8 --output /tmp/expert-timing
```

If you use an alternative expert adapter as described below, pass
`--binding your_module:YourClass` to this standalone command. Add `--check` to
compare two rotating inputs of every measured cell with `expert_math.py` under
the published expert policy.

The declared timing workloads cover counts 1, 8, 64 and 192 with graph execution,
and 256, 1,024 and 4,096 with eager execution, each with balanced and skewed routing.
All measured sizes and both routing families contribute to the component
efficiency requirement. These standalone timing sizes do not prescribe the exact
capture sizes of the serving engine. The operation includes native preparation
and the complete expert computation/TP result; compilation and capture are
reported separately from steady-state execution.

It reports execution time in milliseconds and the storage and workspace
observations of `component-contract.md` with their baselines, with setup timing
reported separately. Neither client reports a task pass verdict, and these
examples are not a complete acceptance suite.

## Operation limits

Evaluation bounds each operation, not a schedule of checks. These are resource
limits, not throughput objectives; `operation-limits.json` holds the same values.
All run on the four H100 80GB GPUs of the task.

| Operation | Limit | Clock |
|---|---:|---|
| Server startup, every launch | 900 s | Process spawn through a ready health response, including model load and startup compilation |
| One HTTP correctness request, or one submitted concurrent batch | 900 s | Client dispatch through the complete response(s), including queueing and first-use compilation |
| One component correctness, timing or instrumented invocation | 1,800 s | Dispatch through every rank's record, including setup, warmup, compilation, the prescribed repetitions, synchronization and readback |
| One existing-engine unit-test file | 1,800 s | Collection and every selected case in the file, including compilation |
| A non-inference parser, configuration or identity action | 300 s | Dispatch through its complete record |
| Teardown of a server or worker group | 180 s | TERM, then KILL after 10 s if needed, including reaping |

An operation that exceeds its limit on a healthy host fails for resources.

## Alternative native layouts

The supplied inspection adapter follows one native layout. Different layouts
and fusion are allowed. To select an alternative adapter, commit
`/app/gpt_oss_binding.json` with `{"expert": "your_module:YourClass"}`. An optional
`attention` entry selects the adapter interface in `attention.py`; omit the file
to use the defaults.

The expert adapter accepts the logical `Weights` object, exposes its actual
native module as `module`, and implements
`__call__(activations, selected_ids, selected_weights)`. Its `locate(model)`
returns `(logical_layer, native_module)` pairs for the implementation used by
serving. `native_experts.py` defines the interface. This mapping identifies your
implementation; it does not substitute a separate test implementation for the
engine's real packed-expert operation.

Commit the engine and kernel changes in `/app`. Evaluation restores trusted
materials and engine tests independently of development edits.
