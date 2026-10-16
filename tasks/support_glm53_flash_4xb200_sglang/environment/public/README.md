# GLM-5.3-Flash working materials

The model release is at `/model/zai-org/GLM-5.3-Flash`. `reference/` contains the
pinned Transformers model source. The SGLang checkout at `/app` already contains
the prerequisite kernels described in the task.

`tests/test_glm53_flash_b200.py` defines the HighThroughput and LowLatency
configurations. `python /app/eval/glm53_flash/tests/serve.py --help` describes the
launch wrapper. Both configurations are required; the DFlash2 class in the
upstream file is outside this task's scope.

The supplied efficiency client measures per-user decode speed and concurrent
output throughput in tokens per second. It uses the same candidate measurement
code as evaluation. Against each corresponding running configuration, use:

```sh
python /app/eval/glm53_flash/serving/measure.py --class high_throughput --url http://127.0.0.1:30000
python /app/eval/glm53_flash/serving/measure.py --class low_latency --url http://127.0.0.1:30000
```

Each command measures both dimensions at that deployment's load.
Public inputs illustrate the measurement protocol;
evaluation uses separate inputs within scope. Results are measurements, not a
task pass verdict.

Recorded public examples illustrate teacher-forced numerical comparison.
Run the comparison against each requested serving deployment:

```sh
python /app/eval/glm53_flash/fidelity/check.py --url http://127.0.0.1:30000
```

There are two examples each of prose, code and math, and one longer code
example. Each has a prompt and a 512-token continuation; the six shorter records
total 782 to 1,198 tokens and the longer one 5,120 (a 4,608-token prompt). These
records preserve the original token IDs and model-author log-probabilities.
The command reports divergence measurements without a task pass threshold.
The examples illustrate the comparison protocol across content families;
evaluation also uses unseen inputs and longer contexts within the deployments.

`attention-numerics.md` states how a complete DSA attention sublayer's output is
compared with the model author's computation of the same sublayer: the
representation boundaries, the relative error and its public allowance. It comes
with one 32-token recording of the author implementation's first DSA attention
layer, `fidelity/attention_example.json`, and the metric in
`glm53_protocols/attention.py`.

`tests/api_examples.py` sends one ordinary chat request and one longer text
completion with prompt log-probabilities to the running server. The same API
example applies to either deployment:

```sh
python /app/eval/glm53_flash/tests/api_examples.py --base-url http://127.0.0.1:30000
```

The longer request contains several thousand tokens, with the exact length
determined by the checkpoint tokenizer. This example illustrates the request
and response protocol; it has no recorded numerical reference and does not
establish long-context fidelity or sustained-service correctness.

The measurement client is supplied by the benchmark to align performance
feedback with evaluation. The other supplied entry points have these roles:

| Resource | Purpose |
|---|---|
| `tests/check_accuracy.py` | GSM8K request and scoring protocol for both deployments |
| `tests/check_native.py` | Native installation/registration feedback |
| `tests/check_regressions.py` | Selected existing-engine regression feedback |
| `tests/check_recipe.py` | Configuration readback |
| `tests/check_speculation.py`, `tests/check_adaptive.py` | Speculative-serving observations under LowLatency |

Their `--help` describes arguments. These include benchmark adaptations, not only
original engine tools. `tests/check_adaptive.py` reports the adaptive schedule a
server logged (its tier ladder and switches); a controller may legitimately hold
one tier under a workload, so a switch count is an observation, not a requirement.
Observed configuration/logs and successful public samples
do not establish all runtime requirements. `instruction.md` specifies the
8,192-token input domain, batch/generation/concurrency ranges and persistent-server
requirement for each deployment.
Offline GSM8K data is at `tests/data/gsm8k/test.jsonl`. Acceptance requires the
model fidelity, serving features, efficiency and engine compatibility specified
in the task to hold in both deployments, including on held-out inputs.

## Resource limits

Every check and measurement runs under these wall-clock limits on the declared
hardware (four NVIDIA B200 GPUs), one operation at a time; `glm53_protocols/limits.py`
holds the same values. They are generous resource limits for correct serving, not
speed targets. An operation that exceeds its limit on a healthy host is a resource
failure of what it ran.

| Operation | Limit | From, to |
|---|---:|---|
| Server startup, each deployment | 1,750 s | process spawn through the ready health response, including model load and start-up compilation |
| One HTTP workload (every input in this task's scope is below 64,000 tokens) | 900 s | client dispatch through the complete response of one request, or of every request of one submitted concurrent batch, including queueing and first-use compilation |
| One measurement or instrumented runtime invocation | 1,800 s | dispatch through every rank's record, including set-up, warm-up, compilation, prescribed repetitions, synchronization and read-back |
| One existing-engine unit-test file | 1,800 s | collection and every selected case in that file |
| One non-inference action (configuration read-back, cache flush, parse) | 300 s | dispatch through its complete record |
| Stopping one server or worker group | 180 s | TERM, then KILL after 10 s if needed, and reaping |

Commit the engine implementation in `/app`. Evaluation restores its trusted
materials and engine tests before checking the submission.
