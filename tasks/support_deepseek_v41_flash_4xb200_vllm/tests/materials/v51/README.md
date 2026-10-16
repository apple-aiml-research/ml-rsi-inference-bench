# DeepSeek-V4.1-Flash working materials

The model release is at `/model/deepseek-ai/DeepSeek-V4.1-Flash`, including its
configuration, tokenizer, model card and source. The vLLM checkout at `/app`
contains the supplied frontends and prerequisite kernels.

`recipe.py` defines the intended serving deployment: its launch arguments and the
resolved engine configuration they must produce. Its `--help` lists the launch
options; `argv` prints the command and `serve --log FILE` starts it. After
changing compiled sources, use `/app/eval/deepseek_v41_flash/build.sh`; Python
source uses the editable installation.

With the server running, these small examples cover completions and chat,
streamed responses, prompt/generated log-probabilities, repeated greedy requests,
a second conversation turn and concurrent completions:

```sh
python /app/eval/deepseek_v41_flash/tests/api_examples.py --base-url http://127.0.0.1:8000
```

The examples check local response and protocol properties (framing, finish
reasons, usage, requested log-probabilities, stream assembly) and report
solo/concurrent agreement. Nonempty concurrent responses alone do not establish
request isolation. Acceptance also requires model quality, numerical fidelity,
the specified serving features, efficiency and preservation of relevant engine
behavior. Evaluation checks these requirements on held-out short and long requests
within the supplied deployment; this example does not provide a task pass verdict.

For efficiency, `serving/README.md` gives a runnable command for measuring
per-user decode speed and concurrent output throughput against this deployment.
The benchmark supplies the client to align your performance measurements with
evaluation. It reports continuous metrics on public inputs, not a task pass
verdict.

`fidelity/README.md` gives the numerical-comparison command and four recorded
examples covering prose, math, code and structured text at 512 to 20,480 tokens.
They illustrate the protocol across input types and lengths; evaluation uses
held-out inputs within the task's scope.

## The model-author runtime

The checkpoint's released `inference/` implementation is installed as its own
runtime, separate from the vLLM environment:

| Path | What it is |
|---|---|
| `/opt/model-author/inference/`, `/opt/model-author/encoding/` | the checkpoint's `inference/` and `encoding/` code, unmodified |
| `/opt/model-author/venv/` | its own Python environment: torch 2.13.0+cu130, TileLang 0.1.12 with apache-tvm-ffi 0.1.11, transformers 5.17.0, safetensors 0.8.0 (TileLang 0.1.12 rather than the 0.1.8 `inference/requirements.txt` names, whose GEMMs return corrupt values on B200) |
| `/model-author/deepseek-v41-mp4/` | the checkpoint converted by the author's own `convert.py --model-parallel 4 --expert-dtype fp4` into four read-only shards, `model{0..3}-mp4.safetensors`, with `manifest.json` linking them to the checkpoint revision and the converter |
| `/opt/model-author/logprobs.py` | teacher-forced per-token log-probabilities from the author model |

It runs at model parallelism 4 on the same four GPUs as the vLLM server, so run
it only when no server holds them. The ordinary entry point, for generation from
a text file of blank-line-separated prompts:

```sh
cd /opt/model-author/inference
/opt/model-author/venv/bin/torchrun --nproc-per-node 4 generate.py \
  --ckpt-path /model-author/deepseek-v41-mp4 --config config.json --input-file PROMPTS.txt
```

Teacher-forced log-probabilities for your own token sequences (a JSON list of
`{"id", "token_ids"}`, or the fidelity examples' own file):

```sh
/opt/model-author/venv/bin/torchrun --nproc-per-node 4 /opt/model-author/logprobs.py \
  --input /app/eval/deepseek_v41_flash/fidelity/examples.json --output /tmp/author.json
```

It writes `[{"id", "positions", "logprobs"}]`, one log-probability for each token
after the first: the author model runs one prefill from position 0 and its output
head's own `full_logits=True` option returns every position; each value is
`log_softmax` in float32, rounded to 6 decimals. This is how the fidelity
examples' reference values were produced.

## Existing engine tests

`tests/selection.json` lists the existing tests of the engine's starting tree
that this task's changes must keep passing: the tests at the prepared starting
commit, unmodified, for the kernels and model code the new support builds on.
`python /app/eval/deepseek_v41_flash/run.py upstream` runs exactly that list from
`/app`, each file in its own process. They are the engine's own tests at its
starting point, not tests of the new model.

## Operation limits

Evaluation runs on four B200 GPUs and bounds each operation separately:

| Operation | Limit | Clock |
|---|---:|---|
| Server startup at the recipe | 7,000 s | process spawn through the ready health response, including model load and startup compilation |
| Compiled-source rebuild (`build.sh`) | 4,580 s | start through successful completion |
| One HTTP correctness request, or one submitted concurrent batch | 900 s | client dispatch through the complete response(s), including queueing and first-use compilation |
| One timing or instrumented runtime invocation: one axis's speed measurement (all its draws), or one runtime observation | 1,800 s | dispatch through every draw's or every rank's records, including warm-up and synchronization |
| One existing-engine test file | 1,800 s | collection and every selected case in that file |
| One configuration or identity check | 300 s | dispatch through its complete record |
| Server teardown | 180 s | TERM, then KILL after 10 s if needed |

An operation that exhausts its limit on a healthy machine fails.

Additional supplied resources have these purposes:

| Resource | Purpose |
|---|---|
| `tests/quality.py` | GSM8K request/scoring protocol and public accuracy measurements |
| `tests/native.py` | Native model integration feedback |
| `tests/served_configuration.py` | Resolved deployment configuration read from a launch log; not proof of runtime behavior |
| `run.py upstream`, `tests/selection.json` | The existing engine tests above |
| `datasets/gsm8k/` | Offline train/test data used by the public quality command |

For example, `python /app/eval/deepseek_v41_flash/tests/quality.py --base-url
http://127.0.0.1:8000` measures the default public quality sample. The command's
`--help` describes sample selection. The task's numerical scope ends at 20,480
input tokens; configured capacity is 32,768. These resources include task
adaptations, not only files present at the engine's starting commit.

Commit implementation changes in `/app`; trusted tests and evaluation materials
are restored independently during evaluation.
