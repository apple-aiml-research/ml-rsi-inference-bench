# MiniMax-M3-MXFP8 evaluation protocols

Everything here is runnable inside the task container. The protocols are the same code the grader runs; grading
supplies its own inputs and acceptance policy, so a command's output is a measurement of the checks it performed, not
the task's verdict.

## Deployments

`serving/recipes.py` defines the two deployments. Both serve `/model` as `MiniMaxAI/MiniMax-M3-MXFP8` with TP4,
128-token KV blocks, a 262,144-token configured context, the language model only, the `minimax_m3` reasoning and
tool-call parsers with automatic tool choice, the dense layers' attention backend named (`FLASHINFER`), GPU memory
utilization 0.85, and engine defaults otherwise (chunked prefill, prefix caching, compilation and CUDA graphs);
`fp8kv` adds `--kv-cache-dtype fp8`. The memory utilization is below the engine default to leave headroom for prompt
log-probability requests at the configured chunk size.

```sh
python /app/eval/minimax_m3/serving/recipes.py bf16kv --port 8000      # exec's vllm serve
python /app/eval/minimax_m3/serving/recipes.py fp8kv --port 8000
```

The prefix-cache reset used by the replay protocols needs `VLLM_SERVER_DEV_MODE=1` in the server's environment
(`POST /reset_prefix_cache`), and the attribution protocol needs a torch profiler configured on the server
(`--profiler-config '{"profiler": "torch", "torch_profiler_dir": "<dir>", "torch_profiler_with_stack": false}'`,
then `POST /start_profile` and `POST /stop_profile`). The grader starts its servers with both, identically for every
implementation it measures; neither is needed for ordinary serving.

## Commands

`python /app/eval/minimax_m3/run.py --help` lists them. Server commands take `--base-url` of a running deployment.

| Command | What it does | Output |
|---|---|---|
| `configuration --log <launch log> --deployment <d>` | Reads the served configuration from a launch log and `/v1/models` | configuration facts and departures from the deployment |
| `api` | Completions from text and token IDs (top-5 log-probabilities, prompt log-probabilities), streaming completions, chat and streaming chat, eight concurrent requests | structure results |
| `chat [--inputs examples/chat.json]` | Reasoning split checks for the three thinking modes, streaming and non-streaming | per-case problems |
| `tools [--inputs examples/chat.json]` | Tool-call checks, streaming and non-streaming | per-case problems |
| `fidelity --inputs /public/fidelity-examples/examples.json` | Teacher-forced log-probabilities against the recorded model-author values | per-sequence k3 means, band medians |
| `replay [--inputs examples/replay.json]` | Generated-continuation replay | per-sequence mean k3 and max \|d\| |
| `prefix-branch [--inputs examples/prefix.json]` | Prefix-branch replay with prefix-cache reuse evidence | per-request hit tokens, mean k3, max \|d\| |
| `accuracy [--limit N]` | GSM8K chat with thinking | accuracy and per-question records |
| `msa-check --example <id>` | MSA operator correctness on a public example (`msa/README.md`) | per-case structure and policy metrics |
| `msa-measure --family <family>` | MSA operator timing | per-call GPU and wall medians |
| `msa-sparsity` | MSA context scaling | `R_dec`, `R_pre` |
| `binding [--traces <dir>]` | MSA kernel attribution | identity sets; per-rank observations |

## Metrics

- **Fidelity.** `/v1/completions` with the prompt as token IDs, `max_tokens` 1, `prompt_logprobs` 0, temperature 0.
  Structure first (one entry per position, each position's own-token entry present, finite and ≤ 10⁻⁶); then per
  position `k3 = expm1(ref − served) − (ref − served)`, the per-sequence mean, the per-sequence mean with the
  sequence's single largest position removed, the capped mean (the mean of `min(k3, 10)`, so no position contributes
  more than 10 / n) and the flip rate (positions whose log-ratio exceeds 4.6 nats, per 1,000). The public examples
  are five sequences (prose 1,024 tokens, code 4,096, dialogue 4,096, records 16,384, prose 65,536) with model-author
  log-probabilities recorded with Transformers 5.13.0 (`/public/fidelity-examples/README.md`). Grading reads other
  sequences in bands of 1,024, 4,096, 16,384 and 65,536 tokens, and 262,143-token sequences against the engine's own
  reference readings.
- **Generated replay.** Greedy generation with `ignore_eos` and the sampled tokens' log-probabilities; after every
  generation completes, `POST /reset_prefix_cache`, then the prompt plus its own continuation is teacher-forced and
  exactly one aligned, finite entry per generated token with the identical token ID is required. Per token
  `d = replay − generated`, `k3 = expm1(d) − d`; each sequence reports its mean k3 and max `|d|`. The public examples
  are three prose prompts of 2,047, 2,048 and 4,097 tokens generating 64 tokens each.
- **Prefix branches.** For a shared prefix and suffixes A and B: reset the prefix cache, send prefix+A, prefix+B,
  prefix+A, prefix+B sequentially, reading `vllm:prefix_cache_hits` from `/metrics` around each request (the reuse
  evidence), then reset and cold-replay the four continuations with the replay statistic. The public example has a
  2,176-token prefix and two 64-token suffixes.
- **Chat and tools.** Each response's own generated token IDs (`return_token_ids`) are decoded and parsed by the chat
  template's format; the served reasoning, content and tool calls must match the parse (`serving/chat.py`).
- **GSM8K.** All 1,319 test questions through chat completions with `thinking_mode: enabled`, greedy, 4,096 tokens,
  64 in flight; the answer is the last integer (commas removed) of `reasoning + "\n" + content`.
- **Serving efficiency.** `/public/efficiency/client.py` (`/public/efficiency/README.md`): natural-text prompts of
  4,096 input and 512 output tokens, greedy with EOS ignored, per-user decode speed at concurrency 1 and output
  throughput at concurrency 16.
- **MSA operators.** `msa/README.md`: the contract, its numerical policies, the timing protocol and workloads, context
  scaling and what the attribution protocol observes.

## Limits

`limits.json` gives every operation's limit with its clock: server startup 3,600 s (spawn to healthy, including load,
compilation, graph capture and JIT); rebuild 7,200 s; an HTTP workload 900 s, or 1,800 s when any input has at least
64,000 tokens (dispatch to complete response, including queueing); the GSM8K run 3,600 s; one MSA component invocation
1,800 s (including compilation and warm-up); one existing-engine test file or Rust crate 1,800 s; a parser,
configuration or identity action 300 s; teardown 180 s.

## Network

Grading runs every phase with no network. Each phase runs in a fresh network namespace that has only loopback
(`unshare -r -n`, which needs unprivileged user namespaces in the container); where the runtime refuses the
namespace, the phases run only in a container the backend already isolates from the network, and a grading run that
has neither is void. Everything the engine's own tests read at run time is in the image: the Hugging Face files of the
models the `vllm-chat` round-trip tests load, in the Hub cache, and the gpt-oss harmony vocabulary under
`TIKTOKEN_ENCODINGS_BASE`.
