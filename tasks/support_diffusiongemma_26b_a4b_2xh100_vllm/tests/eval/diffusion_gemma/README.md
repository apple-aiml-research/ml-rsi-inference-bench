# DiffusionGemma: public evaluation resources

Everything here is public and identical to the grading copy; grading supplies its own inputs and
acceptance policy. Outputs of these commands are measurements, not the hidden verdict.

## Recipes and launches

`serving/recipes.py` holds the two deployments and their fixed-work configurations:

| Recipe | GPUs | Generation configuration |
|---|---|---|
| `tp1` | 0 | the checkpoint's `generation_config.json` (the published recipe) |
| `tp1_fixed` | 0 | `confidence_threshold` 0.0, `t_max` 5.0, `t_min` 0.15: no canvas stops early |
| `tp2_fixed` | 0, 1 | tensor parallel size 2, `confidence_threshold` 0.0 |
| `ar_smoke` | 0 | Qwen2.5-0.5B-Instruct at engine defaults |

    python /app/eval/diffusion_gemma/run.py gencfg --out /tmp/gencfg       # the fixed-work configurations
    python /app/eval/diffusion_gemma/serving/recipes.py serve tp1 --port 8000
    python /app/eval/diffusion_gemma/serving/recipes.py serve tp1_fixed --gencfg-root /tmp/gencfg

`serving/server.py` launches a recipe with private JIT and scratch roots, `VLLM_SERVER_DEV_MODE=1`
(which exposes `/reset_prefix_cache`) and an idle torch profiler (`/start_profile`,
`/stop_profile`); the dev-mode and profiler flags are needed only by the reset and attribution
protocols.

## Commands

`python /app/eval/diffusion_gemma/run.py <command> --help` lists each command's options. Serving
commands take `--base-url` (default `http://127.0.0.1:8000`) and read the public inputs in
`examples/` unless `--inputs` names others; every command writes `--output`.

| Command | What it measures |
|---|---|
| `configuration --log LOG --deployment tp1\|tp2` | the resolved serving configuration: served name, context 262,144, tensor parallel size, four sequences, prefix caching and chunked prefill on, CUDA graphs on, asynchronous scheduling at the engine default, no remote code, the multimodal limits; at TP1, block-diffusion scheduling (one 512-token generation alone schedules at least 16 tokens per engine step); the resolved CUDA-graph mode, batch budget and block size are diagnostics |
| `api` | structure and log-probability coherence of completions (text and token ids), streaming and chat, one image chat, eight concurrent requests; one structured-output request; one completion beyond one canvas (1,024 tokens); 36 concurrent requests with log-probabilities; short-answer first tokens |
| `chat` | reasoning and content against the split of each response's own generated text, greedy (temperature 0), streaming and non-streaming; a generation that opens with the thought-open marker is split as a thought block, whatever the request's thinking setting, and any other generation is all content |
| `tools` | tool calls against an independent parse of each response's own generated text, greedy (temperature 0), streaming and non-streaming |
| `reads`, `step1` | the denoising reads of `serving/reads.md` against the model-author values in `examples/reads.json` |
| `replay` | commit-encode replay: a read on a prompt plus its own seeded generation (temperature 1.0 with a fixed per-prompt seed), warm (the KV the commit passes wrote) and cold (after `/reset_prefix_cache`, the same tokens prefilled in the same 256-token canvases); the KV a commit pass writes must be what an encoding pass over the same tokens writes, and a read leaves nothing in the prefix cache past its prompt (its emitted canvas has no commit pass, so a prefix cache that hashes it under the emitted tokens serves the denoise pass's KV to the cold reads) |
| `prefix-branch` | reads on one prefix with two suffixes, warm and cold |
| `mixed` | reads under concurrent generation load and alone |
| `preemption` | concurrent long requests whose KV outgrows the pool; completion and the preemption count; each request within `budgets.json`'s `preemption_workload` limit, max(1,800 s, 4 × the reference's largest request wall at this load) |
| `semantic` | short factual and visual questions with thinking enabled, 16 draws each |
| `policy` | the schedule instrument (four draws per case) and one-canvas walls for the natural/fixed ratio |
| `long-context` | a 262,044-token prompt with the 100 output tokens the context leaves |
| `accuracy` | GSM8K |
| `attend-check`, `attend-measure` | the operator contract's correctness and timing (`attend/README.md`) |
| `binding` | the served kernels in two profiler windows against the contract's identity sets |

## Metrics

- **k3.** Per position, `k3 = expm1(ref − served) − (ref − served)` on the target token's
  log-probability. A read's statistic is the mean over the positions up to and including the
  reference canvas's first end-of-sequence token (the author's generation pads every later
  position); the mean over all positions is reported beside it. Per band, the median of the
  reads' statistics.
- **Pair statistic.** The same mean between two readings of one read (warm/cold, loaded/alone).
- **Reuse evidence.** The prefix-cache hit counter (`vllm:prefix_cache_hits`) across a request.
- **Policy ratio.** `R = median wall(natural) / median wall(fixed)` over one-canvas requests.
- **Long context.** The near-limit request uses `max_tokens` 100, what the configured context
  leaves after 262,044 prompt tokens; a request for more is refused by the engine's context check.
- **GSM8K.** The upstream harness's 5-shot prompt and `get_answer_value` extraction on the content.

## Limits

`budgets.json` publishes every operation's cap and clock (a launch: spawn to healthy; a rebuild:
`build.sh` start to completion; a workload: dispatch to complete response; and so on).
`build.sh` rebuilds incrementally from `/app/build/temp`.
