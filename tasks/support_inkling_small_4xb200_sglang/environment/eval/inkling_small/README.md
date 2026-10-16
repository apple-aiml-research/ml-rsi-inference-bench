# Inkling-Small working materials

The checkpoint is at `/model/thinkingmachines/Inkling-Small-NVFP4`. SGLang's source and its existing
development tools are in `/app`. The model author's implementation, for reading only, is in
`/public/reference-impl/` with its provenance. Native model support, including the model's relative
attention on this hardware, remains to be implemented.

`python /app/eval/inkling_small/run.py --help` lists the public protocols below. Every command prints
measurements and, with `--output`, writes them as JSON. None prints the task's verdict: evaluation uses hidden
inputs and private bars, and compares efficiency with its own paired reference measurement.

## The two launches

The registered test `tests/test_inkling_small_nvfp4.py` defines both launches the task is evaluated at:
- **production**, `TestInklingSmallNvfp4`: `--tp 4 --trust-remote-code --quantization modelopt_fp4
  --attention-backend fa4 --page-size 128 --fp4-gemm-backend flashinfer_trtllm --moe-runner-backend
  flashinfer_trtllm_routed --mamba-radix-cache-strategy extra_buffer --swa-full-tokens-ratio 0.1
  --mamba-full-memory-ratio 0.1 --mem-fraction-static 0.85`, with `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`;
- **deterministic**, `TestInklingSmallNvfp4Deterministic`: the same plus `--mamba-track-interval 128
  --enable-deterministic-inference`.

Start a server at either launch by hand with `python -m sglang.launch_server --model-path
/model/thinkingmachines/Inkling-Small-NVFP4 <the flags above> --port 30000` and the environment variable
above; the serving commands below take its `--base-url`. Set `INKLING_SMALL_TEST_MODEL_PATH` to the model
path when running the registered test, as the test itself reads it.

`tests/kl_test_utils.py` is the helper the registered test imports as `sglang.test.kl_test_utils`; it is newer
than the base's copy. `run.py registered` stages both files outside `/app`, imports the helper under that name,
and places the public data where they look for it:
- the deterministic helpers read their LongBench v2 prompts from
  `<helper dir>/.longbench_cache/input_ids__model_thinkingmachines_Inkling-Small-NVFP4_3000_48.json`; the public
  draw `examples/longbench_public.json` goes there, so nothing is fetched (evaluation uses its own draw);
- upstream's GSM8K evaluator reads `/tmp/test.jsonl` before downloading; the public GSM8K test file
  `fixtures/gsm8k_test.jsonl` goes there (evaluation uses its own slice of the same file).

## Building

After changing the engine, rebuild it exactly as evaluation does:

```sh
SGLANG_BUILD_RUST_EXTS=none pip install --no-deps --no-build-isolation --no-index -e /app/python
```

The prebuilt `sglang-kernel` is used as supplied; `/app/sgl-kernel` is not rebuilt. New GPU code compiles at
run time through the engine's own mechanisms (`sglang.jit_kernel`, Triton, CuTe DSL, TileLang, Helion or
`torch.utils.cpp_extension`), inside the startup limit. Evaluation gives each arm private JIT and cache
directories, so nothing compiled here is reused there.

## Commands

| Command | Inputs | Output |
|---|---|---|
| `registered --launch deterministic\|production` | the tree at `/app` | each registered test's outcome, message, printed `avg_kl_div` or GSM8K accuracy, walls |
| `configuration --base-url U [--launch-log F]` | a running server | each recipe item `/server_info` reports, against the registered launch; diagnostics |
| `api --base-url U` | a running server | structure probes: `/generate` and `/v1/completions` from text and token IDs, log-probabilities, streaming, eight concurrent requests |
| `fidelity --base-url U --inputs examples/fidelity.json` | sequences with author series (the public ones: prose 1,024, code 8,192, dialogue 8,192, records 32,768) | per sequence mean k3, max k3, max\|d\|; band medians |
| `replay --base-url U --inputs examples/replay.json` | prompts | per continuation mean k3 and max\|d\|, structural problems, per-token d |
| `prefix-branch --base-url U --inputs examples/prefix.json` | waves of prompts | per request cached tokens; per continuation mean k3 and max\|d\| |
| `conv-state --base-url U --inputs F` | token windows, each with its L0 (none ships; any windows of at least L0 + 2 tokens) | per window the cached tokens at the restore, whether the first token generated after the restore differs from a cold run's, the first divergence, the first-token and prompt-mode log-probability differences, structural problems |
| `conv-state --base-url U --inputs F --history` | the same windows | additionally, per window: the same restored request run twice and the same cold request run twice, under different slot histories, and whether each pair came back identical in token ids and output log-probabilities, with the first differing position and its \|difference\|, the two restores' cached tokens, and structural problems. Run it at the deterministic-inference launch, which is where an identical request has to repeat exactly |
| `efficiency --base-url U [--axis decode_c1\|throughput\|both]` | a running server | per-user decode speed at concurrency 1 and output throughput at concurrency 16, three draws each |
| `rel-check [--example ID]` | `/app/inkling_rel_attention.py` | per public example: structure, inputs unchanged, the numerical policy's metrics |
| `rel-measure [--family F]` | `/app/inkling_rel_attention.py` | per family: GPU and wall seconds per call, three draws, medians |
| `binding` | `/app/inkling_rel_attention.py` | the four kernel identity sets of the entry point |
| `binding --base-url U --identity F` | a running server, the identity sets | per rank, the identity kernels observed in the served model and `cudaGraphLaunch` |

For example:

```sh
python /app/eval/inkling_small/run.py rel-check --example paged-decode-local --output /tmp/rel.json
python /app/eval/inkling_small/run.py rel-measure --family decode_full --output /tmp/t.json
python /app/eval/inkling_small/run.py registered --launch deterministic --output /tmp/det.json
python /app/eval/inkling_small/run.py replay --base-url http://127.0.0.1:30000 --inputs /app/eval/inkling_small/examples/replay.json --output /tmp/r.json
python /app/eval/inkling_small/run.py fidelity --base-url http://127.0.0.1:30000 --inputs /app/eval/inkling_small/examples/fidelity.json --output /tmp/f.json
python /app/eval/inkling_small/run.py efficiency --base-url http://127.0.0.1:30000
```

The relative-attention contract, its numerical policy, the timing families and what attribution observes are
in `rel_attention/README.md`. The efficiency client is `serving/client.py`, byte for byte the client evaluation
runs, and `serving/measure.py` drives it; the prompts are natural text from `serving/sample.jsonl.xz`
(`serving/sample.provenance.json`), 4,096 input and 512 output tokens, greedy, EOS ignored.

## Metrics

- **k3.** For a position with reference log-probability `ref` and served log-probability `served`, with
  `d = ref - served`: `k3 = expm1(d) - d`. A sequence's statistic is the mean over its positions.
- **Fidelity** sends each sequence once to `/generate` with `input_ids`, `max_new_tokens 0`, `return_logprob`
  and `logprob_start_len 0`; the reference is the model author's implementation (`/public/reference-impl/`)
  run over the checkpoint.
- **Replay alignment.** A prompt P generates G (greedy, EOS ignored). After every request of the batch has
  finished, `/flush_cache` (which must succeed) and P + G is teacher-forced; the last |G| input
  log-probabilities are compared with the generated ones: `d = replay - generated`, k3 as above, per sequence
  the mean k3 and max|d|. The token IDs at the aligned positions must be identical.
- **Reuse evidence.** A request that repeats or extends a cached prefix reports the tokens served from the
  prefix cache in `meta_info.cached_tokens`.
- Structural checks come before any number: exactly one value per requested position, identical token IDs,
  every value finite and at most 1e-6. An answer that fails them is reported invalid and receives no number.

## Resource limits

Evaluation runs on the task's four B200 GPUs and bounds each operation by the wall-clock limits in
`limits.json`, which the supplied scripts apply. These are generous resource limits, not throughput
objectives; an operation that exceeds its limit fails.

| Operation | Limit | Clock |
|---|---:|---|
| Server startup, each launch | 3,600 s | Process spawn through the first ready health response, including model load, startup compilation and graph capture |
| Documented install (`pip install -e`, offline) | 1,800 s | Start to completion |
| One HTTP request, or one explicitly submitted concurrent batch, every input below 64,000 tokens | 900 s | Client dispatch through the complete response(s), including queueing and first-use compilation |
| The same, with any input of at least 64,000 tokens | 1,800 s | As above |
| One registered GSM8K run | 3,600 s | First request through the last response |
| One deterministic helper | 1,800 s | Helper start through its assertion |
| One component correctness or timing invocation | 1,800 s | Dispatch through the record, including JIT compilation, warm-up and allocation |
| One existing-engine test file | 1,800 s | Collection and all selected cases of that file, including compilation |
| A configuration, registry, identity or profile action | 300 s | Dispatch through the complete answer |
| Server teardown | 180 s | TERM, then KILL after 10 s if needed, including reaping |

Profiling needs a directory the server can write and this process can read: attribution
(`binding --base-url`) sends `/start_profile` with an output directory; nothing else needs one.

Commit implementation changes in `/app`. Evaluation restores these materials and the engine's tests before
grading.
