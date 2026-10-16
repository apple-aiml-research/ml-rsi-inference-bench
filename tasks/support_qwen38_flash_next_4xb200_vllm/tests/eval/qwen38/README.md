# Qwen38 evaluation resources

This directory holds the evaluation's public contract and the same measurement and validation code
grading uses. Grading restores a trusted copy of this directory before it imports the submitted
engine, and supplies its own inputs and acceptance policies to these modules; editing them changes
nothing about acceptance. Use `/opt/vllm/bin/python` with `PYTHONPATH=/app/eval:/app`.

| Path | Contents |
|---|---|
| `serving/recipes.py` | The two deployments: ordinary TP4, and TP4 with three MTP draft tokens (`python -m qwen38.serving.recipes ordinary\|mtp3` serves one). |
| `scope.json` | The supported component, serving-state and fidelity envelopes, numerical allowances and the PLE memory allowance. |
| `budgets.json` | The wall-clock allowance of every evaluation workload. |
| `measurement/` | Native component measurement, whole-serving measurement, observation validation; `measurement/README.md` is the protocol. |
| `serving/` | API, fidelity, MTP, prefix-reuse, configuration and accuracy protocols. |
| `reference/qsa.py`, `reference/ple.py` | Independent QSA and PLE mathematics over logical histories. |
| `reference/model_author/` | The model author's Transformers implementation, to read (its Transformers version is not installed). |
| `examples/` | Representative inputs for every command; `examples/fidelity/` holds five recorded model-author fidelity examples. |
| `tests/` | Small public tests of these protocols. |
| `build.sh` | The offline rebuild the image was built with and grading runs on the committed tree. |
| `run.py` | Command dispatch: `python /app/eval/qwen38/run.py --help`. |

## What evaluation checks, per deployment

Every obligation holds for the same committed implementation. Numerical limits not stated in
`scope.json` (paired speed floors, fidelity and state bands, acceptance floors) are calibrated on
the reference implementation and are not published.

| Obligation | Ordinary | MTP3 |
|---|---|---|
| Native recipe and served configuration; API, chat, streaming and probabilities | Required | Required |
| Independent model-author fidelity (teacher-forced, 40 private sequences of 512–18,432 tokens) | Required | Required |
| Upstream EvalScope accuracy (`/public/upstream-test`) | The complete evaluation | Not repeated; fidelity and generation equivalence apply |
| Whole-serving decode speed at concurrency 1 and output throughput at concurrency 32, paired with the reference | Both | Both |
| QSA mathematics, state and native-region use | Required | Required |
| QSA component efficiency, paired with the reference | Prefill and decode | Decode (prefill has no timing obligation) |
| PLE mathematics, state, memory, prefill and decode efficiency | Required | Required |
| Fresh-request state and prefix-reuse agreement and witness | Required | Covered by the MTP3 state and production evaluation |
| MTP3 target-generation equivalence and draft usefulness | Supplies the non-speculative series | Required |
| Native production interventions, runner coverage and state preservation (separate launch) | Not evaluated | Required |
| Applicable base-engine tests, run once for the task | Required | Same implementation |

Registered architectures: grading restores the base engine's registry tests with a trusted model
fixture, and its registry coverage test accepts registered architecture names only from that
fixture. `Qwen4ExpForConditionalGeneration`, the checkpoint's declared architecture, is required;
`Qwen4ExpForCausalLM` and `Qwen4ExpMTP` are the only other names the fixture carries (their own
import checks are not scored). Any other registered architecture name fails that base test at
grading. `scope.json` (`registered_architectures`) states the same rule.

## Observations are validated before anything is measured

`measurement/observations.py` validates every served record first, for every protocol here and in
grading: the exact case set, one choice, checkpoint token IDs (`return_tokens_as_token_ids`),
aligned token/probability/top-k arrays, finite log probabilities at most zero, consistent usage,
and termination by the engine's documented fields (`finish_reason`; for `stop`, `stop_reason`:
null for end-of-sequence, otherwise a requested stop). A fixed-length `ignore_eos` request returns
exactly its length with `length` termination. A greedy choice scores the reported maximum within a
tie allowance; a tied token need not appear in a truncated top-k map. Teacher-forced prompt
probabilities have the initial null and one own-token score per later position. Streams need every
chunk's probabilities, one finish record, a final usage record and `[DONE]`. A missing, truncated
or malformed record is a structural failure: it never becomes a disagreement, never earns a
numerical reread and never enters a median.

Continuations are compared only up to their first divergence (later tokens have different
histories), where each side's token must be among the other's five most likely (the near-tie rule).
Distribution discrepancy is k3 = expm1(r) − r of a log ratio r; an exponent overflow is a numerical
infinity, serialized as `{"value": null, "overflow": true}`.

## Commands

Start a server with a recipe, then run one command at a time on the idle server; each writes
`--output result.json` and fails on an invalid observation.

| Command | Observation |
|---|---|
| `configuration --recipe ordinary --launch-log server.log` | The served configuration read from every rank's launch log (use `mtp3` for the other recipe). |
| `api` | Repeat and streamed greedy chat agreement, length termination, eight-way concurrent isolation, completions, streamed completions and prompt probabilities (`examples/prompts.json`). |
| `fidelity` | Teacher-forced observations against the model-author examples (`examples/fidelity/examples.json`): per-sequence mean k3 and length-band medians. |
| `mtp-series --tokens 64 --recipe ordinary` | Fixed-length greedy series of `examples/prompts.json`; repeat with `--recipe mtp3` on that server. |
| `mtp-compare --ordinary a.json --mtp3 b.json` | Series identity (evaluation, source, workload, tokenizer) and first-divergence agreement. |
| `mtp-acceptance --tokens 512` | Draft counters over an isolated interval: finite, monotone, positive activity, accepted ≤ proposed. |
| `prefix-reuse` | One warm/cold pair at the served cache's own boundary (`examples/prefix_reuse.json`). |
| `components --recipe ordinary` | Native QSA/PLE measurement of `examples/components.json` (see `measurement/README.md`). |
| `declaration` | The native regions the server resolves, from the defaults or `/app/qwen38-adapter.json`. |
| `accuracy --inputs /public/upstream-test/configs/Qwen3.8-Flash-Next-FP8.yaml` | The offline EvalScope inventories. |

`components` and `declaration` need the measurement worker extension on the server:
`--worker-extension-cls qwen38.measurement.worker.MeasurementWorker` with `VLLM_SERVER_DEV_MODE=1`
(it adds RPC methods and changes nothing about serving). Grading launches every recipe this way.

Whole-serving measurement uses `measurement/efficiency.py`:

```sh
python /app/eval/qwen38/measurement/efficiency.py measure --engine vllm \
  --base-url http://127.0.0.1:8000 --tokenizer /model/tokenizer.json \
  --texts /public/efficiency/sample.jsonl.xz --concurrency 32
```

Prompts have 4,096 input tokens and greedy completions 512 output tokens with EOS ignored, each
request under its own cache salt; a request counts only after a complete, valid streamed response.
Decode speed is the median over eight requests after two warm-ups of the interval between the first
and last emitted token; throughput uses 128 requests at concurrency 32 after one warm-up wave,
divided by the whole interval. `--axis decode_c1` or `--axis throughput` measures one axis. The
sample's provenance is `/public/efficiency/sample.provenance.json`.

## Scope, clocks and state

The recipes keep the checkpoint's native 262,144-token configuration; independent fidelity covers
512–18,432 tokens. Component workloads, serving-state workloads (continued, fresh and reused
requests; at most 32 live requests, 32,768 input and 1,024 generated tokens each) and their bounds
are in `scope.json`; the cases grading uses stay within them but are not published. Every workload's
allowance is in `budgets.json`, including its bounded repeats; each request inside a workload gets
what remains of that allowance (`measurement/budget.py`).

## Public tests

`PYTHONPATH=/app/eval:/app python -m pytest /app/eval/qwen38/tests` runs the validators, the
mathematics and the scope checks on CPU. With a server at a recipe, set `QWEN38_BASE_URL` (and
`QWEN38_RECIPE=mtp3` for the other recipe) to add the served API, fidelity, prefix-reuse and
component examples.
