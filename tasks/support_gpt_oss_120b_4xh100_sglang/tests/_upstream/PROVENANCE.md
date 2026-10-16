# Provenance of the vendored upstream material

Everything in this directory and everything `tests/test.patch` applies comes from upstream SGLang or
from the published GPQA-diamond question set. Nothing here is ours except where this file says so.

## The acceptance test and its evaluator

`tests/test.patch` is a diff onto the base commit `556e4143f0ba8e480a73fc0e08f05d7ea83880cc`
(4 files, +163/−124) that installs this model's upstream CI check. The check was registered by
commit **`442534aa44b1269b4fcbd014e3912147d5fde195`**, three days after the reference commit
`7b5bcb6e84f949362ed1b2b1e5f5a6f722f649e1`.

| Path in the tree | Role |
|---|---|
| `test/srt/test_gpt_oss_4gpu.py` | new file, 50 lines: the registered test. The recipe flags and the floor live here; `test_mxfp4_120b` is the scored method and `test_bf16_120b` is the out-of-scope BF16 repack |
| `test/srt/test_gpt_oss_common.py` | new file, 99 lines: `BaseTestGptOss` — the launch, the `ThreadPoolExecutor` over reasoning efforts, and `_run_one_eval`'s arguments (`num_examples=198`, `num_threads=198`, `max_tokens=4096`, `temperature=0.1`) |
| `python/sglang/test/run_eval.py` | replaced by a thin shim that calls the protected evaluator, below |
| `python/sglang/test/test_utils.py` | one constant: `DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH` 600 → 900, so the registered launch is bounded by the published 900 s startup limit rather than a shorter one |

**Why `run_eval.py` is replaced rather than edited.** Upstream's GPQA branch names its question set by
URL and `GPQAEval` reads it with `pandas.read_csv`, which cannot work in an offline container. The
shim keeps the same entry point and the same `args` contract and forwards to the evaluator the task
already ships to the agent — `gpt_oss.acceptance.run_eval.evaluate`, at `/app/eval/gpt_oss/acceptance/`
in the agent's image (on its `PYTHONPATH`) and `/tests/public_eval/gpt_oss/acceptance/` in the
verifier — reading the local `tests/_upstream/gpqa_diamond.csv`. It accepts the `gpqa` eval name only;
no other upstream eval is reachable through it, and none is used by this task. The agent's copy ships
beside a 4-question `examples.csv`, so the same scoring and request path can be exercised without the
held-out question set. The evaluator itself — `GPQAEval`, its per-question four-choice permutation,
its answer regex, `ChatCompletionSampler`, the multiple-choice template and the aggregation — is
upstream's own `simple_eval_gpqa.py` / `simple_eval_common.py` code, vendored under its own names and
visible to the agent.

| Artifact | Identity |
|---|---|
| `tests/_upstream/gpqa_diamond.csv` | upstream's own published file, `https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv`, 1,373,492 bytes, sha256 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`, 198 rows, unmodified |

**What the L0 node already covers.** A separate assertion that
`get_quantization_config("mxfp4").get_name() == "mxfp4"` would add nothing: `tests/infb_verify/test_l0.py`
constructs `ModelConfig(model_path=...)`, and at the base that call is exactly what raises
`ValueError: Unknown quantization method: mxfp4`, so the registration is a precondition of the node
passing at all.

## The hidden permutation seed

The verifier passes `seed=517093` to `GPQAEval`; upstream's default, which the agent-visible copy at
`/app/eval/gpt_oss/acceptance/` keeps, is `0`.

What the seed actually fixes, read from the code rather than assumed: the registered test calls
`run_eval` with `num_examples=198` and GPQA-diamond holds exactly 198 questions, so
`rng.sample(examples, 198)` is a **full permutation of the question order**, and the following
`rng.sample(range(4), 4)` per example fixes **each question's four-choice permutation**. All 198
questions are still scored, so nothing is withheld from the benchmark's meaning — the seed varies
inputs inside the stated obligation and adds none.

Its secrecy is **not load-bearing and is not claimed to be**. `gpqa_diamond.csv` is upstream's own
published file, answers included, and it is one HTTP fetch away for anyone; only the verifier image
carries it (`COPY _upstream/ /tests/_upstream/`), and the agent's image gets the 4-question
`examples.csv` instead. So a candidate that hard-codes question-to-answer lookups is possible in
principle, and the seed only ensures that a lookup keyed on the default permutation scores wrong. The
gate against lookup is the fidelity statistic against the model-author reference implementation,
which no lookup can satisfy.

## The adapted GPQA floor

`test/srt/test_gpt_oss_4gpu.py`'s `expected_score_of_reasoning_effort` carries
`{"low": 106 / 198}` where upstream asserts `{"low": 0.61}`.

This is one application of a rule fixed before the measurement, not a bar moved to fit a run: keep
upstream's floor if the minimum of five fresh reference-oracle runs clears it by three standard
errors, otherwise publish `floor(mean − 3 sd)`. The five runs, on the hidden seed at low reasoning
effort, scored **127, 133, 117, 133 and 124** of 198 (mean 0.640404, run-to-run sample sd 0.033955,
binomial sd 0.034104). The minimum, 117/198, does **not** clear upstream's 0.61 — which is 120.78 of
198 — so the second branch applies: `mean − 3 sd` = 0.538539, floored to a whole question, is
**106/198 = 0.535354**.

That a correct implementation scores near upstream's own bar is a property of the protocol rather
than of this recipe: upstream's 39 scheduled H100 runs of it read mean 0.618 with a minimum of 0.551,
below 0.61 outright (sgl-project/sglang#22237).

The floor is written into the **test**, not only into `config.json`, because the scored node *is* the
test and the test asserts its own bar; a floor recorded only in the config would be decorative. The
five scores and the arithmetic ship in `config.json` under `infb.bars.gpqa_low_floor_provenance`.

## The model-author reference implementation under `/public`

`gpt_oss/torch/{model,weights,utils}.py` and `gpt_oss/tokenizer.py` come from
`github.com/openai/gpt-oss` at tag **`v0.0.1`**, commit
`4931694686fadfa74a80554473d32f7dd4d059f3`. They are supplied as **source to read and port**, not as
a runtime: nothing puts `/public` on an interpreter path, and the implementation has no KV cache and
no server. `harmony/openai-harmony-README.md` is the README of `github.com/openai/harmony`.

## What transformers ships and this image does not

`transformers` 4.55.0 — the base tree's own pin — includes `models/gpt_oss/modeling_gpt_oss.py`, a
complete importable implementation of this model. The recipe removes it and
`modular_gpt_oss.py`, keeps `configuration_gpt_oss.py` (which `AutoConfig.from_pretrained` needs,
since the landing registers no `gpt_oss` entry in SGLang's own config registry), and asserts at build
time that no gpt-oss modeling module is reachable on any interpreter path while `GptOssConfig` still
resolves through `CONFIG_MAPPING`. `environment/scrub_reference_impl.py` is that step, with the full
reasoning; it runs from a temporary path and is deleted in the same layer.
