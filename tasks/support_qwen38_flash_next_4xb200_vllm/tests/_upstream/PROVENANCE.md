# Vendored upstream material, with provenance

Everything here comes from `vllm-project/vllm` at the landing commit
`e126687a9a828d513c01a07cd69f025f27d63280` (PR #53896), the commit whose acceptance this task freezes.

## `tests/evals/qwen4_exp/` — the maintainers' accuracy evaluation

Shipped twice, byte-identical: to the agent at `/public/upstream-test/` as the public copy
of the acceptance, and to the verifier through `tests/test.patch`, which applies it onto the
prepared base at `/app/tests/evals/qwen4_exp/`. The graded run and the copy the agent can run
are therefore the same file, digest for digest:

| path | sha256 as shipped | verbatim upstream |
|---|---|---|
| `README.md` | `975979d673f41ec947d58af0054c9fb7f16e5d28d20ce393a78233bdd0733fdb` | yes |
| `__init__.py` | `a25c44dbc06b4899620dcfbe93fe6a7b4a3f80835fcf9cc79db61fc43c1ddadb` | yes |
| `conftest.py` | `f59e47dbee21487e66b5e1fac36b81b2cd790cbbbf64bd040c6b8e50ec987b81` | yes |
| `test_accuracy.py` | `b2dbc77b9113e41af0e4ae2c5feddcf1ad58e15fdd121fd17860382310d2f63b` | one hunk, below |
| `configs/Qwen3.8-Flash-Next-FP8.yaml` | `adfdde85de78804cf6dd7f5ee286e3091125c8298113132f7a7a36edb8158304` | no: upstream's per-dataset thresholds removed (the acceptance policy is separate) |
| `configs/models-b200.txt` | `21b2260cb52982d8f5a88432b0265bb1de8d3c59d56f7256e996e4414044abf8` | yes |
| `configs/models-h200.txt` | `60ea3245b4e19ff731c60f19e0dd2f12d823c3f4f131f7546aaeaf8621de62fa` | yes |

### The one adaptation, and why

`test_accuracy.py` carries a single added helper, `_local_dataset_args`, and one line in the
`run_task` dictionary that calls it. It reads `INFB_EVALSCOPE_DATASET_ROOT`; **unset, the
expression is empty and the module behaves exactly as upstream's**, resolving its datasets from
the hub. Set, it points EvalScope at the offline copies under `/public/datasets/`, which is what
makes the maintainers' own test runnable inside an episode with no network -- the condition this
benchmark puts on every scored check (`allow_internet = false`).

EvalScope 1.10.0 switches to its local hub when a `dataset_id` is an existing path
(`api/benchmark/adapters/default_data_adapter.py::load`), and then loads the directory through
`datasets.load_dataset(path=<dir>, name=<subset or None>, split=<split>)`
(`api/dataset/hub.py::load_dataset_from_hub`). Measured consequence, in a clean
`evalscope==1.10.0` environment: GSM8K's subset is `main`, which arrives as a **config name**, so
the directory has to declare that config -- `<dir>/main_train.jsonl` and `<dir>/main_test.jsonl`
alone give `ValueError: BuilderConfig 'main' not found`. The declaration is the YAML front matter
of `/public/datasets/gsm8k/README.md`. AIME25's subset is `default`, so `name=None` is passed and
HF's own split-file pattern matches `default_test.jsonl` with nothing declared.

## The datasets

| file | rows | source | sha256 |
|---|---|---|---|
| `/public/datasets/gsm8k/main_train.jsonl` | 7,473 | `raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl`, fetched at image build and checked | `17f347dc…` |
| `/public/datasets/gsm8k/main_test.jsonl` | 1,319 | the same repository, `test.jsonl` | `3730d312…` |
| `/public/datasets/aime25/default_test.jsonl` | 30 | ModelScope `evalscope/aime25` at revision `master`, the dataset EvalScope's own adapter names | `c19e99ec…` |

The AIME25 file is the ModelScope rows **as that dataset's own feature schema delivers them**,
which types `answer` as a string. The raw JSONL the repository serves (sha256 `b4e273c0…`) types
it as an integer, and EvalScope's `Sample.target` rejects that with a pydantic validation error;
the rows are otherwise identical, field for field, checked over all 30. Nothing else is changed.

## The model-author reference implementation

`/app/eval/qwen38/reference/model_author/` carries `modeling_qwen4_exp.py` (sha256 `ca9f00bb…`)
and `configuration_qwen4_exp.py` (sha256 `b78132d8…`) from Hugging Face Transformers at tag
**v5.17.0**, fetched at image build and checked against those digests. Transformers is pinned at
**5.12.1** in the image, so those files are to read, not to import. They sit beside the evaluation
resources and are excluded from git, so they are never part of the prepared base or of a
submission.

## DeepSWE

`tests/grader.py` is `datacurve-ai/deep-swe` at `0b9fabbb` with comments naming internal tooling
removed for this release, so the sha256 here is of the file as shipped and not of upstream's:
`5b53d19c24d86b345274c8ec4f223b0df9ab620784a7f497dd156852e495771d`; `tests/test.sh` is its shared
frame, likewise with its comments removed, with only the task-specific middle replaced.

## The Day-0 efficiency kit

`tests/infb_efficiency/` is `infb-day0-efficiency-kit/1`;
`tests/infb_efficiency/OWNERSHIP.md` says what it measures. As configured here: the paired floor
`infb.efficiency.rho` is 0.8, the reference keeps the median of three draws per axis
(`reference_draws` 3), the candidate keeps the median of three throughput draws
(`candidate_throughput_draws` 3) and the best of up to three single-request attempts
(`c1_attempts` 3) with an early exit on the first that meets the floor, and the decode statistic
is the default `ratio_of_medians`.

The single-request rule is biased upward by construction -- one reference draw against up to
three candidate draws -- so `first_attempt_ratios` in the core record carries the unbiased value.
It follows upstream's own instrument for this quantity: `test_bs_1_speed` loops
`bs_1_speed_attempts = 3` and stops at the first attempt that clears.

What is under `tests/` and only under `tests/`: the hidden natural-text pool (360 documents from
31 works) and the draw's seed. What the agent gets: the same measurement at
`/app/eval/qwen38/measurement/efficiency.py`, and at `/public/efficiency/` a public sample of 340
documents from 28 works, in the same four families (prose, math, struct, code), sharing no work
with the pool, with its own provenance file.
