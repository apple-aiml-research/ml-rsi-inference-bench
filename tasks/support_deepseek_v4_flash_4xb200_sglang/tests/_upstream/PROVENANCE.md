# Provenance of the verifier's upstream material

Every file the verifier runs that is not this task's own, with where it came from and its digest.
Nothing listed here is edited: upstream's acceptance test is run as upstream wrote it.

## What is vendored, and how

This task vendors **no upstream file into `tests/_upstream/`**. The registered acceptance test and
the test-kit modules it imports live in the engine tree itself, and the verifier takes them from two
places the candidate cannot reach:

1. `tests/test.patch` carries the three test-tree paths the landing changes, as a diff onto the
   task's base commit `eac98398e7adc70dd10d8db3ff77226b026000ce`.
   `tests/infb_verify/restore_paths.py` runs before anything imports from those trees: each path of
   `config.json` `infb.restore.trees` — `eval/deepseek_v4_flash` first, then `python/sglang/test`
   and `test` — is removed as a path (a symlink on the way is unlinked, never followed), written
   from the blobs of the tree the config pins in the base commit of the image's own object store
   (replace objects, grafts and checkout filters off), and hashed back against it. A difference is a
   verifier fault (exit 4, the `-1` sentinel). `tests/test.patch` is re-applied afterwards. So the
   module that runs is the reference's, whatever the candidate did to its own copy.
2. The kit modules the test imports that the landing does **not** change are byte-identical at the
   base commit and at the reference, so no diff can carry them; the reset in step 1 is what
   protects them.

| Path | Role | Blob at `35870d55` | sha256 of the reference content |
|---|---|---|---|
| `python/sglang/test/kits/server_sanity_kit.py` | the twelve `ServerSanityMixin` probes; **added** by the landing | `af9cf0167dd3f5595a581f94d48b981626793137` | `3c7f9c77b7449250694cb11bf563388d5d3d3a552b34edc0f6604108367e8585` |
| `python/sglang/test/test_utils.py` | `popen_launch_server`, `try_cached_model`, `DEFAULT_URL_FOR_TEST`, `CustomTestCase`; **modified** by the landing (+6 lines, an env merge inside `popen_launch_pd_server`, a path this recipe never takes) | `6ef153bd81a0e6ed811db4c80ad7311e0b8838a0` | `6ada89a2f60ff3b4e7229c5005044408bb94156447492dcfd0c0bb3d79514255` |
| `test/registered/4-gpu-models/test_deepseek_v4_flash_fp4_b200.py` | the registered per-commit acceptance test, suite `stage-c-test-dsv4-4-gpu-b200`; **added** by the landing | `b750d04fda1bbc68f9958cc485af14f84edde0f9` | `432b608a28fc68ea7f74bb5ac229a991b5241949301e6dd7a36f71b86d083d6d` |

Byte-identical at the base commit and at the reference, hence protected by the reset rather than by
the patch:

| Path | Blob (both commits) |
|---|---|
| `python/sglang/test/run_eval.py` | `d872966e7497f647ba25a4631d183d1504109091` |
| `python/sglang/test/simple_eval_gsm8k.py` | `88665d9c732acd4d24f860a6578dfe7d83bbb747` |
| `python/sglang/test/simple_eval_common.py` | `b9e4057fa74c18337f20de457c0c887a4b355026` |
| `python/sglang/test/ci/ci_register.py` | `014c408dd5e79b05947bf91e07943169e74085fa` |

`tests/test.patch`: 13,319 bytes, sha256
`bb67d9e92dbf4f9e5c8977615f4a5f65b61be60c066b14bf65a00eb575758490`, 3 files, +316 lines. It is a
`git --no-pager diff --no-ext-diff --no-color --binary --full-index` of the three paths above between
the task's base commit and the landing commit `35870d55aca7c6912aa4c785f583da4acac04938`, checked
with `git apply --check` both on a clean base worktree and on the reference tree after the kit reset.

## The evaluation data

The GSM8K test set is **not** copied into this directory. The task image fetches it from its public
source at image build, checks its sha256, and keeps it at
`/app/eval/deepseek_v4_flash/tests/fixtures/gsm8k_test.jsonl`, committed with the evaluation
materials (`git add -f`: the engine's `.gitignore` matches `*.jsonl`) and restored with them before
grading. `tests/test.sh` copies that file to `/tmp/test.jsonl`, which is the path
`sglang.utils.download_and_cache_file` returns without a network. One copy in the image instead of
two in the repository.

| File | Source | sha256 |
|---|---|---|
| `test.jsonl` (1,319 rows) | `https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl` | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

## Shared with the other model-support tasks

| File | Origin | sha256 |
|---|---|---|
| `tests/grader.py` | DeepSWE `datacurve-ai/deep-swe` at `0b9fabbb`, `tests/grader.py` (one variant across its 113 tasks), with comments naming internal tooling removed for this release, so the digest is of the file as shipped here and not of upstream's | `5b53d19c24d86b345274c8ec4f223b0df9ab620784a7f497dd156852e495771d` |
| `tests/test.sh` frame | the same DeepSWE shared frame; the block between `>>> RUN TESTS` and `>>> END RUN TESTS` is this task's, the rest is shared | — |
| `tests/preflight.py` | the model-support preflight, shared across these tasks and extended here by two measurements that are written into `preflight.json` and never asserted: the `/tests` digest, and a host record for a slow reference load (`nvidia-smi topo -m`, `numactl --hardware`, the checkpoint's mount, and a timed `O_DIRECT` read of up to 1 GiB of the shard that holds the MTP draft alone) | — |
| `tests/infb_verify/test_l0.py` | the shared L0 module, verbatim; every task fact comes from `config.json` | — |
| `environment/fetch_model.py` | the shared checkpoint fetcher, verbatim | — |
| `environment/scripts/start_agent_timer.sh` | the benchmark's shared agent timer, with the internal orchestrator's environment variables removed for this release | — |
| `tests/infb_efficiency/` | this task's own efficiency verifier: the shared Day-0 efficiency kit (`infb-day0-efficiency-kit/1`), only the parts this task uses, owned and tested by the task. `PROVENANCE.json` lists each file, its own digest and what was done to it; `canary.py` and the kit manifest are not carried. The agent's copy of the client is `environment/eval/deepseek_v4_flash/serving/client.py`, byte for byte | see `PROVENANCE.json` |
| `tests/reference/solution.patch` | `solution/solution.patch`, byte for byte: the reference the verifier serves (a guard holds the two equal) | `0fd81aba8fb191abf0bfe9fd495c88a0047b88b99f91a7f121a52ade8f7ed995` |
| `tests/hidden/efficiency_seed.json` | the hidden draw's seed, `secrets.token_hex(16)`, generated once; the efficiency record keeps only its sha256 | — |
| `tests/hidden/recipe_witness.json` | the recipe witness's private workload (`tests/infb_verify/recipe_witness.py`): a warm-up, two concurrency-1 requests and a batch of sixteen, 4,096 input tokens each, drawn once with this task's own client from the private pool (`tests/infb_efficiency/pool.jsonl.xz`), the checkpoint's tokenizer and the seed the file records; materialized once and byte-identical to the draw the witness first ran on (`--check` re-draws and compares) | `519ad95500653a841eac443122091a5d01c7e8e6a38035a266a6c4067ef6f253` |

## The supplied patches

| File | Bytes | sha256 |
|---|---|---|
| `environment/kernel-prereq.patch` (the `python/sglang/jit_kernel` paths of the landing squash that are supplied, not solver work) | 246,932 | `30476191e50c8ad1dd14400a0f7845da0756698e6cb836069995c4f712d8fd7a` |
| `environment/kernel-prereq.msg` (the preparation commit's message, byte-exact so the prepared base is reproducible) | 1,108 | `c1665f3567088540094210979c1b7dc1c7144aa50cd1995af2b1f3f4aaf85540` |
| `solution/solution.patch` (the prepared base to the landing, with the withheld kernel sources restored) | 1,207,423 | `0fd81aba8fb191abf0bfe9fd495c88a0047b88b99f91a7f121a52ade8f7ed995` |

## The agent's own copies in the image

There is no `/public` tree. The agent's materials are `/app/eval/deepseek_v4_flash/`, committed on
the prepared base as the evaluation-materials commit and restored before grading: the acceptance
test and the sanity kit (the bytes of the first table), their adapter, the checking cores the
grader's wrappers load (native support, the served recipe, accuracy, speculation, regressions,
fidelity and prefix caching), the efficiency client and its sample, and the public fidelity,
prefix-cache and replay examples. The acceptance test and sanity kit shipped there are the
reference's own bytes, as development feedback; the verifier never reads those copies.
