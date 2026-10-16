# Upstream material the verifier runs, and where each byte comes from

Reference commit: `sgl-project/sglang` `97c6978369ac1e04c91fcc01c98acc25129a6000` (the landing of
`sgl-project/sglang#36507`, 2026-09-06). Base commit: `a9944aec01e281f5c12c9651f981a53bdece6989` (its parent).

## The registered acceptance test (carried by `tests/test.patch`)

| Upstream path at the reference | git blob | sha256 |
|---|---|---|
| `test/registered/models_e2e/test_glm53_flash_b200.py` | `cd9993dd0376c801bbccff351125d4615492ff31` | `9dbbc47eb0abdcf1af64fbf8b6d1c9fec7be1df2c74f7e4f9f0428efdb000755` |

`tests/test.patch` is `git --no-pager diff --no-ext-diff --binary <base> <reference> -- test/registered/models_e2e/test_glm53_flash_b200.py`
(3,902 bytes, sha256 `7a39dd923e7986456cfb62e987f3fa27c76182a18fea84602da402200f61cab7`), applied by
`grader.py prepare` and re-applied by `tests/test.sh` after the evaluation materials are reset. Verbatim.

## The test helpers the test imports (byte-identical at base and reference; restored to the base blobs by `tests/test.sh`)

| Upstream path | git blob (same at both commits) | sha256 |
|---|---|---|
| `python/sglang/test/kits/eval_accuracy_kit.py` | `034927578d855f3176308941bc257519bd90b271` | `b13b2634b758c02d7005230e75e9ffa0c1c34217f08dcba08b7d20542b2ad367` |
| `python/sglang/test/kits/spec_decoding_kit.py` | `19f1b480deae4c2dc5ba4934e4c441dfaec3978f` | `76d3ba7b58e2bc616146dd4088707047601f42a7ae6a27c48c4cfeb688ad9dde` |
| `python/sglang/test/send_one.py` | `678d7444cfc31f790479d5318cda30e86d60fc3c` | `92b0e2225caf431d982b4368b62cb15fb71f07cfe2e38e7e9833a2902b405493` |
| `python/sglang/test/run_eval.py` | `cf7d9a0f770473a88d133b13ae74c82a9315a6f4` | `bc61ec24b510d39c70c0dc6fe16bd8838fe41517bd046b4eafd7dc201e1ace84` |
| `python/sglang/test/simple_eval_mixed_prefix_gsm8k.py` | `61b997952a95fd54786abf97f0ceff39e6015065` | `4cd17183697484bf1d45a903da1d43316170ea9cfad151a054873c9dee3fda23` |
| `python/sglang/test/simple_eval_common.py` | `6970643c191938da1aca03d1e8f7a6873d0cbb7b` | `68f1122178967eb8c119fa25f132be9bf53b96815b1ae6e0ae486eea5def0354` |
| `python/sglang/test/kl_test_utils.py` | `28efcbfa9ab582cb381fcc065c273e4b05015331` | `7e329bc8b9e01edeb138e3bfa4f51907ee473355e973f03911110155923ea147` |
| `python/sglang/test/test_utils.py` | `8c5ae9eaee4951b6a49ed128cdebd55f5dde5d93` | `80fbd39f9b959be51d791fe4efdda395aa34527a17ca57df1f25c6b99fe256ec` |
| `python/sglang/test/ci/ci_register.py` | `57bee1290c2dd1a76bdd19a113396847b1b116b1` | `697fffeb402a782a7213009036f0dc65f204004ced0defb3579eef95264ad201` |

Because these files are identical at both commits, a diff onto the base cannot carry them; the
verifier protects them by restoring `python/sglang/test` and `test/` to the base blobs after the
candidate's patch is applied (`git checkout <base> -- …`, `git clean`), then re-applying
`test.patch`. Nothing under those paths in a submission plays any part in grading.

## The evaluator's data

`test.jsonl` of `openai/grade-school-math` (`grade_school_math/data/test.jsonl` at the
repository's `master`, 1,319 rows, 749,738 bytes, sha256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`). The task image fetches it
from that URL at build time, verifies the digest, and keeps it at
`/app/eval/glm53_flash/tests/data/gsm8k/test.jsonl` (the agent's public copy). `tests/test.sh`
copies it to `/tmp/test.jsonl`, the path
`sglang.utils.download_and_cache_file` returns without a network when the file exists, so the
registered test runs verbatim and offline on upstream's own rows (20 demonstrations, questions
20 to 519). No copy is kept in this repository.

## DeepSWE's grader and frame

`tests/grader.py` is DeepSWE's `tests/grader.py` (`datacurve-ai/deep-swe` at `0b9fabbb`), with
comments naming internal tooling removed for this release, so the sha256 here is of the file as
shipped and not of upstream's: `5b53d19c24d86b345274c8ec4f223b0df9ab620784a7f497dd156852e495771d`.
`tests/test.sh` carries DeepSWE's shared frame, with the comments naming internal tooling
removed, around this task's RUN TESTS block.
`pre_artifacts.sh` is the form DeepSWE used until its commit `d7a1031` (2026-08-06), the one the
vendored Pier runs.

`tests/infb_efficiency/` is this task's own efficiency verifier; its own `PROVENANCE.md` records what it
measures. The agent's copy of its client is `/app/eval/glm53_flash/serving/client.py`.
