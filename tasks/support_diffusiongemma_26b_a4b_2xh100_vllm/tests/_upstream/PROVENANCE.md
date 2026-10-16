# Provenance of adapted tests, protocols and interfaces

| Item | Source | Change |
|---|---|---|
| `infb_verify/test_parsers.py` | the landing (#45163, `eb28452b`): `tests/tool_parsers/test_gemma4_tool_parser.py`, `test_streaming_single_chunk_complete_tool_call` and `test_streaming_multi_chunk_batched_tool_calls` | the parser from `ToolParserManager.get_tool_parser("gemma4")` with `/model`'s tokenizer (real token ids per delta); assertions only through `extract_tool_calls_streaming`; each function's result as a JUnit property |
| `test.patch` | the landing: `tests/models/registry.py` (the `DiffusionGemmaForBlockDiffusion` fixture entry) and one line of `tests/models/utils.py` | verbatim hunks |
| `eval/diffusion_gemma/serving/reads.py`, `serving/reads.md` | vllm-project/vllm #57250 (`1b3b88ec`) with #57417 (`63d9ad0a`): the read mode for this model | comma-separated `vllm_xargs` fields; every position held at the seed; a step cap with no adaptive stop; the two-step form with step 0's self-conditioning at `t_max` (this benchmark's extension); temperature-1 FP32 log-probabilities with per-position targets; no commit |
| `eval/diffusion_gemma/serving/accuracy.py` | the base's `tests/evals/gsm8k/gsm8k_eval.py` (the 5-shot prompt and `get_answer_value`), the chat call of #47048, the temperature of #47980, the configuration of #46177 | chat completions at temperature 1.0 with the base harness's stops |
| `eval/diffusion_gemma/serving/images.py`, `semantic.py` | image fixtures and semantic-case construction adapted for this task (short factual and visual questions with one-word answers; the judging rule: a word-boundary match, an answer naming every candidate is not a hit) | credited; no file copied |
| `eval/diffusion_gemma/attend/policy.py` | FlashAttention's test rule (`max|o − r| ≤ 2 · max|o_bf16 − r|`) | the equal-logits rule added |
| `infb_efficiency/deadline.py`, `_paired_helper.py` | the task-owned kit of `support_qwen38_flash_next_4xb200_vllm` | byte for byte |
| `preflight.py` | the model-support tasks' shared preflight | the checkpoint check reads this task's two-checkpoint record |
| `grader.py` | the benchmark's shared grader | unchanged |
| `infb_verify/disposition.py` | the disposition of `support_qwen38_flash_next_4xb200_vllm` | the schema label |
