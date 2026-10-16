<p>
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/g-head-dark.svg"><img src="assets/g-head-light.svg" width="880" alt="RSI-Inference Bench: A Long-Horizon Benchmark for Self-Improving Inference Systems. By Junteng Liu, Richard He Bai, Huangjie Zheng, Yizhe Zhang and Ruixiang Zhang. Core contributors: Junteng Liu (jliu999356@apple.com), Richard He Bai (richardbai@apple.com), Huangjie Zheng (huangjie.zheng@apple.com). Project lead: Ruixiang Zhang (ruixiangz@apple.com)."></picture>
</p>

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-b31b1b.svg)](CITATION.cff)
[![License](https://img.shields.io/badge/License-Apple-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://www.python.org/)
[![Tasks](https://img.shields.io/badge/Tasks-37-orange.svg)](#the-benchmark)

---

RSI-Inference Bench is a long-horizon benchmark for coding agents that improve the inference stack
of large language models. It follows a model through its life in a production engine, vLLM or
SGLang: supported on day 0, optimized after launch, and repaired when a regression lands. An agent
works on the real engine, on real GPUs, and is graded by running what it built.

## The Benchmark

<p>
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/g-rail-dark.svg"><img src="assets/g-rail-light.svg" width="880" alt="Evaluation pipeline: an agent gets a production engine (vLLM or SGLang) at a pinned commit, a model and 1 to 8 GPUs, and works offline for up to 8 hours; a separate verifier builds the edited engine, serves the model and measures it; the score is pass or fail, or the measured speed-up."></picture><br>
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/g-lifecycle-dark.svg"><img src="assets/g-lifecycle-light.svg" width="880" alt="The full lifecycle of a new model in an inference system. Day 0, when a new model is released: Day-0 Model Support, 10 tasks. Goal: serve the new model. Scored: pass or fail. In production, when the model goes live: Inference Optimization, 20 tasks. Goal: make it serve faster. Scored: speed-up. In maintenance, when a regression lands: Bug Fix, 7 tasks. Goal: find and fix a slowdown. Scored: speed-up."></picture><br>
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/g-stack-dark.svg"><img src="assets/g-stack-light.svg" width="880" alt="Comprehensive coverage of the AI inference stack: the 37 tasks across the inference stack, by layer. Engine: vLLM 20, SGLang 17. Hardware: H100 17, B200 20. Workload: prefill-heavy 11, decode-heavy 20, high load 14, long context 5. Modality: text 31, text + images 6. Methods: autoregressive 34, continuous diffusion 1, discrete diffusion 2. Parallelism: tensor parallel 30, data parallel 3, expert parallel 2. Model: model size (up to 10B 3, 10–30B 7, 30–100B 3, 100B–1T 18, 1T and up 2); experts (mixture of experts 30, dense 7); attention (MLA 10, sparse attention 14, linear / hybrid 12, sliding window 6, GQA 2); decoding (MTP 6, EAGLE3 3, DSpark 2, DFlash 1); precision (BF16 2, FP8 16, MXFP8 4, FP4 (MXFP4, NVFP4) 13)."></picture>
</p>

## Evaluate Your Own Model

Have a new model or agent? Run it on the benchmark and report its two scores.

1. **Get the code.** Clone this repository. It runs on Pier, a [Harbor](https://www.harborframework.com/docs/tasks)-compatible framework for evaluating coding agents.
2. **Run.** Run each task with your agent and model.

   ```bash
   uv sync
   uv run pier run -p tasks/<task> --env docker --agent <agent> --model <model>
   ```

3. **Report** one score per track, computed as the leaderboard computes it:

<p>
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/g-tracks-dark.svg"><img src="assets/g-tracks-light.svg" width="880" alt="Day-0 Model Support, 10 tasks: report pass@1. Full passes over attempts on each task, averaged over the tasks. A partial pass counts as a failure. Inference Optimization &amp; Regression Fix, 27 tasks: report speed-up. Each task's measured ratio against its baseline, as a geometric mean over the tasks. A result that fails its correctness check counts as 1.00×."></picture>
</p>

## Citation

```bibtex
@misc{liu2026rsiinference,
  title        = {{RSI-Inference}: A Long-Horizon Benchmark for Self-Improving Inference Systems},
  author       = {Liu, Junteng and Bai, Richard He and Zheng, Huangjie and Zhang, Yizhe and Zhang, Ruixiang},
  year         = {2026},
  howpublished = {\url{https://github.com/apple-aiml-research/ml-rsi-inference-bench}},
  note         = {GitHub repository}
}
```

## License

Released under the [Apple Sample Code License](LICENSE).
