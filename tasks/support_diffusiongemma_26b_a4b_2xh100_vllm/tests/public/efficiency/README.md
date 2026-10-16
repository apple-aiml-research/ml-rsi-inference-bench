# Whole-serving efficiency

`client.py` is the client both the reference and a candidate are measured with, in each deployment's
fixed-work configuration (`/app/eval/diffusion_gemma/serving/recipes.py`: `tp1_fixed` on GPU 0,
`tp2_fixed` on GPUs 0 and 1; every canvas runs all 48 denoising steps, so the work per request is
fixed).

- Workload: 4,096 input tokens and 1,024 output tokens per request with EOS ignored, counted with
  `/model`'s tokenizer; every request carries a fresh cache salt.
- C1: output tokens per second of one request end to end (1,024 / wall), the best of up to three
  attempts.
- C4: aggregate output tokens per second over a draw of 16 requests at concurrency 4; three draws,
  their median.

```sh
python /public/efficiency/client.py --base-url http://127.0.0.1:8000 --mode c1 --output /tmp/c1.json
python /public/efficiency/client.py --base-url http://127.0.0.1:8000 --mode c4 --output /tmp/c4.json
```

The public prompts are natural-text documents from `sample.jsonl.xz` (provenance in
`sample.provenance.json`), tokenized and cut to 4,096 tokens. Grading supplies its own prompts of
the same length and compares each statistic with the reference's, measured on the same allocation
in the same verifier run.
