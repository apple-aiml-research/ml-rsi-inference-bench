# MiniMax-M3 whole-serving measurements

Start either deployment from `/app/eval/minimax_m3/serving/recipes.py`, then run:

```bash
python /public/efficiency/client.py measure --engine vllm \
  --base-url http://127.0.0.1:8000 --tokenizer /model/tokenizer.json \
  --texts /public/efficiency/sample.jsonl.xz --concurrency 16
```

The client measures concurrency-one decode speed and concurrency-16 output throughput.
Prompts have 4,096 input tokens and greedy completions have 512 output tokens with EOS
ignored. Unique cache salts keep earlier measurements from warming these requests.
A request counts only after a successful, complete fixed-length streamed response.

Decode speed is the median over eight requests after two warmups, using the interval
between the first and last emitted token. Throughput uses 64 requests at concurrency
16 after a 16-request warmup, divided by the full request interval. The client records
its own timestamps, token counts, seed and invalid-response diagnostics.

Use `--axis decode_c1` or `--axis throughput` to measure one axis. Use `--seed` to
repeat a workload draw. Run each deployment on idle GPUs and preserve the JSON record.
The MSA operators' own timing protocol is separate (`/app/eval/minimax_m3/msa/README.md`).

The sample's works and digests are recorded in `sample.provenance.json`. The public
and graded clients have identical source; grading supplies separate inputs and policy.
