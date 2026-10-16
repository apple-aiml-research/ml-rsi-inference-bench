# Upstream accuracy measurement

The accompanying test adapts the upstream workload to the common offline measurement
and the stated ordinary recipe. The public YAML contains generation and dataset
settings. Measure against a running ordinary server:

```bash
python /app/eval/qwen38/run.py accuracy --base-url http://127.0.0.1:8000 \
  --inputs /public/upstream-test/configs/Qwen3.8-Flash-Next-FP8.yaml \
  --dataset-root /public/datasets --output /tmp/accuracy.json
```

EvalScope evaluates the full GSM8K and AIME 2025 datasets with the supplied thinking-mode
generation settings. The measurement reports dataset scores and preserves EvalScope's
response records under `/tmp/qwen38-accuracy`. The protocol rejects incomplete or invalid
reports. It needs all four GPUs and may take tens of minutes.
