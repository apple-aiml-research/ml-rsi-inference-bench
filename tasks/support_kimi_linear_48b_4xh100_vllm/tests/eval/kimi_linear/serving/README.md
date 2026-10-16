# Serving measurements

The benchmark supplies `client.py` so you can measure the same per-user decode
speed and concurrent output throughput that evaluation measures. Both are
reported in tokens per second; higher is better while required behavior holds.
The natural-text sample is public development input. `sample.provenance.json`
identifies its sources and digests.

With the deployment in `../launch.py` running:

```sh
python /app/eval/kimi_linear/serving/client.py measure --engine vllm \
  --base-url http://127.0.0.1:8000 --tokenizer /model \
  --texts /app/eval/kimi_linear/serving/sample.jsonl.xz --concurrency 64
```

This measures both serving dimensions. `measure --help` describes the options,
including selecting one axis or repeating a sample with a seed. The client
reports measurements and request errors, not a task pass verdict. Public results
support optimization; they do not establish acceptance on the separate
evaluation inputs. Both serving metrics and KDA efficiency are required for
acceptance, together with the model behavior specified in the task.
