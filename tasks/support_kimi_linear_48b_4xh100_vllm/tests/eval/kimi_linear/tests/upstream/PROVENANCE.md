# Provenance of this directory

`gsm8k_eval.py`, `__init__.py` and `README.md` are vLLM's own, taken **verbatim** from
`tests/evals/gsm8k/` at the reference commit. Byte identity checked, not asserted:

| file | bytes | sha256 |
|---|---|---|
| `gsm8k_eval.py` | 7,525 | `16bc290eeb5fd4eff4c1995ae19cd723e35735d243ae6aa3fab659aa8a2ceec7` |
| `__init__.py` | 107 | `d2ef58e5eb24cc6ce8f7c5d2d23bce6c3aa392d375874b5f79a020640187b507` |
| `README.md` | 951 | `7fd755f2a6cd6d3c9753ba52737df22ed738986ed01f97729098708efb88e751` |

**Not taken: `test_gsm8k_correctness.py`, `conftest.py` and `configs/`.** That pytest wrapper launches
its own server with `--max-model-len 4096 --enforce-eager`, which is eager mode with no compilation and
no CUDA graphs — a configuration this task does not serve, and one under which a speed or efficiency
measurement would be meaningless (eager runs roughly ten times slower per steady-state request
than compiled). There is also no config for this model in `configs/`, then or now. So the evaluator is
used against a server launched at **the task's own recipe** (`../../launch.py`), and that substitution
is the one adaptation.

## Running it offline

`load_gsm8k_data()` calls `download_and_cache_file(url)`, which caches to `/tmp/<basename>` and
**returns early when that file already exists**. So the evaluator needs no edit at all to run offline:
put the two dataset files where it would have cached them. `../acceptance.py --mode accuracy` does
exactly this against a running server:

```
cp /app/eval/kimi_linear/datasets/gsm8k/train.jsonl /app/eval/kimi_linear/datasets/gsm8k/test.jsonl /tmp/
/opt/vllm/bin/python -c "
from importlib import util
spec = util.spec_from_file_location('g', '/app/eval/kimi_linear/tests/upstream/gsm8k_eval.py')
g = util.module_from_spec(spec); spec.loader.exec_module(g)
print(g.evaluate_gsm8k(num_questions=1319, num_shots=5, max_tokens=256, port=8000, temperature=0.0, seed=42))
"
```

The protocol is upstream's own and is what the graded node runs: all 1,319 test questions, 5-shot from
the first five `train.jsonl` rows, `Question: …\nAnswer:` prompts, `max_tokens` 256, temperature 0,
seed 42, the stop list `["Question", "Assistant:", "<|separator|>"]`, and last-number extraction.

Two properties of it worth knowing, both read in the source rather than inferred:

- **It has no concurrency parameter.** It issues one request per question and gathers all 1,319 at
  once on a single `aiohttp.ClientSession` with no connector, so aiohttp's default
  `TCPConnector(limit=100)` is what holds 100 requests in flight. That number cannot be changed
  without editing the file.
- **Its `ClientTimeout(total=600)` starts when each coroutine starts**, and every request is issued at
  once — so a server too slow to finish all 1,319 questions within 600 s of the run's start loses the
  unfinished ones as empty strings, which score wrong. It fails on accuracy, not by hanging.
