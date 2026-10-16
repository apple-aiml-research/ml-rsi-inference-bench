# Task: Recover this server's decode throughput on a quantised mixture-of-experts model

Generation is slower than this model and this GPU should give. Find what is costing it and repair it,
in `repo/`.

## What we see

We start the server with no tuning flags and drive it with the load below. The scored number is what
comes out of that run.

```bash
cd /app && PYTHONPATH=repo/python "$INFB_VLLM_PYTHON" -m sglang.launch_server \
  --model-path model --host 127.0.0.1 --port 31137 --tp-size 1

cd /app && PYTHONPATH=repo/python "$INFB_VLLM_PYTHON" -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 31137 \
  --dataset-name random --dataset-path /opt/infb/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 1024 --random-output-len 512 \
  --random-range-ratio 1 --num-prompts 64 --max-concurrency 4
```

**Its output token throughput is well below what this model sustains on this hardware.**

Nothing fails: no request errors, no out-of-memory, no fallback warning in the log, memory use is
normal, and the model's answers are correct.

Everything else about the shape of this — which knobs change it, what a profile says, where the time
goes — is measurable from the box you are on, and measuring it is the task.

One thing to be careful about when you check your work: two runs of the same build produce identical
tokens, so a changed continuation is worth investigating — but it is not by itself proof that a
candidate fix is wrong. Check whether the answers are still right.

## Scoring

Scored metric: **sustained output token throughput**, on the scored shapes listed in
`public/workloads.json`. The load above is one of them; the others vary how many requests are in
flight. 1×H100-80GB, TP=1, `openai/gpt-oss-20b`, server started with no flags beyond `--tp-size 1`.
The verifier benches `repo/` and the unmodified tree in the same run, on the same GPU, every shape
against one server per tree, with a discarded warmup and the two trees taking turns to go first.
`repo/` is a git repo tagged `inference_base` at the unmodified tree, so you can produce that
comparison yourself.

Your score is continuous: the geometric mean, over the scored shapes, of the measured speedup of
`repo/` over the unmodified tree. 1.0x scores nothing, every gain above it counts, and there is no
cap. The guard shapes in the same file are measured the same way and are not scored, but a tree that
reads slower than the unmodified one on a guard shape scores zero: a repair that pays for the scored
shapes by slowing another is not a repair.

Outputs are checked too. The verifier sends both trees the same fixed text and compares the
log-probability each one assigns to every token of it; a tree whose numbers move beyond a small
numerical tolerance scores zero whatever its speed. Generated text is not compared.

A maintainer's own repair of this deployment exists and has been measured on this rig. You are not
told what it changed or what it reached; your result will be read against it. Aim past "faster than
before": stop when you cannot find more, not when it looks fixed.

## Reproducing

`python3 public/run_benchmark.py` starts the server from `repo/`, drives the load above and then every
other shape in `public/workloads.json` against it, tears the server down, and writes
`iterations/iter_N.json` (~6-8 min including model load; `--workloads anchor` runs the load above
only). Run it once with `--repo <a checkout of inference_base>` and pass that file back as
`--baseline-json` to see every shape's ratio, the geometric mean over the scored shapes and the
guard readings, which is what you are scored on. `python3 public/check_candidate.py` is a cheap "does the tree still
import". `public/` also holds `workloads.json`; `model` symlinks the
weights; edit anything under `repo/` — `git diff inference_base` is your patch.

Do not consult upstream history or releases newer than `repo/`; it is a sealed snapshot with no
remotes. Your time budget is in `/app/.timer/remaining_secs`.
