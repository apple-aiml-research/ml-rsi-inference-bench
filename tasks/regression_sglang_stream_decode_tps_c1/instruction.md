# Task: Recover this deployment's streaming generation throughput

We serve `gpt-oss-120b` from the tree in `repo/` and stream completions back to clients token by
token. On the job below this deployment returns generated tokens more slowly than these four GPUs
and these weights deliver. Find what is costing the throughput and repair it, in `repo/`.

## Scoring

Scored metric: **generated tokens/sec** as `aiperf` reports it — 4xB200, TP=4,
`openai/gpt-oss-120b`, streamed from the OpenAI-compatible completions endpoint after one untimed
warm-up, on the scored shapes listed in `public/workloads.json` (the first is input_len 1000,
output_len 1000, 5 requests at **concurrency 1**; the others vary the output length and the number of
streams). Only the ratio is scored, which is why no absolute figure is quoted. `repo/` is a git repo
tagged `inference_base` at the unmodified tree, so you can measure that baseline yourself.

Your score is continuous. The verifier benches `repo/` and the unmodified tree in the same run, on
the same GPUs, every shape against one server per tree, and scores the geometric mean, over the
scored shapes, of the measured speedup of `repo/` over the unmodified tree: 1.0x scores nothing,
every gain above it counts, and there is no cap. The guard shapes in the same file are measured the
same way and are not scored, but a tree that reads worse than the unmodified one on a guard shape
scores zero: a repair that pays for the scored shapes by slowing another is not a repair. Outputs
must not change: the verifier also decodes fixed prompts greedily on both trees and compares what
they generate, and a tree whose output differs scores zero whatever its speed.

A maintainer's own repair of this deployment exists and has been measured on this rig. You are not
told what it changed or what it reached; your result will be read against it. Aim past "faster than
before": stop when you cannot find more, not when it looks fixed.

## Measuring

`python3 public/run_benchmark.py` **is** the scored configuration, not a wrapper around it: it
launches a server against `repo/`, waits for it, benches every shape in `public/workloads.json`
against it, tears it down, and files the result under `iterations/`. Each call is a fresh server.
Run it once with `--repo <a checkout of inference_base>` and pass that file back as
`--baseline-json` to see every shape's ratio, the geometric mean over the scored shapes and the
guard readings, which is what you are scored on. Read it if you want the exact flags — it is the only
copy of them, so it cannot drift from what the verifier runs.

The server takes several minutes to come up on this model, so a bench that reports nothing is
usually a server that was still loading — and a syntax error costs you the same wait.
`python3 -m compileall repo/sglang` catches that in seconds.

## Also in `public/`

`workloads.json` — the scored and guard shapes the readings are taken at. `model` symlinks the weights.
Edit anything under `repo/`; `git diff inference_base` is your patch. How you get the speed back is
up to you — any change that makes this job faster counts, as long as the model still emits the same
tokens it does today.

Do not consult upstream history or releases newer than `repo/`; it is a sealed snapshot with no
remotes. Your time budget is in `/app/.timer/remaining_secs`.
