# Task: Recover the decode-throughput regression under speculative decoding

This vLLM tree serves correctly: it answers as well as a repaired engine does, nothing errors and
nothing is logged. The server simply sustains materially fewer tokens per second than it should for
this configuration.

## Scoring

Scored metric: **total tokens/sec** (prompt + generated) on 1×B200, TP=1, `openai/gpt-oss-120b`
with the `nvidia/gpt-oss-120b-Eagle3` drafter, on the scored shapes in `public/workloads.json`: 1024 in / 1024 out
at 256, 384 and 512 concurrent requests (the engine's `max_num_seqs`, so each shape boots its own engine).

The clock starts once the engine is built and stops when the last request returns, so engine
construction is not timed. Every request runs to exactly `output_len` tokens, so both arms produce
the same token count and the ratio is decided entirely by elapsed time.

Your score is continuous. The verifier benches `repo/` and the pinned tree in the same run, on the
same GPU, shape for shape — the baseline is re-benched live, never read from a file — and scores the
geometric mean over the scored shapes of the measured speedup of `repo/` over the pinned tree: 1.0x
scores nothing, every gain above it counts, and there is no cap. The guard shape in the same file, 128
concurrent requests, is measured and not scored, but a tree that reads slower there scores zero: a
repair that pays for high-concurrency throughput by slowing lower concurrencies is not a repair. Outputs
must not change: the verifier compares the two trees' logprobs on a fixed token sequence, and a tree
that fails that check scores zero whatever its speed.

A maintainer's own repair of this deployment exists and has been measured on this rig. You are not
told what it changed or what it reached; your result will be read against it. Aim past "faster
than before": stop when you cannot find more, not when it looks fixed.

## Measuring

`public/run_benchmark.py` is the harness the verifier runs, on every shape above, and it prints each
shape's ratio, the geometric mean over the scored shapes and the guard reading. A repair has to hold
there, not only under other settings you find along the way. Each shape is its own engine boot, so a
full `--with-baseline` round is long; `--workloads <name>` restricts a call to one shape.

```bash
python3 public/run_benchmark.py                                          # candidate only, every shape
python3 public/run_benchmark.py --with-baseline --rounds 2               # a real ratio, every shape
python3 public/run_benchmark.py --with-baseline --workloads in1024_out1024_c512   # one shape
```

Two cautions: **run each arm more than once and alternate them** — a single unrotated sample of this
workload can show a difference in the wrong direction; and **discard a warm-up**, because the first
run in a fresh process pays one-time initialisation.

## Workspace

`repo/` is the vLLM source tree and a real git repo — edit anything in it, and `git diff
inference_base` shows your total patch. `model` symlinks the weights, `public/` holds the files
above, and `run_benchmark.py` writes each iteration into `iterations/`. A probe you write yourself must put
`repo/` on `PYTHONPATH`, or it will not find vLLM at all: `repo/` is the only importable engine
tree in this image.

## Anti-leak

Do **not** look up upstream pull requests, issues, or any vLLM ref newer than the pinned commit.
`repo/` is a sealed snapshot at that commit with no remotes. Solve it from the tree and from
measurement.

You have a fixed time budget: `cat /app/.timer/remaining_secs` (also `elapsed_secs`, and
`/app/.timer/alert_5min` appears as the deadline nears). Land, verify and commit your best work
before it runs out.
