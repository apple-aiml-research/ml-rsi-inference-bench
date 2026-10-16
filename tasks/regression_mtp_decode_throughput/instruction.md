# Task: Recover this deployment's throughput with the MTP speculator enabled

**The symptom.** We serve `Qwen/Qwen3-Next-80B-A3B-Instruct` on 4×H100 (TP=4) with up to 256
requests in flight and the model's own multi-token-prediction speculator turned on. It
helps, but by far less than two speculative tokens should be worth. The cost side is
where it shows: enabling the speculator roughly doubles the median inter-token latency,
which a speculative step earns, but the tail grows far more than the median does. A
minority of emissions cost several times what a typical one does, and those are what
hold the throughput down.

Something in this tree causes it; find it and fix it in `repo/`. The responses are
unaffected — every request completes, and the two configurations emit within 0.2% of
the same number of output tokens.

Speculation is not free even when it works correctly, and a slower median per emitted
token is its ordinary price. Do not assume the whole gap is one bug.

## Scoring

Scored metric **total tokens/sec** (prompt + generated), served: `vllm serve` at
`max_num_seqs=256`, one freshly started server per scored shape and arm, each driven with
`vllm bench serve` — 2048 in / 1024 out at 128, 192 and 256 concurrent requests. The verifier
benches `repo/` and the pristine tree in the same run, on the same GPUs, shape for shape,
with the speculator on in **both** arms — turning the feature off is not a fix and does
not score. Your score is continuous: the geometric mean over the scored shapes of the
measured speedup of `repo/` over the pristine tree. 1.0x scores nothing, every gain above
it counts, and there is no cap. The guard shapes in the same file — 64 and 8 concurrent
requests, run against the 256-shape's server after it — are measured and not scored, but a
tree that reads slower there scores zero: a repair that pays for large-batch throughput by
slowing smaller batches is not a repair. Correctness is scored too: the verifier greedy-decodes
fixed prompts on both trees and requires the same tokens; a tree that fails it scores zero at any speed.

A maintainer's own repair of this deployment exists and has been measured on this rig.
You are not told what it changed or what it reached; your result will be read against
it. Aim past "faster than before": stop when you cannot find more, not when it looks
fixed.

## Reproducing

```bash
cd /app && python3 public/run_benchmark.py                                    # candidate only, every shape
cd /app && python3 public/run_benchmark.py --with-baseline --rounds 2         # a real ratio, every shape
cd /app && python3 public/run_benchmark.py --with-baseline --workloads in2048_out1024_c192_n750   # one shape
```

That runs the scored configuration against `repo/` — a fresh server per scored shape, arm
and round, the guards after the 256-shape — files the result under `iterations/`, and prints
each shape's ratio, the geometric mean over the scored shapes and the guard readings. Every
server starts on its own clean scratch, and the settings come from `public/workloads.json`. They are
the deployment's, not yours to tune — the verifier scores both arms with exactly them, so
a run of your own with different settings measures something the score does not.

Noise here is per-boot rather than per-request, so run each arm more than once and
alternate them. There is more than one way to make this workload faster and not all of
them are correct: check your own work on shapes the scored workload does not reach.

`repo/` is a sealed snapshot tagged `inference_base`, so `git diff inference_base` is your patch.
**Solve it from this tree and from your own measurements.** Do not fetch anything over the network:
no upstream sources at any ref, no pull requests or issues, no package installs. That includes the
pinned commit itself — comparing this tree against a copy of it is not part of the task.
Budget: `/app/.timer/remaining_secs`.
