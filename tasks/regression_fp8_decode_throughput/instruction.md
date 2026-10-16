# Task: recover the decode-throughput regression on this 8×B200 server

This vLLM tree serves correctly — every request completes and the tokens are exactly right — but its
**mean time per output token** is higher than the work accounts for.

## Scoring

Scored metric: **mean TPOT**, measured on 8×B200, DP=8 with expert parallelism,
`nvidia/DeepSeek-R1-0528-NVFP4`, against a live server at the scored shapes in `public/workloads.json`,
which differ in how many requests are in flight. The verifier benches `repo/` and the pinned tree in
the same run, on the same machine, every shape against one server per tree, so no stored number can
flatter or penalise you. Your score is continuous: the geometric mean, over the scored shapes, of the
measured speedup `baseline_mean / candidate_mean` (TPOT is a cost, so above 1.0 means faster). 1.0x
scores nothing, every gain above it counts, and there is no cap. The guard shape in the same file is
measured the same way and is not scored, but a tree that reads slower than the pinned one there scores
zero: a repair that pays for the scored shapes by slowing another is not a repair.

Output must not change, and that is scored too: the verifier compares the two trees' decoding on fixed
prompts, and a tree that fails that check scores zero whatever its speed. Any change that makes this
deployment faster counts — you are not being matched against a reference patch.

A maintainer's own repair of this deployment exists and has been measured on this rig. You are not
told what it changed or what it reached; your result will be read against it. Aim past "faster than
before": stop when you cannot find more, not when it looks fixed.

## Measuring

```bash
python3 public/run_benchmark.py --with-baseline
```

It reproduces the scored run, every shape against one server per tree, and benches a pristine copy
alongside it; it prints each shape's ratio, the geometric mean over the scored shapes and the guard
reading. Each boot costs several minutes. Discard the first sweep after a cold start — it can read less than half the throughput of
the same configuration warm — and treat a few percent between repeats of one build as unchanged.

## Rules

Solve this from `repo/` and from your own measurements. Do not consult any source outside this
container about this engine's history or development, whatever its date — that includes upstream
history, pull requests, issues, commit messages, release notes, package mirrors and model hubs.
`repo/` is a sealed snapshot tagged `inference_base`, so `git diff inference_base` is your whole
patch. Do not edit anything under `public/`.

Budget: `cat /app/.timer/remaining_secs`. Land, verify and commit your best work before it runs out.
