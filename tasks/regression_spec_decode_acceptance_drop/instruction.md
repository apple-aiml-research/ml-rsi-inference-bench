# Task: Recover this deployment's speculative-decoding throughput

**The symptom.** We serve `meta-llama/Llama-3.1-8B-Instruct` on one H100 with EAGLE3 speculative decoding through
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`, seven draft tokens per step. On the current build the drafter's tokens are
accepted far less often than the previous build managed, and throughput has fallen with it. Something in the tree
causes it; find it and undo its effect in `repo/`.

## Scoring
Scored metric **total tokens/sec** on the scored shapes in `public/workloads.json`: the same natural prompts and 512
output tokens each, EOS ignored, served at 1, 8 and 32 streams. Your score is continuous, the geometric mean over the
scored shapes of the measured speedup of `repo/` over the pristine tree, benched in the same run on the same GPU, shape
for shape: 1.0x scores nothing, every gain above it counts, there is no cap. The guard in the same file, the first-token
latency of the 8-stream shape, is measured and not scored, but a tree that reads slower there scores zero: a repair that
pays for throughput by slowing the first token is not a repair. Correctness is scored too: speculative decoding must
return what the target model returns without a drafter, so the verifier serves the pristine tree without one as the
reference and requires your answers to reproduce it request for request; a tree that fails that scores zero at any speed.
A maintainer's repair of this deployment has been measured on this rig and your result is read against it, so stop only
when you cannot find more.

## Reproducing
```bash
cd /app && python3 public/run_benchmark.py --with-baseline
```
That benches the scored shapes against `repo/` and a pristine copy alongside, shape for shape, and prints each shape's
ratio, the geometric mean and the guard reading. **Budget ~35 minutes per call**: a discarded warm-up plus both arms at
every shape, each booting a fresh engine (a shape's stream count is a server setting); `--workloads <name>` restricts it
to one shape. Settings come from `public/workloads.json` and are the deployment's, not yours to tune: the verifier scores
both arms with exactly them; the raw files under `iterations/.raw/` carry a precondition check. `repo/` is a sealed
snapshot with no remotes, tagged `inference_base`, so `git diff inference_base` is your patch. Diagnose it from the tree
in front of you: the upstream project's history, issues and pull requests are out of scope. Budget: `/app/.timer/remaining_secs`.
