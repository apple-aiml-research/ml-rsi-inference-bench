# Task: Recover this deployment's time to first token

We serve multimodal chat on one H100. Requests carry one, two or four JPEGs, generated
at resolutions drawn at random inside a fixed range, with a short text prompt.

Time to first token is far higher than this deployment should need, and
essentially every request pays it. The outputs are correct — it is only slow.
Make it faster, in `repo/`. Any change that holds up counts; there is no
particular fix being looked for.

## Scoring

Scored metric: **time to first token**, lower is better. Each request generates one
token, so a request IS a prefill and the ratio is exactly the TTFT speedup. The scored
shapes are listed in `public/workloads.json`: 120 requests of two images, 120 of one
image and 80 of four images, each at resolutions drawn inside the same range, one
request in flight at a time. The verifier serves exactly those shapes, on a server
started with the flags `public/run_benchmark.py` starts it with — it is the deployment
that has to get faster, not the benchmark.

Your score is continuous: the geometric mean, over the scored shapes, of the measured
speedup of `repo/` over the unmodified tree, both benched in the same run, on the same
GPU, every shape against one server per tree. 1.0x scores nothing, every gain above it
counts, and there is no cap. The two guard shapes in the same file — the two-image set
sent a second time to the same server, and 64 text-only prompts of one fixed length — are
measured and not scored, but a tree that reads slower there scores zero: a repair that
pays for the scored shapes by slowing another is not a repair. Outputs must not change:
the verifier compares the two trees' logprobs on fixed prompts, and a tree that fails
that check scores zero whatever its speed.

A maintainer's own repair of this deployment exists and has been measured on this
rig. You are not told what it changed or what it reached; your result will be read
against it. Aim past "faster than before": stop when you cannot find more, not
when it looks fixed.

## Working

`python3 public/run_benchmark.py` starts a server from `repo/`, drives every shape in
`public/workloads.json` against it with `public/_mm_ttft_client.py` — the client the
scored run uses — tears it down and writes `iterations/iter_N.json`. `--with-baseline`
also stages the unmodified tree from the `inference_base` tag and benches it the same
way, printing each shape's ratio, the geometric mean over the scored shapes and the
guard readings; `--rounds 2` alternates the two trees; `--workloads <name>` restricts a
call to one shape. Two cautions: bench each tree more than once and alternate them, and
keep the discarded warm-up — the first server in a fresh process pays one-time
initialisation.

`repo/` is a git repo tagged `inference_base` at the unmodified tree; `git diff
inference_base` is your patch.

Do not consult upstream history or releases newer than `repo/`; it is a sealed
snapshot with no remotes. Your time budget is in `/app/.timer/remaining_secs`.
