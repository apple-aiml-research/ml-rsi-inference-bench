# SGLang throughput optimization

You are optimizing **SGLang** 0.5.18 (the `sglang` serving runtime) throughput for `inclusionAI/LLaDA2.0-mini-preview` on 1×H100, TP=1. Model: 16.1B total / 1.4B activated parameters, bfloat16. Toolchain: torch 2.13.0+cu130, CUDA 13.0.

Your goal: substantive **source-level edits to SGLang** that produce real speedup on hidden bench workloads. The verifier launches your candidate (served via `python -m sglang.launch_server`) and the baseline (`inference_base` tag) against the same workloads and scores the geometric mean of per-workload speedups. SGLang server-launch flag flips alone (e.g. `--mem-fraction-static`, `--chunked-prefill-size`) are classified `category=config` and contribute 0 regardless of measured speedup — you must change the source. What is scored is the state of `repo/` you leave behind — your commits AND anything still uncommitted (`git diff inference_base`). You have a fixed time budget: `cat /app/.timer/remaining_secs` shows the seconds left (also `/app/.timer/elapsed_secs`, and `test -f /app/.timer/alert_5min` appears as the deadline nears). Pace yourself — land, verify, and commit your best candidate before it runs out.

## How to work

Edit `repo/sglang/**` — **this tree, not the venv copy**; `import sglang` resolves here via `PYTHONPATH=<repo>`. `scripts/run_correctness.py` and `scripts/run_benchmark.py` are both SGLang-aware (serve via `sglang.launch_server` + `sglang.bench_serving`).

### Correctness gate

Run `scripts/run_correctness.py` to verify your edits don't break the model. If the correctness gate fails, your score is zero regardless of measured speed.

**Startup limit.** The verifier gives every server start 60 minutes to begin serving. Its first start of your candidate, for this gate, begins with empty compile and autotune caches, and missing the limit there fails the gate. Your own runs reuse warm caches; to time a cold start, run `python3 scripts/run_correctness.py --workdir <new empty dir>`.

### Speed measurement

```bash
python3 scripts/run_benchmark.py --output /logs/agent/bench.json   # PAIRED — drift-controlled, requires clean commit
```

PAIRED runs your candidate and the baseline (`inference_base`) back-to-back in the same session (each = one `sglang.launch_server` + `sglang.bench_serving` HTTP client per side) so cross-session GPU drift cancels.

## This model is not autoregressive, and the schedule is the verifier's

Both servers -- yours and the baseline -- are launched with `--dllm-algorithm LowConfidence`. Generation proceeds by iteratively unmasking positions inside a **32-token block**; attention inside the block is bidirectional, so "one forward pass buys one token" does not hold here. Block size is not a flag at this pin: sglang reads it from its own per-architecture table. You do not enable this mode and you cannot turn it off -- flag flips are `category=config` (0 credit) regardless.

Read the design space below in that light. `LowConfidence` unmasks **every** position whose confidence clears its threshold in one step and falls back to a single argmax when none does, and the block loop runs `max_steps = block_size + 1` with an early break once the block fills. So the number of forward passes per block is a function of the prompt, not a constant -- kernels, graph captures and batching heuristics written for prefill's many tokens or plain decode's one frequently do not attach to a 32-wide unmasking step, and that gap is this task's headroom. Concretely open to you: the diffusion loop itself (`python/sglang/srt/dllm/**`), the attention path it drives, the per-step confidence/gather work, and the graph-capture and launch overhead around all of it.

Two facts worth having before you start, both MEASURED on this model at this pin, 1xH100:

* **Per-forward overhead dominates.** Serving with cuda graphs disabled costs **4.56x** throughput (415.0 vs 91.0 tok/s on the scored workload shape). A 32-token block spends up to 33 forwards going through a Python-level scheduler, so launch and dispatch cost is a first-order term rather than a rounding error.
* **Throughput depends on prompt content.** The bench reads a fixed real-prompt bank, not random tokens. Measured on the same server, an arithmetic ramp of token ids reads **1.60x FASTER** than real text (662.2 vs 415.0 tok/s) because a degenerate continuation clears the confidence threshold at every position. Tuning against synthetic prompts optimises a regime this task does not score.

One warning specific to this task, and it is the one that decides whether your run counts. **The correctness gate compares the GENERATED TOKEN SEQUENCE against the baseline's, position by position.** Unmasking more positions per step, raising the threshold, or cutting the step count all make the model faster and make it emit different tokens -- that is a change to the model's output, not an optimisation of it, and it scores 0 no matter what the throughput says. Make the same schedule cheaper to execute; do not make it a different schedule.

**Check that cheaply, and check it often.** `scripts/run_correctness.py` with no arguments boots both servers itself and takes ~15 minutes, which is too slow to run after every edit. It has a fast mode that reuses a server you already have up:

```bash
# ONCE, against a PRISTINE server, before you change anything:
python3 scripts/run_correctness.py --server-url http://127.0.0.1:$PORT     --capture-reference /logs/agent/gate_ref.json
# after every edit, seconds instead of minutes:
python3 scripts/run_correctness.py --server-url http://127.0.0.1:$PORT     --reference /logs/agent/gate_ref.json
```

It runs the same arithmetic and the same floor as the scored gate, and unlike a check you write yourself it also runs the second gate -- a prompt-tail perturbation that must move the first block. Capture the reference BEFORE your first edit: one captured afterwards compares your candidate against itself and can never fail.

## What you can do

The design space is intentionally wide. Common starting points:

- **Operator fusion** — collapse hot ops the compiler doesn't auto-fuse.
- **Custom Triton kernels** — write Triton kernels for hot paths.
- **Multi-stream overlap** — concurrent kernel chains via aux CUDA streams.
- **Memory layout** — re-lay weights / KV / hidden state for alignment.
- **Kernel autotune** — sweep autotune for this task's shapes.
- **Hardware features** — TMA, WGMMA, FP8 matmul (SM_90+).
- **Hot-path Python** — cache per-step invariants, hoist work out of the per-token loop.
- **Algorithmic substitution** — compressed KV, quantization with custom dequant, alternative attention.

Edit policy: Python at `repo/sglang/**` AND the runtime-JIT CUDA kernels at `repo/sglang/kernels/jit/csrc/**/*.{cu,cuh}` are editable+scored — SGLang JIT-recompiles them at serve time from `$repo`, NO rebuild step (baseline gets the unedited kernel). A prebuilt `.so` is never itself scored — what is scored is the source it is rebuilt from, named below. Also editable (HEAVY, scored): `repo/sgl-kernel/**/*.{cu,cuh,cpp,h}` and its vendored kernel deps under `repo/sgl-kernel/_editsrc/**` — after each sgl-kernel edit run `bash scripts/build_sgl_kernel.sh` (~30-90 min) so your own paired bench measures it.

## Profiling

Profile per `docs/developer_guide/benchmark_and_profiling.md`. `$repo` goes first on `PYTHONPATH`, so this profiles YOUR edited `sglang/` rather than the venv copy:

```bash
REPO="${INFB_WORKSPACE:-/app}/repo"
PY="$(python3 -c 'import sys;sys.path.insert(0,"/app/scripts");from _target_env import resolve_venv_python as r;print(r())' 2>/dev/null)"
NSYS="$(command -v nsys || ls /opt/nvidia/nsight-systems/*/target-linux-*/nsys /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys 2>/dev/null | head -1)"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
# Export PYTHONHOME only if it actually works: nsys can otherwise lose the venv's stdlib and die
# `No module named encodings`, but a base prefix that has no stdlib causes the same crash.
PYHOME="$("$PY" -c 'import sys;print(sys.base_prefix)' 2>/dev/null)"
PYTHONHOME="$PYHOME" "$PY" -c 'import encodings' 2>/dev/null && export PYTHONHOME="$PYHOME"
FI="${INFB_WORKSPACE:-/app}/.flashinfer_overlay"
export PYTHONPATH="$REPO$( [ -d "$FI" ] && printf ':%s' "$FI" )"
BENCH=( "$PY" -P -m sglang.bench_one_batch --model-path "${INFB_WORKSPACE:-/app}/model" --tp-size 1
        --trust-remote-code --batch-size 8 --input-len 128 --output-len 32 --profile )   # tune these; keep small
"$NSYS" profile --trace-fork-before-exec=true --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  -o /logs/agent/profile --force-overwrite=true "${BENCH[@]}" --profile-activities CUDA_PROFILER
"$NSYS" stats --report cuda_gpu_kern_sum /logs/agent/profile.nsys-rep   # top GPU kernels
```

## Constraints

Don't modify model weights (`.safetensors`, `pytorch_model.*`). Persistent debug output → `/logs/agent/`; `/tmp/` is per-shell.
