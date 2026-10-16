# vLLM throughput optimization

You are optimizing vLLM 0.20.1 throughput for `deepseek-ai/DeepSeek-V4-Pro` on 8×B200, TP=8. Model: 1.6T total / 49B activated parameters, bfloat16. Toolchain: torch 2.10.0+cu128, CUDA 12.8.

Your goal: substantive source-level edits to vLLM that produce real speedup on hidden bench workloads. The verifier launches your candidate and the baseline (`inference_base` tag) against the same workloads and scores the geometric mean of per-workload speedups. CLI flag flips and `vllm/config.py` default-value changes alone are classified `category=config` and contribute 0 regardless of measured speedup. What is scored is the state of `repo/` you leave behind — your commits AND anything still uncommitted (`git diff inference_base`). You have a fixed time budget: `cat /app/.timer/remaining_secs` shows the seconds left (also `/app/.timer/elapsed_secs`, and `test -f /app/.timer/alert_5min` appears as the deadline nears). Pace yourself — land, verify, and commit your best candidate before it runs out.

## How to work

### Correctness gate

Run `scripts/run_correctness.py` to verify your edits don't break the model. If the correctness gate fails, your score is zero regardless of measured speed.

### Speed measurement

```bash
python3 scripts/run_benchmark.py --output /logs/agent/bench.json   # PAIRED — drift-controlled, requires clean commit
```

PAIRED runs your candidate and the baseline (`inference_base`) back-to-back in the same session so cross-session GPU drift cancels.

## What you can do

The design space is intentionally wide. Common starting points:

- **Operator fusion** — collapse kernels inductor doesn't auto-fuse.
- **Custom Triton kernels** — write Triton kernels for hot ops.
- **Multi-stream overlap** — concurrent kernel chains via aux CUDA streams.
- **Memory layout** — re-lay weights / KV / hidden state for alignment.
- **Kernel autotune** — sweep Triton autotune for this task's shapes.
- **Hardware features** — TMA, WGMMA, FP8 matmul (SM_90+).
- **Hot-path Python** — cache invariants, hoist work from per-token to per-step.
- **Algorithmic substitution** — compressed KV, quantization with custom dequant, alternative attention.

Edit policy: Python source AND `csrc/**/*.{cu,cpp,h}` are editable. After csrc edits, run `bash scripts/build_vllm_csrc.sh` to rebuild `_C` / `_moe_C` so your own paired bench measures it; `bash scripts/build_vllm_csrc.sh --restore` puts `repo/vllm/` back on the baseline `.so`. Also editable (HEAVY, scored): the vendored kernel dependency trees CMake compiles against, staged at `repo/.csrc_src/**` (`.cu/.cuh/.cpp/.cc/.h/.hpp/.inl`) — `bash scripts/build_vllm_csrc.sh` builds your copy, not the image's, so your own paired bench measures the edit.

## Profiling

Profile per `repo/docs/contributing/profiling.md`, via vLLM's cudaProfilerApi hooks so Nsight finalizes before multiprocess teardown:

```bash
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/target-linux-*/nsys /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys 2>/dev/null | head -1)
PY=${INFB_VLLM_PYTHON:-python3}
REPO="${INFB_WORKSPACE:-/app}/repo"
FI="${INFB_WORKSPACE:-/app}/.flashinfer_overlay"
export PYTHONPATH="$REPO$( [ -d "$FI" ] && printf ':%s' "$FI" )"
# IN/OUT/N come from public/workloads.json; the defaults below are a declared, scored pair.
IN=${IN:-1024}; OUT=${OUT:-1024}; N=${N:-20}
BENCH=( "$PY" -P -m vllm.entrypoints.cli.main bench throughput
        --model /app/model --tensor-parallel-size 8 --trust-remote-code
        --dataset-name random --random-input-len "$IN" --random-output-len "$OUT"
        --num-prompts "$N" --profile )
"$NSYS" profile --trace cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  -o /logs/agent/profile --force-overwrite=true \
  -- "${BENCH[@]}" --profiler-config.profiler=cuda
```

Model-specific, if it applies to yours: multimodal → `--skip-mm-profiling`; DeepSeek-V4 → `--kv-cache-dtype fp8`; Mamba / linear-attention → cap `--max-num-seqs` at the Mamba-cache block count.

## Constraints

Don't modify model weights (`.safetensors`, `pytorch_model.*`). Persistent debug output → `/logs/agent/`; `/tmp/` is per-shell.
