# SGLang diffusion-engine interfaces

`python/sglang/multimodal_gen/` is the image/video diffusion engine. The
public entry point is `DiffGenerator`; it is separate from LLM serving.

```python
from sglang.multimodal_gen import DiffGenerator

if __name__ == "__main__":
    gen = DiffGenerator.from_pretrained(model_path="/app/model", num_gpus=1)
    out = gen.generate(sampling_params_kwargs=dict(
        prompt="A red ceramic teapot on a wooden table", seed=0,
        num_inference_steps=28, guidance_scale=7.0,
        height=1024, width=1024, return_frames=True, save_output=False,
    ))
```

## Existing interfaces

- `registry.py` resolves model identifiers to sampling and pipeline configs
  through `register_configs(...)`. Check how the pinned engine resolves
  local checkpoint directories as well as model identifiers.
- `configs/` contains architecture, pipeline and sampling configuration.
- `runtime/models/` contains native components; `runtime/loader/` loads their
  checkpoint weights.
- `runtime/pipelines/` and `runtime/pipelines_core/` compose input validation,
  encoding, denoising and decoding.
- `runtime/layers/attention/` supplies the attention implementations required
  by the task.

Inspect these interfaces in the provided checkout. Choose model-specific
filenames and helper classes freely; no particular file list is required.
Preserve existing model behavior when extending shared interfaces.

## Execution contract

Follow `instruction.md` for conditioning, precision, generation and serving
behavior. Layer-level Diffusers building blocks are allowed within native
components; complete external models, pipelines and loaders are not.
Parallelism and batching must execute genuine shared or partitioned work,
and the returned images must consume that work.

Use a `__main__` guard because workers use `spawn`. The example above exercises
one GPU; the instruction gives the supported two-GPU configurations.

## Python environment

`python` and `python3` use the task's prepared interpreter and dependencies.
The default import path loads SGLang from `/app/repo/python`, including your
edits. This also applies to new shells; no virtual-environment activation or
package installation is needed. You can check the entry point with:

```bash
python -c 'import sys, sglang; from sglang.multimodal_gen import DiffGenerator; print(sys.executable, sglang.__file__)'
```

An explicitly supplied `PYTHONPATH` keeps its precedence. If you override it
for a reference script, omit the candidate source deliberately; use the
interpreter reported by `sys.executable` for a fully isolated reference.

## Syntax helper

`python /app/public/check_candidate.py` compiles Python source beneath
`/app/repo/python/sglang/multimodal_gen/` without importing or executing it.
A syntax pass does not establish registration, loading, image correctness or
serving behavior. Exercise these through the public API.
