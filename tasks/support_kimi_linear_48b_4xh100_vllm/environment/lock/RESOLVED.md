# The dependency resolution, and every decision in it

Resolved **once**, on 2026-09-23, inside the pinned CUDA base image on a development machine with
eight H100 80GB GPUs, by `resolve.sh` in this directory. The recipe installs the two
locks with `--require-hashes --no-deps`, so what the image contains is this resolution and nothing an
index decided later.

| | |
|---|---|
| resolver | `uv 0.9.28` |
| resolved inside | `docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c` |
| Python | 3.12 (`python3.12-full` from Ubuntu 24.04, 3.12.3) |
| index state | `--exclude-newer 2025-10-31T00:00:00Z` — one day after the base commit's date |
| indexes | `pypi.org/simple`; `download.pytorch.org/whl/cu129`; `flashinfer.ai/whl`; `--index-strategy unsafe-best-match` |
| inputs | `requirements/cuda.txt` (which includes `common.txt`) and `requirements/build.txt` at the base commit, plus `pytest`, `tblib` and the three decisions below |

| lock | packages | bytes | sha256 |
|---|---|---|---|
| `requirements.lock` | 152 | 303,805 | `97ad7df36aa045b0d01351148489deb1023d6b9f44a4722e0b66077fadf7b8d1` |
| `requirements-flashinfer.lock` | 2 | 3,477 | `72bb82a88969a7ef416d957ac666751130bc6f777617786c993081947cd5ce65` |

Selected pins, for the record: `torch 2.9.0+cu129`, `torchvision 0.24.0+cu129`,
`torchaudio 2.9.0+cu129`, `transformers 4.57.1`, `tokenizers 0.22.1`, `huggingface-hub 0.36.0`,
`triton 3.5.0`, `numba 0.61.2`, `ray 2.51.0`, `safetensors 0.6.2`, `numpy 2.2.6`, `setuptools 79.0.1`,
`compressed-tensors 0.12.2`, `xgrammar 0.1.25`, `aiohttp 3.13.2`, `pytest 8.4.2`, `tblib 3.2.0`.

`aiohttp 3.13.2` is worth naming: the accuracy evaluator passes no connector, so **aiohttp's default
`TCPConnector(limit=100)` is what fixes the accuracy run's concurrency at 100**, and that is the
number the efficiency load C follows from.

## D-LOCK-1 — flashinfer 0.4.1 is resolved in a second, dependency-free pass

`requirements/cuda.txt` at the base pins `flashinfer-python==0.4.1`. That release declares
`apache-tvm-ffi==0.1.0b15`, and **that pre-release no longer exists on PyPI**: the earliest surviving
release of the package is the stable `0.1.0` of 2025-10-21 (PyPI JSON API, read 2026-09-23). So the
engine's own declared dependency set is **unsatisfiable by any resolver today**, at any `--prerelease`
setting — measured here as `No solution found … there is no version of apache-tvm-ffi==0.1.0b15`.

PyPI also carries **only an sdist** for flashinfer 0.4.1; the wheel
(`flashinfer_python-0.4.1-py3-none-any.whl`) exists on `flashinfer.ai/whl`, which is where upstream's
own Dockerfile gets it. It is pure Python — flashinfer 0.4.x JIT-compiles its kernels at first use
rather than shipping binaries — so there is no H100-versus-B200 question about the artifact itself.

**What is done:** the flashinfer line is stripped from the engine input, and pass 2 pins
`flashinfer-python==0.4.1` together with the surviving `apache-tvm-ffi==0.1.0` under `--no-deps`, so
the withdrawn requirement is never consulted. Both lines are hash-generated from the indexes; neither
is a hand-written hash.

**Why the pin and not a removal.** The alternative — deleting that one line from
`requirements/cuda.txt` — is held in reserve for the case where no offline wheel exists.
Keeping the pin means **the agent's base tree carries upstream's own requirements**, with no
undisclosed edit to them; the removal would have made the supplied base a modified tree. The cost is
that the installed `apache-tvm-ffi` is `0.1.0`
rather than the withdrawn `0.1.0b15`.

**The ABI assumption, stated so a future failure is diagnosed in a minute.** `0.1.0b15` was
**withdrawn**, not yanked with a declared successor, so `apache-tvm-ffi 0.1.0` is **not guaranteed to be
a drop-in** for it: nothing in either project's metadata says the ABI flashinfer 0.4.1 compiles against
is the one 0.1.0 exposes. What stands behind the pairing is evidence, not a guarantee — it imports at
image build and again on four real devices (below). If a future rebuild ever fails inside
`apache_tvm_ffi`, that is the first thing to suspect, and the fix is either the wheel's own pinned
dependency restored from an archive or the one-line removal described above.

**This is a property of the 0.4.x line, not of flashinfer.** A later pin has no such problem and needs
no wheel URL: `flashinfer-python 0.6.18`, for instance, resolves cleanly, because
`apache-tvm-ffi==0.1.0b15` never enters its graph at all.

**What is still a measurement, not a claim.** Whether flashinfer 0.4.1 imports and serves against
`apache-tvm-ffi 0.1.0` is checked twice — in the image smoke on four devices, and again when the
reference implementation serves at the recipe. This engine was measured to import and serve
**without** flashinfer on H100 at all, so it is not on the recipe's critical path. **If it breaks
either check, the fallback is the one-line removal above**, and the finding is
recorded rather than worked around.

## D-LOCK-2 — transformers is pinned at 4.57.1, not merely bounded

`requirements/common.txt` asks for `transformers >= 4.56.0` with no upper bound. Two things depend on
the exact line:

1. the engine reads tokenizer internals (`all_special_tokens_extended`) that the 5.x line removed —
   measured, not assumed;
2. **no released transformers through v5.2.0 carries a native `models/kimi_linear`, but the
   development head does.** Checked by the GitHub contents API on 2026-09-23: `v4.57.1`, `v4.58.0`,
   `v4.59.0`, `v5.0.0` and `v5.2.0` all answer 404 for
   `src/transformers/models/kimi_linear`; `main` answers 200. A resolution that ever reached such a
   release would put a working implementation of the very model this task asks for on an interpreter's
   path.

The `--exclude-newer` cutoff alone would have produced a 4.x release, but a cutoff is not a guard: the
explicit pin and `environment/checks/no_reference_impl.py` are the two halves, one stopping the
resolution and one failing the build if anything reintroduces it.

## D-LOCK-3 — the torch triple comes in as direct, hashed wheel URLs

Two measurements taken here forced this:

- With only upstream's `torch==2.9.0` to go on, uv **mixed the channels**: PyPI's plain `torch 2.9.0`
  and `torchvision 0.24.0`, which bundle the **cu128** runtime, beside the cu129 channel's
  `torchaudio 2.9.0+cu129`. This image's toolkit is CUDA 12.9.1 and the engine's extensions link
  torch's bundled runtime, so a cu128 torch here is a mismatch that surfaces at run time, not at
  build time.
- Pinning `torch==2.9.0+cu129` instead **fails under `--exclude-newer`**: that index serves no upload
  dates, so the cutoff drops its files and uv reports `there is no version of torch==2.9.0+cu129`.
  Without the cutoff the same pin resolves in 2 ms. Isolated with a three-way probe (with cutoff,
  without cutoff, cu129-only index).

A direct URL requirement is exempt from the cutoff — the pin already fixes the artifact, which is what
the cutoff exists to do — `--generate-hashes` still hashes it, and `torch==2.9.0` from
`requirements/cuda.txt` is satisfied by `2.9.0+cu129` under PEP 440, so nothing else in the input
changes. All three cp312 `+cu129` wheels were confirmed present on the channel before they were named.

## What does not enter the image

Only the two `.lock` files are copied into the image. `resolve.sh` and this file stay in the build
context as provenance: a resolver script is a natural place for a reference commit or a PR number to
hide, and the answer-address rule keeps both off the agent's surface.
