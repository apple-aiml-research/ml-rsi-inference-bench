# Native component measurement

`run.py components` measures the loaded model's own QSA and PLE implementations through the
engine's runner, on the same GPUs and with the same protocol for the reference and for a submission.
The server needs `--worker-extension-cls qwen38.measurement.worker.MeasurementWorker` and
`VLLM_SERVER_DEV_MODE=1`. The extension only adds RPC methods (`/collective_rpc`); it supplies no
model code, metadata, cache layout, expected values or verdicts.

## Regions and the optional declaration

A *region* is one loaded native module method whose calls compute a complete QSA or PLE layer, or a
contiguous enclosing region around it. By default the QSA regions are `*.layers.*.self_attn`
(method `forward`, hidden-state input `hidden_states`) and the PLE region is `*.layers.*.ple`
(`forward`, first argument), each output its return value, each layer read from the module path
(`\.layers\.(\d+)\.`). An implementation with other paths, methods, argument positions or a fused
boundary commits `/app/qwen38-adapter.json` (`examples/adapter.json` shows the defaults):

```json
{"schema": "qwen38-native-regions/1",
 "regions": [{"component": "qsa", "module": "<glob over model.named_modules()>", "method": "forward",
              "hidden_input": "<argument name or position>", "output": "return" | {"argument": <name or position>},
              "checkpoint_layer": "<regex with one group: the checkpoint layer index>",
              "encloses": ["qsa"]},
             {"component": "ple", ..., "encloses": ["ple"] | ["ple", "ple_residual"]}]}
```

`encloses` names the contiguous checkpoint operations a region computes: `qsa` is the complete
attention layer (q/k/v, indexer projections, norms, rotary embedding, compressed index keys, selection,
sparse attention, output gate and o_proj); `ple` is the complete PLE layer (n-gram embedding,
projections, gating, convolution); `ple_residual` adds the residual addition of the PLE output to the
layer's hidden streams, for an implementation that fuses it. Fusion inside a complete layer needs no
declaration. Reference and submission are measured over the same enclosure: when a submission's
region encloses more than the reference's, the reference's native region is measured with the
missing trailing operation composed onto it (the same composition applies to the comparison). A declaration names native values only — never a case, verdict, replacement
callable or constructed metadata — and default discovery is a convenience, not a naming requirement.
Every region must resolve to engine source under `/app`. `run.py declaration` shows what a server
resolves.

## Capture

For each case the worker acts as the engine's scheduler for a few private requests while the server
is idle: the engine's own `KVCacheManager` allocates blocks from the runner's KV cache
configuration, and every step is a `SchedulerOutput` for the runner's own `execute_model` and
`sample_tokens`, so the implementation builds its own attention metadata and writes its own state.
History is prepared natively, by prefilling it within the scheduler's token budget. Then the
case's measured steps run (prefill: one step; decode and speculative: `steps` steps) and one more
probe step, whose outputs read the state the last measured step wrote. Steps run without compiled
graphs (the engine's own `skip_compiled` flag) so region calls are observable; every call of the
region is recorded with its native arguments, the forward context the runner built (attention
metadata included), its output, and the rows of every cache tensor bound under the region before
and after the call. The region must run exactly once per runner step.

Speculative cases (MTP3) verify four positions per request per step. With `"acceptance": "native"`
the engine's own draft head and verification decide progress; grading's cases instead fix the
accepted draft count of each step (0–3) through the engine's sampling interfaces, an admissible
sampling outcome that is identical for every implementation paired on a case. Rejected positions
are not history. Reference and submission run the same logical requests, initial histories,
active tokens and scheduled progress, and the record carries them per step; physical padding and
cache representation may differ.

A speculative case may name joining prefills (`"joining": [[step, tokens], ...]`): in each named
measured step a fresh request prefills `tokens` prompt tokens in the same runner step as the
speculative requests' verification, a mixed speculative/prefill iteration, and finishes after it.
Every region output of that step, the speculative rows included, is checked like any other. A
joining case is untimed (`"timed": false`): it is captured and checked, but it has no timed
matrices and no allocation evidence.

## Timing

A *draw* restores the rows the measured work starts from, then executes the complete measured
sequence of region calls through their captured native inputs: prefill eagerly (synchronized),
decode and speculative sequences as one CUDA graph over all of their steps. Per-call input
transforms, metadata use and packing, projections, selection, communication, state writes and
synchronization are inside the draw; history preparation, graph capture and state restoration are
outside it. After every draw its outputs and a probe continuation from the state it left are
validated at the numerical allowances (`scope.json`); a failure is terminal for the case, never a
reason to draw again.

A *matrix* times every case once, in the fixed case order, with three warm-ups and nine draws per
case. A collection runs three complete matrices. Statistic, per case and matrix: each TP rank's
median synchronized wall time, then the maximum over the four ranks; per case: the median of its
three matrix statistics. CUDA-event time is recorded beside it. Grading compares the submission's
statistic with the reference's, case by case, from a collection of the reference made earlier on
the same four GPUs; there is no averaging across cases, ranks, components or phases. Allocation
tracing never runs inside timing draws.

## PLE memory

For every PLE case one separate traced draw per rank records the PyTorch allocator's event history
together with CUDA runtime and driver allocations (`cuMemAlloc*`, `cuMemCreate`, `cudaMalloc*` and
their frees, observed through CUPTI) on one aligned timeline. The charged bytes are persistent
auxiliary storage reachable from the region (outside parameters and bound state), bound-state
capacity beyond the logical live state of the case's requests, and the peak of storage allocated
during the draw and still live — retained outputs, workspace, temporaries and external allocations
included. Parameters, the region's mathematical inputs and the logical live state (per request
`C` channels by `H` positions, plus the three speculative positions under MTP3, in the state dtype)
are excluded; storage ranges are deduplicated and an allocation the PyTorch allocator makes for its
own segments counts once. Unused reserved pool capacity is reported, not charged. The allowance
per rank and case is `scope.json`'s `ple_memory` formula; padding is allowed when it fits.

## Use

For the first case of each component the worker also runs the measured steps three times from
identical cache contents: twice unperturbed and once with the region's output perturbed in place.
The next decoder layer's input must repeat within the restoration tolerance between the two
unperturbed runs and leave it in the perturbed run: the region's result is consumed by the model.

## Records

Every request carries the protocol identity (a digest of this package) and a request ID; every TP
rank answers with its rank, the recipe, the candidate's source identity and a status. A missing,
stale or failed rank answer fails its case, and the raw answers are kept. The record
(`qwen38-native-measurement/1`) is rewritten after every answer, carries each case's identity
(owner module, class, method, source digest, bound state), executed steps, captures, three
matrices of draws, memory and use evidence, and is bounded by `budgets.json`
(`component_collection` for the collection, `component_matrix` per matrix); work left when an
allowance is spent is recorded as not run.
