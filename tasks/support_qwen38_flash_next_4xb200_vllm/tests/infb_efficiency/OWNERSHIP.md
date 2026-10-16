# `tests/infb_efficiency/`

This task owns its whole-serving efficiency measurement. `client.py` draws the workload and
measures a server; `reference.py` serves and measures the reference before any candidate code
runs; `snapshot.py` builds the reference tree into the verifier image; `nodes.py` holds the two
scored axes.

The measurement is a paired comparison against the reference at the floor
`infb.efficiency.rho` in `tests/config.json` (0.8), on two axes: per-user decode speed at
concurrency 1 and output throughput at concurrency 32, with 4,096 input and 512 output tokens,
greedy with `ignore_eos`, and each request under its own cache salt. The same client measures
the ordinary and the MTP3 recipe. The reference keeps the median of three draws per axis; the
candidate keeps the median of three throughput draws and the best of up to three
single-request attempts, stopping at the first that meets the floor.

`eval/qwen38/measurement/efficiency.py` is the same file this task ships to the agent, byte for
byte, so a
public run and a graded run draw the same workload by the same rules. The hidden document pool
(360 documents) and the draw's seed are only under `tests/`; the agent's sample is 340
documents from the same families, disjoint from the pool by work.

Private prompts, reference construction and calibration stay in the verifier.
