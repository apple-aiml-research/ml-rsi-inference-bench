# This task's efficiency verifier: what it is

`tests/infb_efficiency/` is this task's own efficiency verifier. It measures the candidate against the
human reference implementation on the same GPUs, in the same verifier container, within the same bounds.
Nothing in building, running or grading the task depends on code outside this directory.

**What it holds.**

| file | what it does |
|---|---|
| `client.py` | the measurement client: the timed requests, the per-prompt speeds in exact rational arithmetic, the record and its summary |
| `reference.py` | the reference phase — serve and measure the reference before any candidate code runs |
| `snapshot.py` | builds the reference tree into the verifier image at image-build time, from the pristine task image |
| `nodes.py` | the per-axis measurement nodes the hidden deployment classes mix in, and the config they require |
| `_paired_helper.py` | the teardown, IPC reaping and GPU-free waits the reference phase and the group runner call |
| `deadline.py` | the group runner: per-node alarms, the junit and results streams, reaping |
| `bounds.py` | composes every bound in `config.json` from the verification DAG |
| `canary.py` | the fixed GPU and CPU workloads read between draws, which decide only whether a draw is re-drawn |
| `pool.jsonl.xz`, `pool.provenance.json` | the prompt pool the workload draws from, and where it came from |
| `efficiency.schema.json`, `schema_check.py` | the record's schema, and the check that the record satisfies it |

**The contract it measures** (all of it from `config.json`, `infb.efficiency`): engine `sglang`; the
reference served and measured over `reference_draws` 3; `decode_c1` drawn `decode_c1_draws` times a side
with `decode_c1_statistic` `median_of_ratios`, the median draw deciding; throughput drawn
`candidate_throughput_draws` 3 times; the acceptance ratio `rho` 0.8 unless an axis carries its own bar;
`flush_between_draws` true, so the serving engine's prefix cache is flushed before every draw; and the
canary on at `tolerance` 0.1.

`c1_attempts` is retired: a config that still names it is refused (`nodes.py`). This task sets neither
`infb.efficiency.relaunch` nor `infb.verifier_clock`, and the paths that would serve them are not carried.

**One identity worth stating.** `client.py` here is byte-identical to the copy the agent is given
under `/app/eval/glm53_flash/serving/`, so the measurement the agent can read and run is the
measurement that grades it.
