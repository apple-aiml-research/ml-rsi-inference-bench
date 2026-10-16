# Request-state lifecycle

Kimi Linear keeps per-request model state: for each of the 20 KDA layers, three short-convolution windows
(q, k, v) and one recurrent state. Serving must give every request the state of its own complete token
history, through its whole lifetime, in every deployment of `../launch.py`. This directory states how that
is checked and gives a small client for the representative examples.

    policy.json   the reading rules, the continuation statistic and its tolerance
    client.py     `sample` (the representative examples) and `compare` (the continuation comparison)

## Deployments

`../launch.py --deployment <name>` starts one of four deployments; `--print` shows the command.

| Deployment | Devices | Serving |
|---|---|---|
| `tp4` | the first four | the recipe: TP4/BF16, prefix caching off |
| `dp2` | all eight | two TP4 data-parallel replicas, prefix caching off |
| `apc` | the first four | prefix caching on, 512-token state checkpoints (`--mamba-block-size 512`) |
| `apc-pressure` | the first four | `apc` at a 16,384-token context with a 64-block cache, at most 16 running requests and 8,192 batched tokens per step: the capacity-limited variant, where cached prefixes are evicted and running requests are preempted and resumed |

Each deployment uses chunked prefill, compiled execution with full CUDA graphs for ordinary decode, and
FlashMLA, and must serve requests on all of its devices; in `dp2`, both replicas do real work and ordinary
decode uses full graphs on both, including steps where one replica has fewer requests than the other or
none.

## What is required

* **Fresh state.** A request with no history starts from zero state wherever its state storage lives and
  whatever earlier requests left there, including one-token prompts and fresh requests scheduled beside
  requests that are decoding.
* **Continued state.** A request with history continues exactly that history: across prefill chunks,
  decode steps, mixed prefill/decode batches and graph padding. Clearing all state every step is not a
  repair.
* **Data parallelism.** The same per-request behaviour on both replicas under uneven and idle load.
* **Prefix reuse (`apc`, `apc-pressure`).** A completed common prefix -- including one that ends inside
  previously generated tokens -- is restored for later requests and branches instead of recomputed, and
  serving then continues exactly as if the complete history had been processed independently. A prefix
  state may be reused only once it has been completely computed; reusing a state still being produced in
  the same step is not allowed. Branching from a prefix never changes the retained prefix state or another
  branch. When a reusable checkpoint is unavailable (evicted, never produced at that boundary), recomputing
  is valid; recomputing every repeated prefix is not prefix-cache support.
* **Pressure.** Under the capacity-limited variant, evicted prefixes, preempted and resumed requests and
  cancelled requests keep every other request's state intact; a resumed request keeps its generated
  history; a cancelled request is terminated and its storage released.

Evaluation reads the engine's own execution: which requests were scheduled, their state storage, the
state a restored request actually consumes at the KDA operation, graph modes and replica activity. An
enabled flag, an aggregate counter or a response alone is not evidence of reuse. The engine's
`vllm:prefix_cache_hits` counter is reported beside the observed consumption.

## Continuation equivalence

Every scored request is generated greedily (`temperature` 0, `ignore_eos`, `logprobs` 1, returned token
ids; caching requests omit `prompt_logprobs`). Its prompt and exact returned continuation are then
teacher-forced on a cold `tp4` server, one request at a time, and the two log-probabilities of each
continuation token are compared by the k3 statistic of `../fidelity/kl.py` (reference minus candidate,
Python double precision):

    k3(window) = mean over the window's positions of (exp(r) - 1) - r,   r = reference - candidate

over the **whole continuation** and over **every contiguous window of 4 and of 16 generated positions** (a
continuation shorter than a window has one window, its whole length). A request passes when every one of
these values is finite and at most `T_lifecycle` (`policy.json`). The windows keep a short disagreement
after a later state transition -- a restore, a branch, a resumption -- from being averaged away by the rest
of the continuation. Comparing the probability of the same returned tokens, rather than requiring identical
free-running tokens, keeps legitimate near-tie differences between deployments from counting as errors.

A response is read strictly: exactly the requested number of tokens with finish reason `length`, integer
ids and one finite log-probability per token; teacher-forced values by `../fidelity/check.py`'s reader.
Missing, duplicate, non-finite or shifted values are failures. A deliberately cancelled request needs at
least one returned token, and every returned token is compared.

`T_lifecycle` is `max(0.05, 3 * N_reference)`, where `N_reference` is the largest full or window value
measured for the qualified reference implementation against an independently restarted cold reference.
The reference qualification froze the value, and `policy.json` publishes it. Evaluation also
teacher-forces each scored continuation on the deployment under test itself (prompt log-probabilities
requested, so no prefix is reused) and records the same statistic beside the judged one, as supplemental
consistency evidence: it is never judged and never replaces the comparison with the independent reference.

## Workload scope

Prompts of 1 to 12,000 tokens and continuations up to 1,024 tokens, including: fresh one-token and short
prompts after the state storage has been turned over by earlier requests; fresh requests submitted while
others decode; continued requests; uneven bursts and overlapping waves across both `dp2` replicas; warm
prefixes and siblings at, just below and just above 512-token boundaries and at multiples of them; several
branches from one prefix, advanced in interleaved order; prefixes made of a prompt and part of its generated
continuation; long prompts whose prefill spans several chunks, with siblings at the reached boundaries;
mixed waves of cached and fresh requests; and, under `apc-pressure`, prefix turnover, generation across
the first state-block boundary with preemption and resumption, reuse of a resumed request's generated
history, and cancellation followed by fresh requests and branches. These are declared workload families,
not every combination of context, concurrency and deployment.

## The examples

    python client.py sample --deployment tp4 --url http://127.0.0.1:8000 --out tp4.json

* `tp4`: one ordinary fresh one-token request, and the loaded-layer example of mixed fresh and continued
  rows (`../kda/client.py`'s sample, case `mixed_fresh_continued`, assessed by `../kda/numerical-policy.json`);
* `dp2`: one ordinary concurrent workload of eight requests;
* `apc` and `apc-pressure`: one warm-prefix/sibling pair run alone: a 1,024-token prefix plus one token,
  then the same prefix plus another, sharing one cache salt; the sibling's hit-counter delta is reported
  (1,024 when the prefix is reused), or `unavailable` when the server exports no counter.

An unimplemented feature is reported as unsupported with its error. Then stop the server, start a cold
`tp4` server, and compare:

    python client.py compare --records tp4.json --cold-url http://127.0.0.1:8000 --out tp4-compare.json

`compare` reports every full and window value, the maximum per window size with its start, and the
verdict once `T_lifecycle` is published. Against your own cold server this is **self-consistency**: it
cannot establish agreement with the independent qualified reference that evaluation also uses. The two
servers never need to run at the same time. Evaluation's workloads are held out; passing the examples does
not establish acceptance.

## Mapping a different native layout

Observation finds each loaded KDA layer and the state it consumes through the declarative mapping of
`../kda/README.md` (`layer_adapter.py`): the layer's complete forward entry, its positions argument, and the
`lifecycle` keys -- the state-consumption entry, the metadata paths of its state indices, query offsets
and state validity, and how its output is returned. A layout the defaults do not describe (for example a
fused q/k/v convolution state exposed as three views, a different consumption entry, or different metadata
names) is described there by those keys; evaluation then reads the actual tensors at that boundary.
