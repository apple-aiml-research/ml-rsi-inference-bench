# Router semantics

This note is authoritative for the routing of every MoE layer of
`moonshotai/Kimi-Linear-48B-A3B-Instruct` where the model author's source differs.

Every decoder layer after the first (`first_k_dense_replace` 1) is an MoE layer with 256 routed
experts (`num_experts`), one shared expert (`num_shared_experts`) and a correction bias
`e_score_correction_bias` of one value per routed expert. The checkpoint's configuration names
one expert group (`num_expert_group` 1, `topk_group` 1), so the group stage of grouped top-k
keeps every expert. For each token, with router logits `z` over the routed experts:

1. `s = sigmoid(z)` (`moe_router_activation_func`), the routing scores.
2. `b = s + e_score_correction_bias`, the biased scores, used only to choose.
3. The 8 experts (`num_experts_per_token`) with the largest biased scores `b` are chosen.
4. Each chosen expert's weight is its **unbiased** score `s`, renormalized over the chosen eight
   so the weights sum to one (`moe_renormalize`).
5. The routed experts' weighted output is multiplied by `routed_scaling_factor` (2.446) and the
   shared expert's output is added to it.

The correction bias therefore selects experts and does not enter their weights. This is the
engine's grouped top-k convention for a correction bias (`grouped_topk` in
`vllm/model_executor/layers/fused_moe/fused_moe.py` at this tree).

`modeling_kimi.py` differs at step 4: its in-place addition leaves the bias in the scores it
then gathers, so its weights carry the bias too. Where the two differ, follow this note.
