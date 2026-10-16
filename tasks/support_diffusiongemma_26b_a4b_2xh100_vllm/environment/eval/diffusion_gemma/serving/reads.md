# Denoising reads

A read asks the server to evaluate the model's denoiser on a canvas that you supply,
without generating. A read is an ordinary request that carries the fields below in
`vllm_xargs`, each as a string. Text prompts use `/v1/completions` with the prompt given as
token IDs; prompts with images use `/v1/chat/completions`.

| Field | Value |
|---|---|
| `diffusion_read_only` | `"1"` |
| `diffusion_seed_canvas` | L token IDs, comma-separated: the canvas input, 1 ≤ L ≤ 256 |
| `diffusion_max_steps` | `"1"` or `"2"` |
| `diffusion_target_token_ids` | L token IDs, comma-separated |

The request also sets:

- `max_tokens` to L;
- `temperature` to 1.0;
- `logprobs` to 20 (chat: `logprobs: true` and `top_logprobs: 20`);
- `return_tokens_as_token_ids: true` and `return_token_ids: true`.

Semantics:

1. The prompt is encoded as for any request; the prefix cache may serve it.
2. One canvas of length L follows the prompt. It is initialized to the seed canvas.
3. The denoiser runs `diffusion_max_steps` steps on the canvas, with every position held
   at its seed token.
   - Step 0 receives a zero self-conditioning input.
   - With two steps, step 1 receives the self-conditioning input that generation passes
     from step 0 to step 1: step 0's logits at step 0's temperature, which is the
     deployment's `t_max`.
4. The response's L tokens are the last step's argmax canvas.
5. Position i's log-probabilities are the log-softmax, in FP32, of the model's final logits
   (after the final soft-capping) at temperature 1. They include the 20 most likely tokens
   and target i.
6. End-of-sequence tokens and stop strings do not end a read.
7. Nothing of the canvas is committed to the sequence or cached for later requests.
8. A read may run concurrently with other requests and must not change them.

L is 256 except where the context limit leaves fewer positions.

This interface is adapted from the upstream engine's later read mode for this model; the
two-step form is part of this benchmark.

`python /app/eval/diffusion_gemma/run.py reads --base-url http://127.0.0.1:8000 --inputs
/app/eval/diffusion_gemma/examples/reads.json --output /tmp/reads.json` sends the public
examples. It reports each read's structure and its per-position agreement with the
model-author values recorded in the examples.
