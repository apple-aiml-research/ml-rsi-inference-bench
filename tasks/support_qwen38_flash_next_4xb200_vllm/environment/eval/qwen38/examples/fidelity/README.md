# Fidelity examples

`examples.json` holds five sequences (prose 512, code 1,024, math 4,096, a structured interior
window of 12,288 and prose 18,432 tokens) as token IDs with the model author's teacher-forced log
probabilities of every next token, recorded independently with the Transformers implementation in
`../../reference/model_author/`; `provenance` in the file records how. Run the protocol against a
server at either recipe:

```sh
python /app/eval/qwen38/run.py fidelity --base-url http://127.0.0.1:8000 --output /tmp/fidelity.json
```

The client requests every prompt token's own probability (`prompt_logprobs`, which recomputes the
prompt and never reads the prefix cache), validates each record structurally, and reports per
sequence the mean over scored positions of k3 = expm1(r) − r with r the reference minus the served
log probability, and length-band medians. Grading evaluates 40 private sequences of 512–18,432
tokens the same way under calibrated per-band limits with one bounded reread; these examples are
representative, and their success is not an acceptance result.
