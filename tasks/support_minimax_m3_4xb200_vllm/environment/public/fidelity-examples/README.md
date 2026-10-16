# Public fidelity examples

`examples.json` holds five teacher-forced sequences of fixed public text with the model author's own-token
log-probabilities:

| id | family | tokens | source |
|---|---|---:|---|
| `prose-1024` | prose | 1,024 | Project Gutenberg ebook 1342 (*Pride and Prejudice*) |
| `code-4096` | code | 4,096 | CPython v3.12.0 `Tools/`, every `*.py` file in sorted path order, each preceded by `# file: <path>` |
| `dialogue-4096` | dialogue | 4,096 | Project Gutenberg ebook 2542 (*A Doll's House*) |
| `records-16384` | records | 16,384 | GSM8K train rows 0–999 as their JSON lines |
| `prose-65536` | prose | 65,536 | Project Gutenberg ebook 1342 |

Each is tokenized with `/model/tokenizer.json` without special tokens (text as UTF-8, line endings normalized to LF);
windows start at token 10,000 of their source, 1,000 tokens apart. `reference_logprobs[i]` is the log-probability of
`token_ids[i + 1]` given `token_ids[: i + 1]`, recorded once with the model-author implementation in Transformers
v5.13.0 (the first release whose index selection is per key-value group): the checkpoint's MXFP8 weights dequantized
exactly to BF16 at load, eager attention, BF16, teacher-forced in 1,024-token query chunks through the model's cache
with explicit position ids, FP32 log-softmax, rounded to six decimals. The engine quantizes activations to MXFP8
dynamically and the recording does not.

```sh
python /app/eval/minimax_m3/run.py fidelity --base-url http://127.0.0.1:8000 \
    --inputs /public/fidelity-examples/examples.json --output /tmp/f.json
```

The command reports per-sequence mean k3 (`expm1(ref − served) − (ref − served)` averaged over positions), the same
mean with the sequence's single largest position removed, and band medians. These examples explain the protocol; grading reads other sequences, and there is no public pass threshold.
