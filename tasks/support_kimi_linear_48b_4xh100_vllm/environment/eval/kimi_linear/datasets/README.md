# Offline datasets

`gsm8k/` holds the GSM8K splits from `openai/grade-school-math`, fetched at image build from their
public source and pinned by sha256 there. It is a tracked link onto the image's copy,
`/opt/infb/datasets/gsm8k/`:

| file | rows | bytes | sha256 |
|---|---|---|---|
| `gsm8k/train.jsonl` | 7,473 | 4,166,206 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| `gsm8k/test.jsonl` | 1,319 | 749,738 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

They are named exactly as the evaluator under `../tests/upstream/` caches them, so
`cp gsm8k/*.jsonl /tmp/` is the whole of running it without a network, and
`python ../tests/acceptance.py --mode accuracy` does it for you. See `../tests/upstream/PROVENANCE.md`.
