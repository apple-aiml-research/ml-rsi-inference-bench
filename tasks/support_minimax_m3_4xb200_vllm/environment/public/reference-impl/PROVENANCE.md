# Model-author implementation of MiniMax-M3 (for reading)

`modeling_minimax_m3_vl.py` and `configuration_minimax_m3_vl.py` are taken verbatim from Hugging Face
Transformers at tag `v5.13.0`, the first release whose sparse-attention index selection is per key-value group:

- `src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`
  sha256 `fd17ebfc48e8597607c017674053338029871fe9c924042d64cc8f92fa3b2584`
- `src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py`
  sha256 `7ee031c3220fed242265388bdd8d7d31780ce268ed5244f5923002ebee695017`

The engine's Transformers is 5.5.3 and does not know this model, so these files are here to read, not to import.
