# Model-author implementation, for reading

These two files are the model author's implementation of Inkling, as released in Hugging Face
Transformers v5.16.1 (`src/transformers/models/inkling/`). They define the model's semantics:
the attention with its relative bias, the short convolutions and their cached state, the expert
routing and the rest of the forward pass.

They are supplied for reading only. They are not installed and cannot be imported: the engine's
Transformers is 5.12.1, which does not include this model.

| File | Released sha256 (v5.16.1) | Supplied sha256 |
|---|---|---|
| `modeling_inkling.py` | `234e9f7e42fdeade27a88f369deb46176db8d37748c7a7a16fb92ec762c7fe7c` (60,265 bytes) | `4afd169e8a0ae652f0da0f696e8b06aae4c287da7d7ca33cf68dee90764026a8` (60,179 bytes) |
| `configuration_inkling.py` | `ec8c7edf135c514103213829701f0151e339ce5f92f556ef57995de064ea0e91` (13,998 bytes) | `20f2c4923d7cbea5293b08aa2f2b0c41601319b528390ed0abaa4eb2fdca6187` (13,957 bytes) |

**Documented edit.** Engine-implementation class names were removed from two docstring sentences
and one comment: the relative-logits module's docstring, the multimodal configuration's
introduction and a comment beside the output projection. No code, configuration default or
docstring statement about the model's mathematics changed. The released files are public.
