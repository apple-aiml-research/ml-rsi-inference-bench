# Model-author implementation (for reading)

The files in this directory are the DiffusionGemma model-author implementation, taken verbatim from
Hugging Face Transformers at tag `v5.13.0`, fetched by sha256 when the image is built:

| File | Source path at `v5.13.0` | sha256 |
|---|---|---|
| `modeling_diffusion_gemma.py` | `src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py` | `3a114c48ccd270577c6caf4f04009442f706aea4d5cb0a48f08beaa0c2b602ae` |
| `generation_diffusion_gemma.py` | `src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py` | `0a4bf0c9fbcaef2cf86d3bb2177eb985bb83568af815a6812a4c8f7a1eeb71de` |
| `configuration_diffusion_gemma.py` | `src/transformers/models/diffusion_gemma/configuration_diffusion_gemma.py` | `69131b618dcdf8e95b4b17fadf40c658c53cd3a4a7f4ae410b26673b7d552ec0` |
| `cache_utils.py` | `src/transformers/cache_utils.py` | `16bbbba2394c0151af53009800e5a4052479417a8c7e5edfbd5ef128a46fcbc1` |
| `masking_utils.py` | `src/transformers/masking_utils.py` | `ad7b312bc3830894a198760520cc1baf02e35d327d216d41f00b0c61e9604334` |

The engine's Transformers installation is 5.5.3, which does not include this model, so these files
are here to read, not to import. They are not part of the Git history of `/app`.
