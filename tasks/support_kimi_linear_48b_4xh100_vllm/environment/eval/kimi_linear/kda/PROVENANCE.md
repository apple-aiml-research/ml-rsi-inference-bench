# Mathematical source provenance

`recurrence.py` contains only `naive_recurrent_kda`, unchanged except its file header/imports,
from flash-linear-attention commit `2e0f6c31f9db50f010cb157c712c048b8a3a89d2`,
[fla/ops/kda/naive.py](https://github.com/fla-org/flash-linear-attention/blob/2e0f6c31f9db50f010cb157c712c048b8a3a89d2/fla/ops/kda/naive.py).
The MIT license is in `LICENSE`. The chunked implementation from that file is intentionally absent.
`mathematics.py` supplies independent PyTorch equations for normalization, the released gate,
sigmoid-gated RMSNorm, causal convolution and the enclosing KDA layer; it imports no engine code.
The historical author could reuse optimized KDA code. This task supplies the mathematics and
ordinary generic primitives while leaving an efficient KDA implementation to the solver.
