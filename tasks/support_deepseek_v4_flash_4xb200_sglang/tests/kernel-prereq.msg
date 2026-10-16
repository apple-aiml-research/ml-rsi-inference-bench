Supply the DeepSeek-V4 kernel prerequisites, attention and compression withheld

The JIT kernel paths that upstream landed together with this model's integration,
applied on this base except six completed sources: 28 of the 34 paths (23 added,
5 modified, +6070/-147) and a README naming what is withheld. The C4/C128 compression,
fused post-compression norm/RoPE and fused cache-store sources (c4.cuh, c128.cuh,
c128_v2.cuh, c128_online.cuh, fused_norm_rope.cuh, store.cuh) are not supplied;
their Python entry points in deepseek_v4.py remain.

FlashMLA 9241ae3 is vendored at third_party/flash_mla as an editable part of this
repository, with its optimized sparse attention withheld (csrc/sm100/decode/head64,
csrc/sm100/prefill/sparse, csrc/sm90/decode/sparse_fp8, csrc/sm90/prefill/sparse and
the two sparse API headers); its two sparse entry points are registered as stubs that
raise NotImplementedError, and its setup.py builds offline at a fixed version. The
generic CUTLASS it builds against is at /opt/infb-deps/flashmla-cutlass, outside the
repository. Nothing in the tree calls these kernels yet.
