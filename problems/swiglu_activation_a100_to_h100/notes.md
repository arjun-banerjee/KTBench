# SwiGLU Activation - CUDA A100 → CUDA H100

**Op:** `silu(gate) * up`, where the input is `x = concat(gate, up)` along the last dim.

* Input  `x: [N, 2*D]` (fp16 / bf16)
* Output `y: [N, D]`

## Origin

- I/O contract from `KernelBench/level5/hardware_translation/io/04_swiglu_activation.toml`
  (`NUM_TOKENS=512`, `D=2048`).
- Numerical contract from the matching `.../oracle/04_swiglu_activation.py`
  (a vanilla `F.silu(gate) * up`).
- The vLLM reference kernels in
  `KernelBench/level5/kernels/8xa100/04_swiglu_activation.cu` and
  `.../8xh100/04_swiglu_activation.cu` are byte-identical; the kernels
  shipped in `source.py` and `reference_tgt.py` are simpler standalone
  implementations that compile via `torch.utils.cpp_extension.load_inline`
  without pulling in the vLLM private headers (`cuda_compat.h`,
  `dispatch_utils.h`, `cuda_vec_utils.cuh`).

## Translation hooks

A candidate aiming to clear a meaningful SOL fraction on H100 has plenty
of room - the source kernel is intentionally naive (one thread per
element, scalar fp16 → fp32 → fp16). Useful Hopper-leaning rewrites:

- Vectorised loads/stores (`half2`, `__nv_bfloat162`, `int4`) - this op
  is bandwidth-bound.
- Coalesced access patterns with one block per row + one thread per
  vector lane.
- Async copy (`cp.async` / TMA) to overlap DRAM loads with the SiLU
  arithmetic.
- A reciprocal-sigmoid identity (`silu(x) = x * sigmoid(x)`) using
  `__expf` and FMA-friendly arithmetic.

The output buffer is zeroed before the kernel call to defeat the
`torch.empty()` stale-memory exploit.
