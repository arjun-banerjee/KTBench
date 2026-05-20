# Fast Walsh-Hadamard Transform: A100 CUDA → H100 Triton

## Description
Computes the **normalised Walsh-Hadamard Transform** (WHT):

    y = (1/√N) · H_N · x

where `H_N` is the N×N Walsh-Hadamard matrix defined recursively:
```
H_1 = [1],    H_{2k} = (1/√2) [[H_k,  H_k],
                                 [H_k, -H_k]]
```

Tensor shapes: `x [B, N]` fp16 → `y [B, N]` fp16, N must be a power of 2.

Used in QuIP# and SpinQuant for rotating LLM weight matrices to reduce
quantisation error; also in randomised smoothing and Hyena-H3 models.
Adapted from HadaCore (pytorch-labs/applied-ai, Apache 2.0, 2024).

## Translation Challenges

- **Warp shuffle → Triton XOR butterfly**: The A100 source uses
  `__shfl_xor_sync(0xffffffff, v, s)` to exchange values between lane pairs
  inside each warp.  In Triton there is no warp-shuffle primitive; the
  equivalent is `tl.load` with XOR-permuted index offsets:
  `partner_offs = offs ^ stride`, then load the partner values and compute
  `(v ± u) * 0.707...`.  The key insight is that the XOR pattern maps
  directly to the butterfly structure.

- **Shared-memory transpose → implicit in Triton**: The CUDA kernel uses
  explicit `smem[lane * nWarps + warpId] = v; __syncthreads()` transposes
  between inter-warp stages.  In Triton the data rearrangement is expressed
  purely as index arithmetic (XOR offsets at each stage), so no `tl.atomic`
  or explicit barriers are needed.

- **Static loop unrolling**: The CUDA source uses a runtime loop over
  `nStages = log2(nWarps)`.  In Triton the loop must be annotated with
  `tl.static_range(LOG2_N)` using a `constexpr` argument, otherwise the
  compiler cannot unroll it and performance collapses.

- **BLOCK_N as constexpr**: `BLOCK_N = N` must be a compile-time constant.
  Pass it as a kernel argument with `tl.constexpr` type annotation.

## Known Gotchas

- The normalisation factor per stage is `1/√2 ≈ 0.70710678`.  Accumulating
  `log2(N)` stages gives the overall factor `(1/√2)^{log2 N} = 1/√N`, which
  is correct for the normalised WHT.  Using `0.5` instead produces the
  unnormalised Hadamard (wrong by a factor of `√N`).

- The partner index `offs ^ stride` wraps correctly for power-of-2 N without
  any explicit modular arithmetic.

- For N > 4096 a tiled multi-program strategy is required; the current
  reference target handles up to N=4096 in one program.

## Provenance
Adapted from HadaCore (pytorch-labs/applied-ai, Apache 2.0, 2024):
https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/inference/hadamard_transform
