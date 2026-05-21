# BF16 GEMM Warp-Level Tiling A100 → H100 WGMMA+TMA multi-warpgroup

## Operation
C = A @ B, BF16 inputs and output.

- A: `[M, K]` bfloat16, row-major
- B: `[K, N]` bfloat16, row-major
- C: `[M, N]` bfloat16, row-major

## Source kernels (verbatim, unmodified)

### A100 (nvidia_a100_sxm)
- Repository: https://github.com/HamzaElshafie/h100_gemm/blob/main/src/kernels/general/gemm_warptiling_bf16.cuh
- Strategy: A100: HamzaElshafie/h100_gemm general/gemm_warptiling_bf16.cuh (Apache 2.0)

### H100 (nvidia_h100_sxm)
- Repository: https://github.com/HamzaElshafie/h100_gemm/blob/main/src/kernels/hopper/gemm_bf16_wgmma_tma_shapes.cuh
- Strategy: H100: HamzaElshafie/h100_gemm hopper/gemm_bf16_wgmma_tma_shapes.cuh (Apache 2.0)

## Key hardware differences

| Feature         | A100 (sm_80)                        | H100 (sm_90a)                          |
|-----------------|-------------------------------------|----------------------------------------|
| Tensor cores    | `mma.sync.aligned.m16n8k16` (warp)  | `wgmma.mma_async` (warp-group, 128 T) |
| Memory copy     | `cp.async.cg` / sync loads          | TMA (`cp.async.bulk.tensor`)           |
| SMEM barrier    | `cp.async.wait_group` + `__syncthreads` | `cuda::barrier<block_scope>` + arrive/wait |
| SMEM cap        | 164 KB shared                       | 228 KB shared                          |
| Producer/consumer | single warp role             | explicit warp-group specialization     |

## Tolerance
atol=0.05, rtol=0.05 (BF16 accumulated rounding).
