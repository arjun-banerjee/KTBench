"""
Source kernel: normalised Fast Walsh-Hadamard Transform, optimised for A100.
Adapted from HadaCore (pytorch-labs/applied-ai, Apache 2.0, 2024).

Input:  x [B, N]  fp16    N must be a power of 2
Output: y [B, N]  fp16    y = (1/sqrt(N)) * H_N @ x

A100 strategy
─────────────
For N ≤ 32: each warp (32 threads) handles one row entirely in registers.
  log2(N) butterfly stages run via __shfl_xor_sync, keeping all values
  in the warp's register file — no shared memory needed.

For N > 32: the row is split into 32-element segments.  Each warp handles
  one segment and performs 5 butterfly stages in registers.  Then a shared-
  memory transpose permutes the data across segments, and the next set of
  stages continues.  This alternating register/smem pattern repeats until all
  log2(N) stages are done.

This mirrors the HadaCore approach of decomposing the Hadamard recursion into
warp-local stages (cheap: register shuffles) and cross-warp stages (expensive:
smem, but only ⌈log2(N/32)⌉ of them).
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <math.h>

static const float ISQRT2 = 0.7071067811865476f;

// ── 5 warp-level butterfly stages operating on 32 fp32 values in registers ──
__device__ __forceinline__
float warp_had32(float v, int lane) {
    #pragma unroll
    for (int s = 1; s <= 16; s <<= 1) {
        float u = __shfl_xor_sync(0xffffffff, v, s);
        // Even lane in pair: (v + u), odd lane: (v - u), both scaled /sqrt(2)
        v = ((lane & s) == 0) ? (v + u) : (v - u);
        v *= ISQRT2;
    }
    return v;
}

// ── Full transform for arbitrary power-of-2 N ──────────────────────────────
// Grid: (B, N/32) — each warp handles 32 consecutive elements of one row.
// For N ≤ 32 the y-grid dimension is 1 and no smem transpose is needed.
__global__ __launch_bounds__(1024)
void hadamard_fwd_a100(
    const __half* __restrict__ x,
    __half*       __restrict__ y,
    int B, int N)
{
    extern __shared__ float smem[];  // N floats per block (shared across warps)

    const int b     = blockIdx.x;
    const int lane  = threadIdx.x % 32;
    const int warpId= threadIdx.x / 32;
    const int nWarps= blockDim.x / 32;  // = N / 32

    // Load one element per thread into register
    int gidx = warpId * 32 + lane;  // index within this row
    float v = __half2float(x[b * N + gidx]);

    if (N <= 32) {
        // Single warp: all 5 stages in registers.
        v = warp_had32(v, lane);
        y[b * N + gidx] = __float2half(v);
        return;
    }

    // Stage 1: intra-segment butterfly (32 elements per warp)
    v = warp_had32(v, lane);

    // Remaining log2(nWarps) stages via smem transposes.
    // At each stage: write to smem in transposed layout, sync, read back.
    int nStages = 0;
    { int tmp = nWarps; while (tmp > 1) { nStages++; tmp >>= 1; } }

    for (int stage = 0; stage < nStages; ++stage) {
        // Write to smem: row = lane, col = warpId
        smem[lane * nWarps + warpId] = v;
        __syncthreads();
        // Read from smem: row = warpId, col = lane (transposed)
        v = smem[warpId * nWarps + lane];
        __syncthreads();

        // One additional butterfly stage within the new arrangement
        float u = __shfl_xor_sync(0xffffffff, v, 1 << stage);
        v = ((lane & (1 << stage)) == 0) ? (v + u) : (v - u);
        v *= ISQRT2;
    }

    // Write back (transposed back to original layout via smem)
    smem[warpId * 32 + lane] = v;
    __syncthreads();
    y[b * N + gidx] = __float2half(smem[warpId * 32 + lane]);
}

void hadamard_fwd(torch::Tensor x, torch::Tensor y) {
    int B = x.size(0), N = x.size(1);
    int nWarps = N / 32;
    int threads = N;  // one thread per element, max N=1024 per block here
    // For N > 1024 we'd need a multi-block strategy; restrict to N ≤ 1024 here.
    int smem_bytes = N * sizeof(float);

    hadamard_fwd_a100<<<B, threads, smem_bytes>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),
        B, N);
    cudaDeviceSynchronize();
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="hadamard_fwd_src",
            cpp_sources="void hadamard_fwd(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["hadamard_fwd"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) <= 1024, "source kernel supports N ≤ 1024; use tiled variant for larger N"
        y = torch.empty_like(x)
        _get_ext().hadamard_fwd(x.contiguous(), y)
        return y
