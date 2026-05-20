"""
Source kernel: fused RMSNorm + residual add, optimised for A100.
Pattern from FlashAttention / Apex fused layer-norm kernels (2022-2024).

Inputs:  x [B, L, D]  fp16,  r [B, L, D]  fp16,  w [D]  fp16
Output:  y [B, L, D]  fp16
Formula: y = ((x + r) / rms(x + r)) * w
         rms(v) = sqrt(mean(v^2) + eps)

A100 strategy
─────────────
Grid: (B*L,) — one warp-group (4 warps = 128 threads) per token position.
Each block normalises one row of D elements.

Two-pass warp reduction using __shfl_down_sync:
  Pass 1: each thread accumulates sum-of-squares over its D/128 elements.
          Four rounds of warp-level reduction collapse to a per-block scalar.
  Pass 2: threads load the same elements again, normalise, scale, and store.

The shared-memory intermediate (partial sums per warp) is the standard
warp-reduce pattern tuned for A100's 4 MB L2 and 32-thread warp size.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <math.h>

static const int kNT  = 128;  // threads per block (4 warps)
static const int kWarp = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ __launch_bounds__(kNT)
void rms_norm_residual_a100(
    const __half* __restrict__ x,
    const __half* __restrict__ r,
    const __half* __restrict__ w,
    __half*       __restrict__ y,
    int D, float eps)
{
    // Each block handles one (b, l) token: row index = blockIdx.x
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* xrow = x + row * D;
    const __half* rrow = r + row * D;
    __half*       yrow = y + row * D;

    // Pass 1: accumulate sum-of-squares of (x + r) over D.
    float partial = 0.f;
    for (int i = tid; i < D; i += kNT) {
        float v = __half2float(xrow[i]) + __half2float(rrow[i]);
        partial += v * v;
    }

    // Warp reduce, then cross-warp reduce via shared memory.
    __shared__ float warp_sums[kNT / kWarp];  // 4 entries
    partial = warp_reduce_sum(partial);
    if (tid % kWarp == 0)
        warp_sums[tid / kWarp] = partial;
    __syncthreads();

    float total_sq = 0.f;
    if (tid < kNT / kWarp) {
        total_sq = warp_sums[tid];
    }
    total_sq = warp_reduce_sum(total_sq);
    // Broadcast rms inverse to all threads via shared memory.
    __shared__ float rms_inv;
    if (tid == 0)
        rms_inv = rsqrtf(total_sq / D + eps);
    __syncthreads();

    // Pass 2: normalise, scale, store.
    float inv = rms_inv;
    for (int i = tid; i < D; i += kNT) {
        float v   = __half2float(xrow[i]) + __half2float(rrow[i]);
        float wi  = __half2float(w[i]);
        yrow[i]   = __float2half(v * inv * wi);
    }
}

void rms_norm_residual_fwd(
    torch::Tensor x, torch::Tensor r,
    torch::Tensor w, torch::Tensor y,
    float eps)
{
    int B = x.size(0), L = x.size(1), D = x.size(2);
    int rows = B * L;
    rms_norm_residual_a100<<<rows, kNT>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(r.data_ptr()),
        reinterpret_cast<const __half*>(w.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),
        D, eps);
    cudaDeviceSynchronize();
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="rms_norm_residual_src",
            cpp_sources="void rms_norm_residual_fwd(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float);",
            cuda_sources=_CUDA_SRC,
            functions=["rms_norm_residual_fwd"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        _get_ext().rms_norm_residual_fwd(
            x.contiguous(), r.contiguous(), w.contiguous(), y, 1e-6
        )
        return y
