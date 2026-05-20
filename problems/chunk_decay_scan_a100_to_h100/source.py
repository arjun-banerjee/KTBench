"""
Source kernel: exponential-decay cumulative scan, optimised for A100.
Inner loop of Gated Linear Attention (Yang et al. 2023) and RetNet.

Inputs:  x [B, H, L, D]  fp16   log_decay [H, D]  fp32
Output:  y [B, H, L, D]  fp16

Formula: y[b,h,t,d] = sum_{s=0}^{t} x[b,h,s,d] · exp((t-s)·log_decay[h,d])
Recurrence: state[d] = decay[d] · state[d] + x[t,d]

A100 strategy
─────────────
Grid: (B*H, D/kD) — one block per (batch, head) pair and D-dimension tile.
Threads: kNT=128 per block.  Each thread owns a contiguous kD-wide slice of D.

The scan runs sequentially over L inside each block, accumulating a running
fp32 state vector.  All state lives in registers — no shared memory is needed.
The sequential loop is the only correct approach for a data-dependent scan;
the A100 kernel exploits register residency and instruction pipelining to keep
the GPU busy across the L iterations.

Thread mapping: thread tid within a block handles elements
  [tile_start + tid, tile_start + tid + kD, …] across the D dimension (stride
  = kNT, so each thread handles D/(kNT) elements when D is large, or fewer
  elements when the tile is small).  For clarity, here each thread handles a
  single D element (kD=1 effective).
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <math.h>

static const int kNT = 128;

__global__ __launch_bounds__(kNT)
void chunk_decay_scan_a100(
    const __half* __restrict__ x,      // [B, H, L, D]
    const float*  __restrict__ log_dec,// [H, D]
    __half*       __restrict__ y,      // [B, H, L, D]
    int B, int H, int L, int D)
{
    // Each block handles one (b, h) pair and a contiguous D-tile.
    const int bh  = blockIdx.x;   // in [0, B*H)
    const int b   = bh / H;
    const int h   = bh % H;
    const int tid = threadIdx.x;  // in [0, kNT)

    // Stride over D dimension: each thread handles elements at tid, tid+kNT, ...
    // Preload per-thread decay value(s) into registers.
    // For simplicity one decay value per thread (assumes D divisible by kNT).
    if (tid >= D) return;  // guard for D < kNT

    float dec = expf(log_dec[h * D + tid]);  // decay per thread element
    float state = 0.f;                        // running fp32 accumulator

    const __half* xbhd = x + (b * H + h) * L * D + tid;
    __half*       ybhd = y + (b * H + h) * L * D + tid;

    // Sequential scan over L — state lives in registers the entire time.
    for (int t = 0; t < L; ++t) {
        float xt = __half2float(xbhd[t * D]);
        state    = dec * state + xt;
        ybhd[t * D] = __float2half(state);
    }
}

void chunk_decay_scan_fwd(
    torch::Tensor x, torch::Tensor log_decay, torch::Tensor y)
{
    int B = x.size(0), H = x.size(1), L = x.size(2), D = x.size(3);
    dim3 grid(B * H);
    chunk_decay_scan_a100<<<grid, kNT>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        log_decay.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr()),
        B, H, L, D);
    cudaDeviceSynchronize();
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="chunk_decay_scan_src",
            cpp_sources="void chunk_decay_scan_fwd(torch::Tensor, torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["chunk_decay_scan_fwd"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, log_decay: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        _get_ext().chunk_decay_scan_fwd(x.contiguous(), log_decay.contiguous(), y)
        return y
