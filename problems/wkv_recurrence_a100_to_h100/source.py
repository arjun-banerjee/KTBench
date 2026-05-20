"""
Source kernel: WKV token-mixing recurrence, optimised for A100.
Adapted from BlinkDL/RWKV-LM WKV5 forward kernel (Apache 2.0).

Inputs:  r, k, v [B, H, T, N]  fp16   w [H, N]  fp32   u [H, N]  fp16
Output:  y [B, H, T, N]        fp16

Recurrence:
  state[n] = 0   (per batch, head, feature)
  for t in range(T):
    kv[n]   = k[t,n] * v[t,n]
    y[t,n]  = r[t,n] * (u[n] * kv[n] + state[n])
    state[n] = exp(w[n]) * state[n] + kv[n]

A100 strategy
─────────────
Grid: (B*H,) — one block per (batch, head) pair.
Threads: N — one thread per feature dimension n (assumes N ≤ 1024).

Each thread owns one dimension n and maintains its scalar `state` in a
register across T time steps.  The inner loop reads k[t,n], v[t,n], r[t,n]
from global memory (one __half per thread per input tensor per step) and
writes y[t,n].

Using float4 vectorisation would require N divisible by 4; for generality
this kernel uses scalar loads.  The A100's large register file (255 regs/thread)
easily holds the running state without spilling.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <math.h>

__global__ void wkv_fwd_a100(
    const __half* __restrict__ r,   // [B, H, T, N]
    const __half* __restrict__ k,   // [B, H, T, N]
    const __half* __restrict__ v,   // [B, H, T, N]
    const float*  __restrict__ w,   // [H, N]  log-decay
    const __half* __restrict__ u,   // [H, N]  bonus
    __half*       __restrict__ y,   // [B, H, T, N]
    int B, int H, int T, int N)
{
    const int bh  = blockIdx.x;   // in [0, B*H)
    const int b   = bh / H;
    const int h   = bh % H;
    const int n   = threadIdx.x;  // feature dimension

    if (n >= N) return;

    // Load per-head constants for this feature into registers.
    float dec = expf(w[h * N + n]);                // decay scalar
    float un  = __half2float(u[h * N + n]);        // bonus scalar

    float state = 0.f;                             // running state

    // Pointers to the first time step for (b, h, n).
    const __half* rn = r + (b * H + h) * T * N + n;
    const __half* kn = k + (b * H + h) * T * N + n;
    const __half* vn = v + (b * H + h) * T * N + n;
    __half*       yn = y + (b * H + h) * T * N + n;

    // Sequential scan over T; each step touches 4 scalar global reads + 1 write.
    for (int t = 0; t < T; ++t) {
        float rt = __half2float(rn[t * N]);
        float kt = __half2float(kn[t * N]);
        float vt = __half2float(vn[t * N]);

        float kv = kt * vt;
        float yt = rt * (un * kv + state);
        state = dec * state + kv;

        yn[t * N] = __float2half(yt);
    }
}

void wkv_fwd(
    torch::Tensor r, torch::Tensor k, torch::Tensor v,
    torch::Tensor w, torch::Tensor u, torch::Tensor y)
{
    int B = r.size(0), H = r.size(1), T = r.size(2), N = r.size(3);
    dim3 grid(B * H);
    wkv_fwd_a100<<<grid, N>>>(
        reinterpret_cast<const __half*>(r.data_ptr()),
        reinterpret_cast<const __half*>(k.data_ptr()),
        reinterpret_cast<const __half*>(v.data_ptr()),
        w.data_ptr<float>(),
        reinterpret_cast<const __half*>(u.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),
        B, H, T, N);
    cudaDeviceSynchronize();
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="wkv_fwd_src",
            cpp_sources="void wkv_fwd(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["wkv_fwd"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        w: torch.Tensor, u: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.empty_like(r)
        _get_ext().wkv_fwd(
            r.contiguous(), k.contiguous(), v.contiguous(),
            w.contiguous(), u.contiguous(), y,
        )
        return y
