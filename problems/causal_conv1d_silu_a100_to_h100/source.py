"""
Source kernel: causal 1D depthwise convolution with SiLU, optimised for A100.
Adapted from Dao-AILab/causal-conv1d (Apache 2.0, Tri Dao 2024).

Layout:  x [B, D, L]  fp16   weight [D, 4]  fp16   →   y [B, D, L]  fp16

A100 strategy
─────────────
Grid: (B, D) — one thread-block per (batch, channel) pair.
Threads: 128 per block, each handling kElts=4 elements per chunk iteration.
Chunk size: 512 elements (128 threads × 4 elements).

Shared memory holds a [kW-1 history | chunk buffer] staging area.
  • The kW-1=3 history slots survive across chunk iterations so no extra
    global-memory re-reads are needed for the causal context.
  • All 128 threads cooperate to load one chunk into smem, then each thread
    computes its 4 output values with a manually unrolled #pragma unroll loop.

SiLU (x / (1 + exp(-x))) is applied in-place before the fp16 store.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <math.h>

static const int kNT    = 128;   // threads per block
static const int kW     = 4;     // filter width
static const int kElts  = 4;     // elements per thread per chunk
static const int kChunk = kNT * kElts;   // 512

// smem layout: [kW-1 history floats | kChunk chunk floats]
static const int kSmem  = (kW - 1 + kChunk) * sizeof(float);

__global__ __launch_bounds__(kNT)
void causal_conv1d_silu_a100(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half*       __restrict__ y,
    int L)
{
    const int b   = blockIdx.x;
    const int d   = blockIdx.y;
    const int D   = gridDim.y;
    const int tid = threadIdx.x;

    // Load per-channel filter weights into registers.
    float wf[kW];
    #pragma unroll
    for (int i = 0; i < kW; ++i)
        wf[i] = __half2float(weight[d * kW + i]);

    // Shared memory: [hist: kW-1 floats] [buf: kChunk floats]
    extern __shared__ float smem[];
    float* hist = smem;             // causal history from previous chunk
    float* buf  = smem + (kW - 1); // current chunk staging area

    // Initialise history to zero (left-boundary padding).
    if (tid < kW - 1)
        hist[tid] = 0.f;
    __syncthreads();

    const __half* xrow = x + (b * D + d) * L;
    __half*       yrow = y + (b * D + d) * L;
    const int n_chunks = (L + kChunk - 1) / kChunk;

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int chunk_start = chunk * kChunk;

        // Cooperatively load kChunk elements into smem buf.
        #pragma unroll 4
        for (int i = 0; i < kElts; ++i) {
            int pos = chunk_start + tid * kElts + i;
            buf[tid * kElts + i] = pos < L ? __half2float(xrow[pos]) : 0.f;
        }
        __syncthreads();

        // Each thread computes kElts output values.
        #pragma unroll 4
        for (int i = 0; i < kElts; ++i) {
            const int local = tid * kElts + i;  // position within smem buf
            const int gpos  = chunk_start + local;
            if (gpos >= L) break;

            float acc = 0.f;
            #pragma unroll
            for (int wi = 0; wi < kW; ++wi) {
                // Tap index into the [hist | buf] smem window:
                //   positive → buf[tap]
                //   negative → hist[kW-1 + tap]
                int tap = local - (kW - 1 - wi);  // = local + wi - (kW-1)
                float xval = tap >= 0 ? buf[tap] : hist[kW - 1 + tap];
                acc += wf[wi] * xval;
            }
            // SiLU activation
            acc /= 1.f + expf(-acc);
            yrow[gpos] = __float2half(acc);
        }
        __syncthreads();

        // Persist last kW-1 elements of buf into history for next chunk.
        if (tid < kW - 1)
            hist[tid] = buf[kChunk - (kW - 1) + tid];
        __syncthreads();
    }
}

void causal_conv1d_silu_fwd(
    torch::Tensor x, torch::Tensor weight, torch::Tensor y)
{
    int B = x.size(0), D = x.size(1), L = x.size(2);
    dim3 grid(B, D);
    causal_conv1d_silu_a100<<<grid, kNT, kSmem>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),
        L);
    cudaDeviceSynchronize();
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="causal_conv1d_silu_src",
            cpp_sources="void causal_conv1d_silu_fwd(torch::Tensor, torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["causal_conv1d_silu_fwd"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        y.zero_()
        _get_ext().causal_conv1d_silu_fwd(x.contiguous(), weight.contiguous(), y)
        return y
