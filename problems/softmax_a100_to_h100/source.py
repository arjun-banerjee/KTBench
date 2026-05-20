"""
Source kernel: online softmax in CUDA (H200).
Uses the Milakov-Norrie two-pass online algorithm for numerical stability.
Candidate must translate this to Triton.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <float.h>

// One warp per row. Works for D <= 1024 (32 threads × 32 elements).
// For D > 1024 a block-reduce version is needed.
template <typename scalar_t>
__global__ void softmax_kernel(
    const scalar_t* __restrict__ x,
    scalar_t*       __restrict__ y,
    int N, int D)
{
    int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t* row_x = x + row * D;
    scalar_t*       row_y = y + row * D;

    // Pass 1: find max and sum exp(x - max) simultaneously (online algorithm)
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        float v = (float)row_x[col];
        if (v > max_val) {
            sum_exp *= expf(max_val - v);
            max_val = v;
        }
        sum_exp += expf(v - max_val);
    }

    // Warp reduce max and sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        float peer_max = __shfl_xor_sync(0xffffffff, max_val, offset);
        float peer_sum = __shfl_xor_sync(0xffffffff, sum_exp, offset);
        if (peer_max > max_val) {
            sum_exp *= expf(max_val - peer_max);
            max_val = peer_max;
        } else {
            peer_sum *= expf(peer_max - max_val);
        }
        sum_exp += peer_sum;
    }

    // Pass 2: write normalised output
    for (int col = threadIdx.x; col < D; col += blockDim.x)
        row_y[col] = (scalar_t)(expf((float)row_x[col] - max_val) / sum_exp);
}

void softmax_forward(
    torch::Tensor x,
    torch::Tensor y)
{
    int N = x.size(0);
    int D = x.size(1);
    int threads = min(D, 32);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "softmax_forward", [&]() {
        softmax_kernel<scalar_t><<<N, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            N, D);
    });
    cudaDeviceSynchronize();
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="softmax_cuda_src",
            cpp_sources="void softmax_forward(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["softmax_forward"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        # Zero the output buffer - prevents stale-memory exploits
        y.zero_()
        _get_ext().softmax_forward(x.contiguous(), y)
        return y
