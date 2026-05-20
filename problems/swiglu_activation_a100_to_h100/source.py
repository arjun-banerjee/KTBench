"""Source kernel: SwiGLU activation in CUDA, tuned for A100 (Ampere).

silu(gate) * up, where the input x = concat(gate, up) along the last
dim. Element-wise; one thread per output element. Float32 sigmoid
accumulation to keep numerical stability on fp16 inputs.

Candidate must translate this to a Hopper-friendly CUDA kernel.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void swiglu_kernel(
    const scalar_t* __restrict__ x,
    scalar_t*       __restrict__ y,
    int N, int D)
{
    // x: [N, 2*D], y: [N, D]
    int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t* x_row = x + row * 2 * D;
    scalar_t*       y_row = y + row * D;

    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        float gate = (float)x_row[col];
        float up   = (float)x_row[col + D];
        float silu = gate / (1.0f + expf(-gate));
        y_row[col] = (scalar_t)(silu * up);
    }
}

void swiglu_forward(torch::Tensor x, torch::Tensor y) {
    int N = x.size(0);
    int D = y.size(1);
    int threads = D < 256 ? D : 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "swiglu_forward", [&]() {
        swiglu_kernel<scalar_t><<<N, threads>>>(
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
            name="swiglu_a100_src",
            cpp_sources="void swiglu_forward(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["swiglu_forward"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2*D] -> y: [N, D]
        D = x.shape[-1] // 2
        y = torch.empty(x.shape[0], D, dtype=x.dtype, device=x.device)
        y.zero_()
        _get_ext().swiglu_forward(x.contiguous(), y)
        return y
