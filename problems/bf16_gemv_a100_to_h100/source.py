"""
source.py — A100 CUDA kernel for bf16_gemv_a100_to_h100.

y = W @ x  (BF16): x[K], W[N,K] row-major → y[N]

Algorithm: 4 warps per block; each warp owns one output row.
  Each thread loads 8 bf16 per iteration via 128-bit global loads (v4.b32).
  x is read from global memory independently by every warp.
  K must be divisible by WARP_SIZE * 8 = 256.
  N must be divisible by NUM_WARPS = 4.

Source: adapted from https://github.com/gau-nernst/learn-cuda/blob/main/11_gemv/cuda_v1.cu (MIT)
Compile target: sm_80 (A100).
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
// Each thread processes 8 bf16 per iteration (4 int32 = 128-bit load)
constexpr int NUM_ELEM = 8;

// Unpack one packed pair of bf16 → two fp32
__device__ inline void bf16x2_to_fp32x2(float *out, uint32_t packed) {
    asm volatile(
        "shl.b32  %0, %2, 16;\n"
        "and.b32  %1, %2, 0xFFFF0000;\n"
        : "=f"(out[0]), "=f"(out[1])
        : "r"(packed));
}

// 128-bit load: 4 × int32 = 8 × bf16 from global memory
__device__ inline void ldg128(uint32_t *d, const void *ptr) {
    asm volatile(
        "ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "l"(ptr));
}

__global__ void gemv_a100_kernel(
    const __nv_bfloat16 * __restrict__ x,   // [K]
    const __nv_bfloat16 * __restrict__ W,   // [N, K] row-major
    __nv_bfloat16       * __restrict__ y,   // [N]
    int N, int K)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int row     = blockIdx.x * NUM_WARPS + warp_id;
    if (row >= N) return;

    float acc = 0.f;
    const int iters = K / (WARP_SIZE * NUM_ELEM);

    for (int i = 0; i < iters; i++) {
        const int col = (i * WARP_SIZE + lane_id) * NUM_ELEM;
        uint32_t xr[4], wr[4];
        float    xf[NUM_ELEM], wf[NUM_ELEM];

        // Each warp reads x from global memory independently
        ldg128(xr, x + col);
        ldg128(wr, W + row * K + col);

        for (int j = 0; j < 4; j++) {
            bf16x2_to_fp32x2(xf + j * 2, xr[j]);
            bf16x2_to_fp32x2(wf + j * 2, wr[j]);
            acc += xf[j*2+0] * wf[j*2+0] + xf[j*2+1] * wf[j*2+1];
        }
    }

    // Warp reduction
    for (int s = WARP_SIZE / 2; s > 0; s /= 2)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, s);

    if (lane_id == 0)
        y[row] = __float2bfloat16_rn(acc);
}

void gemv_a100(at::Tensor x, at::Tensor W, at::Tensor y) {
    const int N = W.size(0);
    const int K = W.size(1);
    const int blocks = N / NUM_WARPS;
    gemv_a100_kernel<<<blocks, NUM_WARPS * WARP_SIZE>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        N, K);
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="gemv_a100_cuda",
            cpp_sources="void gemv_a100(at::Tensor, at::Tensor, at::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["gemv_a100"],
            extra_cuda_cflags=["-O3", "-arch=sm_80"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # x: [K] bf16, W: [N, K] bf16  →  y: [N] bf16
        y = torch.zeros(W.size(0), dtype=x.dtype, device=x.device)
        _get_ext().gemv_a100(x.contiguous(), W.contiguous(), y)
        return y
