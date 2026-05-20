"""
Handwritten reference: online softmax in Triton (H200).
This is the performance baseline — candidates are scored relative to this.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr, y_ptr,
    N, D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    # Load a block of D elements
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=-float("inf")).to(tl.float32)

    # Numerically stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp

    tl.store(y_ptr + row * D + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        y = torch.empty_like(x)
        BLOCK_D = min(_next_power_of_2(D), 16384)
        _softmax_kernel[(N,)](x, y, N, D, BLOCK_D=BLOCK_D)
        return y
