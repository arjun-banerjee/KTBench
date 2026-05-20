"""
Reference target: fused RMSNorm + residual in Triton for H100.

H100 strategy
─────────────
Grid: (B*L,) — one program per token row.
Each program loads the entire D-vector, computes sum-of-squares with tl.sum,
normalises, scales, and stores — all in a single pass.

tl.sum replaces the two rounds of __shfl_down_sync + shared-memory cross-warp
reduction used in the A100 source.  On H100 the wider SIMT units and higher
L2 bandwidth allow larger BLOCK_D tiles without the explicit 2-pass structure.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rms_norm_residual_kernel(
    x_ptr, r_ptr, w_ptr, y_ptr,
    D, eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # Load x + r
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    z = x + r  # residual-added vector

    # RMS: sqrt(mean(z^2) + eps)
    mean_sq = tl.sum(z * z, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)

    # Normalise and affine scale
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    out = z * inv_rms * w

    tl.store(y_ptr + row * D + offs,
             out.to(x_ptr.dtype.element_ty),
             mask=mask)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        y = torch.empty_like(x)
        rows = B * L

        BLOCK_D = min(_next_power_of_2(D), 8192)

        _rms_norm_residual_kernel[(rows,)](
            x.contiguous().view(rows, D),
            r.contiguous().view(rows, D),
            w, y.view(rows, D),
            D, 1e-6,
            BLOCK_D=BLOCK_D,
        )
        return y
