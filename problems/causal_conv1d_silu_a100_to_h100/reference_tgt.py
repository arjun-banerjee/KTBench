"""
Reference target: causal 1D depthwise conv + SiLU in Triton for H100.

H100 strategy
─────────────
Grid: (n_blocks_L, B*D) — each program handles BLOCK_L output positions
for one (batch, channel) pair.  No shared memory or inter-program
communication is needed: for each weight tap wi, a contiguous slice of L
is loaded independently, masked at the left causal boundary with other=0.0.
This maps naturally to H100's wider vector load instructions.

Differences from the A100 source
──────────────────────────────────
• No explicit shared-memory staging; each program owns its data.
• No inter-chunk state: the BLOCK_L-sized window plus (kW-1) masked loads
  fully replaces the smem history ring buffer.
• The inner loop over weight taps is statically unrolled by the Triton
  compiler, generating back-to-back vector loads with constant offsets.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

kWidth = 4


@triton.jit
def _causal_conv1d_silu_kernel(
    x_ptr, w_ptr, y_ptr,
    B, D, L,
    stride_b, stride_d,
    BLOCK_L: tl.constexpr,
    kW: tl.constexpr,
):
    # Program axes: pid_l = block along L, pid_bd = (batch, channel) index
    pid_l  = tl.program_id(0)
    pid_bd = tl.program_id(1)

    b = pid_bd // D
    d = pid_bd % D

    block_start = pid_l * BLOCK_L

    # Load per-channel filter weights [kW] into registers.
    w = tl.load(w_ptr + d * kW + tl.arange(0, kW)).to(tl.float32)

    # Output positions this program is responsible for.
    out_offs  = block_start + tl.arange(0, BLOCK_L)
    out_mask  = out_offs < L
    base_ptr  = x_ptr + b * stride_b + d * stride_d

    # Accumulate: for each weight tap wi, load x[out_pos - (kW-1-wi)].
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    for wi in tl.static_range(kW):
        tap_offs = out_offs - (kW - 1 - wi)   # causal offset; may be negative
        tap_mask = out_mask & (tap_offs >= 0)
        xv = tl.load(base_ptr + tap_offs, mask=tap_mask, other=0.0).to(tl.float32)
        acc += w[wi] * xv

    # SiLU: x / (1 + exp(-x))
    acc = acc / (1.0 + tl.exp(-acc))

    tl.store(y_ptr + b * stride_b + d * stride_d + out_offs,
             acc.to(x_ptr.dtype.element_ty),
             mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        B, D, L = x.shape
        y = torch.empty_like(x)

        BLOCK_L = min(triton.next_power_of_2(L), 512)
        grid = (triton.cdiv(L, BLOCK_L), B * D)

        _causal_conv1d_silu_kernel[grid](
            x, weight, y,
            B, D, L,
            x.stride(0), x.stride(1),
            BLOCK_L=BLOCK_L, kW=kWidth,
        )
        return y
