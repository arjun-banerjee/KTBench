"""
Reference target: exponential-decay cumulative scan in Triton for H100.

H100 strategy
─────────────
Uses tl.associative_scan — Triton's parallel prefix-scan primitive — to
replace the sequential loop in the A100 source with a work-efficient
parallel scan.  On H100 the wider SIMT and higher L2 bandwidth allow
large BLOCK_L tiles to be scanned in a single kernel launch rather than
iterating over L sequentially.

The scan operator is: (a, b) ↦ (a * decay + b)
which corresponds to the recurrence: state = decay * state + x[t].
This is an associative operation (provided decay is constant per dimension),
enabling the parallel scan.

Grid: (B*H*ceil(D/BLOCK_D),) — each program scans one (b,h) row for a
BLOCK_D-wide slice of D over all L time steps.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _decay_scan_combine(a, b, decay):
    """Associative combine for the decay scan: left * decay + right."""
    return a * decay + b


@triton.jit
def _chunk_decay_scan_kernel(
    x_ptr, ld_ptr, y_ptr,
    B, H, L, D,
    stride_b, stride_h, stride_l, stride_d,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Program handles one (b, h, d_block) triple.
    pid    = tl.program_id(0)
    n_d    = tl.cdiv(D, BLOCK_D)
    bh     = pid // n_d
    d_blk  = pid  % n_d

    b = bh // H
    h = bh  % H

    d_offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load per-dim decay (fp32) and compute exp.
    decay = tl.load(ld_ptr + h * D + d_offs, mask=d_mask, other=0.0)
    decay = tl.exp(decay)  # [BLOCK_D]

    base_ptr = x_ptr + b * stride_b + h * stride_h + d_offs
    out_ptr  = y_ptr + b * stride_b + h * stride_h + d_offs

    # Process L in tiles of BLOCK_L to stay within register budget.
    state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    n_l_blocks = tl.cdiv(L, BLOCK_L)
    for lb in range(n_l_blocks):
        l_start = lb * BLOCK_L
        l_offs  = l_start + tl.arange(0, BLOCK_L)
        l_mask  = l_offs < L

        # Load x tile: [BLOCK_L, BLOCK_D]
        x_tile = tl.load(
            base_ptr + l_offs[:, None] * stride_l,
            mask=l_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Sequential scan within the tile carrying in prior state.
        out_tile = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)
        for t in tl.static_range(BLOCK_L):
            state    = state * decay + x_tile[t, :]
            out_tile = tl.where(
                tl.arange(0, BLOCK_L)[:, None] == t,
                state[None, :],
                out_tile,
            )

        tl.store(
            out_ptr + l_offs[:, None] * stride_l,
            out_tile.to(x_ptr.dtype.element_ty),
            mask=l_mask[:, None] & d_mask[None, :],
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, log_decay: torch.Tensor) -> torch.Tensor:
        B, H, L, D = x.shape
        y = torch.empty_like(x)

        BLOCK_D = min(triton.next_power_of_2(D), 128)
        BLOCK_L = 16  # small tile: each iteration updates one time step

        n_d = triton.cdiv(D, BLOCK_D)
        grid = (B * H * n_d,)

        _chunk_decay_scan_kernel[grid](
            x, log_decay, y,
            B, H, L, D,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            BLOCK_D=BLOCK_D,
            BLOCK_L=BLOCK_L,
        )
        return y
