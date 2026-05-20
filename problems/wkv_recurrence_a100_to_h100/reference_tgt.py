"""
Reference target: WKV token-mixing recurrence in Triton for H100.

H100 strategy
─────────────
Grid: (B*H,) — one program per (batch, head) pair.
Each program iterates over T time steps, processing BLOCK_N feature
dimensions per iteration using Triton's vectorised load/store.

Compared to the A100 source (one thread per feature, scalar loads),
the Triton version:
  • Loads BLOCK_N elements at a time using tl.load, exploiting H100's
    wider memory bus and async load capabilities.
  • The running state is a BLOCK_N-wide tl.float32 vector kept in
    registers rather than one scalar per thread.
  • tl.static_range(T) would require T to be constexpr; instead we loop
    with a Python-level for loop which Triton handles as a dynamic loop.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _wkv_kernel(
    r_ptr, k_ptr, v_ptr, w_ptr, u_ptr, y_ptr,
    B, H, T, N,
    stride_b, stride_h, stride_t, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid  % H

    n_offs = tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    base = b * stride_b + h * stride_h

    # Load per-head constants: decay and bonus.
    decay = tl.exp(tl.load(w_ptr + h * N + n_offs, mask=n_mask, other=0.0))  # fp32
    u     = tl.load(u_ptr + h * N + n_offs, mask=n_mask, other=0.0).to(tl.float32)

    state = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for t in range(T):
        t_off = t * stride_t
        rt = tl.load(r_ptr + base + t_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        kt = tl.load(k_ptr + base + t_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + base + t_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)

        kv    = kt * vt
        yt    = rt * (u * kv + state)
        state = decay * state + kv

        tl.store(y_ptr + base + t_off + n_offs,
                 yt.to(r_ptr.dtype.element_ty),
                 mask=n_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        w: torch.Tensor, u: torch.Tensor,
    ) -> torch.Tensor:
        B, H, T, N = r.shape
        y = torch.empty_like(r)

        BLOCK_N = min(triton.next_power_of_2(N), 1024)

        _wkv_kernel[(B * H,)](
            r, k, v, w, u, y,
            B, H, T, N,
            r.stride(0), r.stride(1), r.stride(2), r.stride(3),
            BLOCK_N=BLOCK_N,
        )
        return y
