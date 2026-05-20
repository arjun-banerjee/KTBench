"""
Ground-truth oracle: WKV token-mixing recurrence in PyTorch eager.

Recurrence (fp32 accumulation):
  state = 0
  For t in range(T):
    kv   = k[t] * v[t]                     elementwise, [B, H, N]
    y[t] = r[t] * (u * kv + state)
    state = exp(w) * state + kv
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        w: torch.Tensor, u: torch.Tensor,
    ) -> torch.Tensor:
        # r, k, v: [B, H, T, N]   w: [H, N] fp32   u: [H, N]
        B, H, T, N = r.shape
        dt = r.dtype

        decay = torch.exp(w).to(torch.float32)  # [H, N]
        state = torch.zeros(B, H, N, dtype=torch.float32, device=r.device)
        out   = torch.empty(B, H, T, N, dtype=torch.float32, device=r.device)

        u32 = u.float()

        for t in range(T):
            rt = r[:, :, t, :].float()  # [B, H, N]
            kt = k[:, :, t, :].float()
            vt = v[:, :, t, :].float()

            kv = kt * vt                                     # [B, H, N]
            yt = rt * (u32.unsqueeze(0) * kv + state)       # [B, H, N]
            state = decay.unsqueeze(0) * state + kv          # [B, H, N]
            out[:, :, t, :] = yt

        return out.to(dt)
