"""
Ground-truth oracle: exponential-decay cumulative sum in PyTorch eager.

y[t] = decay * y[t-1] + x[t]   where decay = exp(log_decay).

Sequential loop in fp32 — slow but unambiguously correct.
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, log_decay: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, D]   log_decay: [H, D]
        B, H, L, D = x.shape
        dt = x.dtype
        x32 = x.float()
        decay = torch.exp(log_decay)  # [H, D], values in (0, 1)

        y = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
        out = torch.empty(B, H, L, D, dtype=torch.float32, device=x.device)

        for t in range(L):
            y = y * decay.unsqueeze(0) + x32[:, :, t, :]  # [B, H, D]
            out[:, :, t, :] = y

        return out.to(dt)
