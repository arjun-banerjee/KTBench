"""
Ground-truth oracle: fused RMSNorm + residual in PyTorch eager.
Accumulates in fp32 for numerical stability.
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x, r: [B, L, D]    w: [D]
        dt = x.dtype
        z = (x.float() + r.float())          # residual add in fp32
        rms = z.pow(2).mean(dim=-1, keepdim=True).add_(1e-6).rsqrt()
        y = z * rms * w.float()              # normalize + affine scale
        return y.to(dt)
