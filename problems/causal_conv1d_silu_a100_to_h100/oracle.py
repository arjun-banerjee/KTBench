"""
Ground-truth oracle: causal depthwise conv1d + SiLU in PyTorch eager.
Uses fp32 accumulation throughout for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

kWidth = 4


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x:      [B, D, L]
        # weight: [D, 4]
        B, D, L = x.shape
        dt = x.dtype
        x32 = x.float()
        w32 = weight.float()  # [D, 4]

        # F.conv1d expects weight [out_channels, in_channels/groups, kW]
        # For depthwise: out=D, groups=D, in_per_group=1 → weight [D, 1, 4]
        w_conv = w32.unsqueeze(1)  # [D, 1, 4]

        # Left-pad with kWidth-1 zeros for causal semantics
        x_padded = F.pad(x32, (kWidth - 1, 0))  # [B, D, L + kWidth-1]

        y32 = F.conv1d(x_padded, w_conv, groups=D)  # [B, D, L]
        y32 = F.silu(y32)

        return y32.to(dt)
