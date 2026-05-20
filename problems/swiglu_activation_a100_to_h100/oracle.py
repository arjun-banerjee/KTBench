"""Ground-truth oracle for SwiGLU activation.

Input  x: [N, 2*D] (gate concatenated with up)
Output y: [N, D]   silu(gate) * up

Matches /scratch/abaner/.../hardware_translation/oracle/04_swiglu_activation.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
