"""
Ground-truth oracle: normalised Walsh-Hadamard Transform in PyTorch eager.

H_N is the N×N Walsh-Hadamard matrix defined recursively:
  H_1 = [1]
  H_N = (1/sqrt(2)) * [[H_{N/2}, H_{N/2}], [H_{N/2}, -H_{N/2}]]

The normalised transform y = H_N @ x satisfies ||H_N x|| = ||x||.
We compute it with the standard in-place butterfly algorithm in fp32.
"""

import torch
import torch.nn as nn


def hadamard_transform_ref(x: torch.Tensor) -> torch.Tensor:
    """In-place Cooley-Tukey butterfly, O(N log N)."""
    B, N = x.shape
    h = x.clone().float()
    step = 1
    while step < N:
        for i in range(0, N, step * 2):
            u = h[:, i : i + step].clone()
            v = h[:, i + step : i + step * 2].clone()
            h[:, i : i + step]             = (u + v) * 0.7071067811865476  # /sqrt(2)
            h[:, i + step : i + step * 2] = (u - v) * 0.7071067811865476
        step *= 2
    return h


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        y = hadamard_transform_ref(x)
        return y.to(dt)
