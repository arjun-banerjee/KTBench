"""PyTorch eager GEMM reference."""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
