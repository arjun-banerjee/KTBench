"""H100 reference placeholder (torch.mm, fp32 accumulation)."""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.mm(A.float(), B.float()).to(A.dtype)
