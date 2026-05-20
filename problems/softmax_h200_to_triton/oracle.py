"""
Ground-truth oracle: row-wise softmax in PyTorch eager (fp32 accumulation).
This is numerically correct by definition and used to verify candidate outputs.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 for stable accumulation, cast back to input dtype.
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
