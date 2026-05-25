"""PyTorch SDPA reference (fp32 accumulation)."""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        )
        return out.to(Q.dtype)
