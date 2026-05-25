"""Source stub for flash_attn_dao_a100_to_h100.

The A100 kernel is in source.cu and companion headers (included in the prompt).
This stub exists so the harness can load the problem; agents translate from the
prompt bundle, not from executing this file.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        ).to(Q.dtype)
