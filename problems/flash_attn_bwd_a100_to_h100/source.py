"""Source stub for flash_attn_bwd_a100_to_h100.

The A100 kernel is in source.cu and companion headers (included in the prompt).
This stub exists so the harness can load the problem; agents translate from the
prompt bundle, not from executing this file.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                dO: torch.Tensor) -> tuple:
        Q = Q.float().requires_grad_(True)
        K = K.float().requires_grad_(True)
        V = V.float().requires_grad_(True)
        O = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        O.backward(dO.float())
        return Q.grad.to(Q.dtype), K.grad.to(K.dtype), V.grad.to(V.dtype)
