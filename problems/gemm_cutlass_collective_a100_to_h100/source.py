"""Source stub for gemm_cutlass_collective_a100_to_h100.

The A100 collective GEMM mainloop is in source.cu (included in the prompt).
This stub exists so the harness can load the problem; agents translate from the
prompt bundle, not from executing this file.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.mm(A.float(), B.float()).to(A.dtype)
