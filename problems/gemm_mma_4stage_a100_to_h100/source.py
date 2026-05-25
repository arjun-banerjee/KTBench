"""
src.py — gemm_mma_4stage_a100_to_h100 (src kernel via load_inline).
Raw CUDA lives alongside this file; the ensemble prompt includes those sources.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_DIR = Path(os.environ.get("KTBENCH_PROBLEM_PATH", Path(__file__).resolve().parent))

_CUDA_BRIDGE = r"""

"""

_CPP_DECL = r"""
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        sources = ["_DIR / 'source.cu'"]
        _ext = load_inline(
            name="gemm_mma_4stage_a100_to_h100_src",
            cpp_sources=_CPP_DECL,
            cuda_sources=sources,
            extra_include_paths=[str(_DIR)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            functions=["hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4"],
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        _, N = B.shape
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        C.zero_()
        _get_ext().hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(A.contiguous(), B.contiguous(), C, 4, True, 256)
        return C

