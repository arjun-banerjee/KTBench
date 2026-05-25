"""
src.py — gemm_v1_a100_to_h100 (src kernel via load_inline).
Raw CUDA lives alongside this file; the ensemble prompt includes those sources.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_DIR = Path(os.environ["KTBENCH_PROBLEM_PATH"])

_CUDA_BRIDGE = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
extern void run_gemm(void* A, void* B, void* C, int M, int N, int K);
void run_src_gemm(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr, int M, int K, int N) {
  run_gemm((void*)A_ptr, (void*)B_T_ptr, (void*)C_ptr, M, N, K);
  cudaDeviceSynchronize();
}
"""

_CPP_DECL = r"""
void run_src_gemm(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr, int M, int K, int N);
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        sources = ["_DIR / 'common.h'",
        "_DIR / 'source.cu'"]
        _ext = load_inline(
            name="gemm_v1_a100_to_h100_src",
            cpp_sources=_CPP_DECL,
            cuda_sources=sources,
            extra_include_paths=[str(_DIR)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            functions=["run_src_gemm"],
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        _, N = B.shape
        B_T = B.t().contiguous()
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        C.zero_()
        _get_ext().run_src_gemm(A.data_ptr(), B_T.data_ptr(), C.data_ptr(), M, K, N)
        return C

