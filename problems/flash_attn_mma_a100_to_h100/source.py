"""
src.py — flash_attn_mma_a100_to_h100 (src kernel via load_inline).
Raw CUDA lives alongside this file; the ensemble prompt includes those sources.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_DIR = Path(os.environ.get("KTBENCH_PROBLEM_PATH", Path(__file__).resolve().parent))

_CUDA_BRIDGE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
void flash_attn_mma_stages_split_q_shared_kv(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int stages);
void run_flash_fwd(int64_t q_ptr, int64_t k_ptr, int64_t v_ptr, int64_t o_ptr,
                   int B, int H, int N, int D) {
  auto opts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
  auto Q = torch::from_blob((void*)q_ptr, {B, H, N, D}, opts);
  auto K = torch::from_blob((void*)k_ptr, {B, H, N, D}, opts);
  auto V = torch::from_blob((void*)v_ptr, {B, H, N, D}, opts);
  auto O = torch::from_blob((void*)o_ptr, {B, H, N, D}, opts);
  flash_attn_mma_stages_split_q_shared_kv(Q, K, V, O, 2);
  cudaDeviceSynchronize();
}
"""

_CPP_DECL = r"""
void run_flash_fwd(int64_t q_ptr, int64_t k_ptr, int64_t v_ptr, int64_t o_ptr, int B, int H, int N, int D);
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        sources = ["_DIR / 'hopper_softmax.h'",
        "_DIR / 'source.cu'",
        "_DIR / 'utils.h'"]
        _ext = load_inline(
            name="flash_attn_mma_a100_to_h100_src",
            cpp_sources=_CPP_DECL,
            cuda_sources=sources,
            extra_include_paths=[str(_DIR)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            functions=["run_flash_fwd"],
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, N, D = Q.shape
        O = torch.empty_like(Q)
        O.zero_()
        _get_ext().run_flash_fwd(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), B, H, N, D)
        return O

