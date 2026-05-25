"""
src.py — flash_attn_v4_a100_to_h100 (src kernel via load_inline).
Raw CUDA lives alongside this file; the ensemble prompt includes those sources.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_DIR = Path(os.environ.get("KTBENCH_PROBLEM_PATH", Path(__file__).resolve().parent))

_CUDA_BRIDGE = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
extern void attention_v4(const nv_bfloat16 *Q, const nv_bfloat16 *K, const nv_bfloat16 *V,
                         nv_bfloat16 *O, int bs, int len_q, int len_kv, int dim);
void run_attn_v4(int64_t q_ptr, int64_t k_ptr, int64_t v_ptr, int64_t o_ptr,
                 int bs, int len_q, int len_kv, int dim) {
  attention_v4((const nv_bfloat16*)q_ptr, (const nv_bfloat16*)k_ptr, (const nv_bfloat16*)v_ptr,
               (nv_bfloat16*)o_ptr, bs, len_q, len_kv, dim);
  cudaDeviceSynchronize();
}
"""

_CPP_DECL = r"""
void run_attn_v4(int64_t q_ptr, int64_t k_ptr, int64_t v_ptr, int64_t o_ptr, int bs, int len_q, int len_kv, int dim);
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        sources = ["_DIR / 'common.h'",
        "_DIR / 'hopper_flash.h'",
        "_DIR / 'hopper_utils.h'",
        "_DIR / 'source.cu'"]
        _ext = load_inline(
            name="flash_attn_v4_a100_to_h100_src",
            cpp_sources=_CPP_DECL,
            cuda_sources=sources,
            extra_include_paths=[str(_DIR)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            functions=["run_attn_v4"],
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        bs, len_q, dim = Q.shape
        len_kv = K.shape[1]
        O = torch.empty_like(Q)
        O.zero_()
        Qc = Q.contiguous()
        Kc = K.contiguous()
        Vc = V.contiguous()
        _get_ext().run_attn_v4(
            Qc.data_ptr(), Kc.data_ptr(), Vc.data_ptr(), O.data_ptr(),
            bs, len_q, len_kv, dim)
        return O

