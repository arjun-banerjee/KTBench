"""
source.py — A100 CUDA kernel for bf16_gemm_reg2d_a100_to_h100.
Source: https://github.com/HamzaElshafie/h100_gemm/blob/main/src/kernels/general/gemm_2D_registertiling_bf16.cuh
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); } } while(0)

#define CU_CHECK(call) \
    do { CUresult _r = (call); if (_r != CUDA_SUCCESS) { \
        const char* _s = nullptr; cuGetErrorString(_r, &_s); \
        throw std::runtime_error(std::string("CU: ") + (_s ? _s : "unknown")); } } while(0)

#define CEIL_DIV(value, divisor) (((value) + (divisor) - 1) / (divisor))

// gemm_2D_registertiling_bf16.cuh (verbatim from GitHub)
#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * @brief 2D Register Tiling GEMM for BF16 (accumulates in FP32).
 */
template <const uint TILE_SIZE_M, const uint TILE_SIZE_N, const uint TILE_SIZE_K, const uint ROWS_PER_THREAD, const uint COLS_PER_THREAD>
__global__ void gemm_2D_registertiling_bf16(const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    // Allocate shared memory
    __shared__ __nv_bfloat16 sharedA[TILE_SIZE_M * TILE_SIZE_N];
    __shared__ __nv_bfloat16 sharedB[TILE_SIZE_N * TILE_SIZE_K];

    // Identify the tile of C this thread block is responsible for
    const uint block_row = blockIdx.y;
    const uint block_column = blockIdx.x;

    // Calculate position of thread within tile (Remapping from 1-D to 2-D) Note --> Each thread is a grid in itself hanlding ROWS_PER_THREAD x COLS_PER_THREAD
    const uint ty = threadIdx.x / (TILE_SIZE_K / COLS_PER_THREAD);
    const uint tx = threadIdx.x % (TILE_SIZE_K / COLS_PER_THREAD);

    // Move pointers from A[0], B[0] and C[0] to the starting positions of the tile
    A += block_row * TILE_SIZE_M * N;                                  // Move pointer (block_row * TILE_SIZE_M) rows down
    B += block_column * TILE_SIZE_K;                                   // Move pointer (block_column * TILE_SIZE_K) columns to the right
    C += (block_row * TILE_SIZE_M * K) + (block_column * TILE_SIZE_K); // Move pointer (block_row * TILE_SIZE_M * K) rows down then (block_column * TILE_SIZE_K) columns to the right

    // Calculate position of thread within shared memory tile (To be used while loading into proper postions in smem)
    const uint smem_ty_A = threadIdx.x / TILE_SIZE_N;
    const uint smem_tx_A = threadIdx.x % TILE_SIZE_N;

    const uint smem_ty_B = threadIdx.x / TILE_SIZE_K;
    const uint smem_tx_B = threadIdx.x % TILE_SIZE_K;

    // Total results calculated by a single tile in C
    const uint total_results_per_tile = TILE_SIZE_M * TILE_SIZE_K;
    // Calculate total threads needed per block
    const uint num_threads_per_block = total_results_per_tile / (ROWS_PER_THREAD * COLS_PER_THREAD);

    // Calculate the srides for loading sharedA and sharedB from GMEM.
    // Threads are assigned across columns and will walk down rows using these strides.
    // At each offset step, threads in a warp access the same row, but different columns,
    // which are contiguous in row-major layout so this achieves coalesced global loads.
    const uint strideA = num_threads_per_block / TILE_SIZE_N;
    const uint strideB = num_threads_per_block / TILE_SIZE_K;

    // Calculate how many tiles we have
    const uint num_tiles = CEIL_DIV(N, TILE_SIZE_N);
    float thread_results[ROWS_PER_THREAD * COLS_PER_THREAD] = {0.0f};
    __nv_bfloat16 reg_m[ROWS_PER_THREAD] = {};
    __nv_bfloat16 reg_k[COLS_PER_THREAD] = {};

    // Outer loop iterate over tiles
    for (int t = 0; t < num_tiles; t++) {
        // Loop to load SMEM tiles 
        for (int load_offset = 0; load_offset < TILE_SIZE_M; load_offset+=strideA) {
            sharedA[(smem_ty_A + load_offset) * TILE_SIZE_N + smem_tx_A] = 
                A[(smem_ty_A + load_offset) * N + smem_tx_A]; // Remember A is already offset at the start of the tile so
                                                             // we only index locally. Thats the beauty of it!!
        }
        for (int load_offset = 0; load_offset < TILE_SIZE_N; load_offset+=strideB) {
            sharedB[(smem_ty_B + load_offset) * TILE_SIZE_K + smem_tx_B] = 
                B[(smem_ty_B + load_offset) * K + smem_tx_B];
        }
        __syncthreads();

        // Outer loop over shared dimension N
        for (int i = 0; i < TILE_SIZE_N; i++) {
            // Load into registers one col from sharedA and one row from sharedB
            for (int row = 0; row < ROWS_PER_THREAD; row++) {
                uint global_smem_row_idx = ty * ROWS_PER_THREAD + row;
                reg_m[row] = sharedA[global_smem_row_idx * TILE_SIZE_N + i];
            }
            for (int col = 0; col < COLS_PER_THREAD; col++) {
                uint global_smem_col_idx = tx * COLS_PER_THREAD + col;
                reg_k[col] = sharedB[i * TILE_SIZE_K + global_smem_col_idx];
            }

            // Calculate outer product between reg_m and reg_k to produce the partial results matrix of the thread 
            for (uint m = 0; m < ROWS_PER_THREAD; m++) {
                float am = __bfloat162float(reg_m[m]);
                for (uint k = 0; k < COLS_PER_THREAD; k++) {
                    thread_results[m * COLS_PER_THREAD + k] += am * __bfloat162float(reg_k[k]); // --> (ROWS_PER_THREAD x COLS_PER_THREAD) matrix
                }
            }
        }
        __syncthreads();

        A += TILE_SIZE_N; // Move right
        B += TILE_SIZE_N * K; // Move down                               
    }
    // Write results of the thread back to C
    for (uint row = 0; row < ROWS_PER_THREAD; row++) {
        for (uint col = 0; col < COLS_PER_THREAD; col++) {
            uint global_row_idx = ty * ROWS_PER_THREAD + row;
            uint global_col_idx = tx * COLS_PER_THREAD + col;
            C[global_row_idx * K + global_col_idx] = __float2bfloat16_rn(
                alpha * thread_results[row * COLS_PER_THREAD + col] + beta * __bfloat162float(C[global_row_idx * K + global_col_idx]));
        }
    }
}



static const int TM_SRC=64, TN_SRC=64, TK_SRC=64, RPT_SRC=8, CPT_SRC=4, NT_SRC=128;

void run_gemm_a100(int64_t A_ptr, int64_t B_ptr, int64_t C_ptr,
                   int M, int K_inner, int N_out)
{
    auto* A = reinterpret_cast<const __nv_bfloat16*>(A_ptr);
    auto* B = reinterpret_cast<const __nv_bfloat16*>(B_ptr);
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_ptr);
    dim3 grid(CEIL_DIV(N_out, TK_SRC), CEIL_DIV(M, TM_SRC));
    dim3 block(NT_SRC);
    gemm_2D_registertiling_bf16<TM_SRC, TN_SRC, TK_SRC, RPT_SRC, CPT_SRC>
        <<<grid, block>>>(A, B, C, M, K_inner, N_out, 1.0f, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
}

"""

_CPP_DECL = """
void run_gemm_a100(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr,
                   int M, int K_inner, int N_out);
"""

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="bf16_gemm_reg2d_a100_to_h100_src",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["run_gemm_a100"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: dict) -> dict:
        A = inputs["A"].contiguous()   # [M, K] BF16
        B = inputs["B"].contiguous()   # [K, N] BF16
        M, K_inner = A.shape
        _, N = B.shape
        # HamzaElshafie kernel takes B[K_inner, N_out] row-major directly
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        _get_ext().run_gemm_a100(
            A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K_inner, N)
        return {"C": C}
