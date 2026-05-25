"""
source.py — A100 CUDA kernel for bf16_gemm_mma_v6_a100_to_h100.
Source: https://github.com/gau-nernst/learn-cuda/blob/main/02b_matmul_sm80/matmul_v6.cu
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cstdint>
#include <iostream>
#include <cuda_bf16.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ ":" << __LINE__ << std::endl;        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

constexpr int WARP_SIZE = 32;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;

// convert generic address (C++ address, 64-bit) to shared state space address (32-bit)
// all PTX instructions expect share memory address to be in shared state space (not 100%)
__device__ inline
uint32_t cvta_shared(const void *ptr) { return static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); }

__device__ inline
void ldmatrix_x2(uint32_t reg[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(reg[0]), "=r"(reg[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t reg[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
              : "r"(addr));
}

__device__ inline
void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, "  // D
               "{%4, %5, %6, %7}, "  // A
               "{%8, %9}, "          // B
               "{%0, %1, %2, %3};"   // C
              : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]));
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy
__device__ inline
void cp_async(uint32_t dst, const void *src) {
  // .ca means cache to L1 and L2. .cg means cache to L2 only.
  // .cg only accepts cp-size=16
  // .ca results in significantly slower kernel, probably because it uses up L1 resources
  // + additional copy, which is unnecessary, since we already manually cache it in shared memory.
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
};

__device__ inline
void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); };

template <int N>
__device__ inline
void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); };

__device__ inline
void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); };

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / std::max(128 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

// STRIDE in bytes, col in the units of 16-byte
template <int STRIDE>
__device__ static
uint32_t swizzle_better(uint32_t row, uint32_t col) {
  if constexpr (STRIDE >= 128)
    col ^= (row % 8) / std::max(128 / STRIDE, 1);
  return row * STRIDE + col * 16;
}

template <typename T, typename... Args>
void launch_kernel(T *kernel, int num_blocks, int block_size, int shm_size, Args... args) {
  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<num_blocks, block_size, shm_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}


// matmul_v6.cu (verbatim from GitHub)
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

template <int TB_SIZE, int HEIGHT, int WIDTH>
__device__ static
void global_to_shared_async(const nv_bfloat16 *in, int in_stride, uint32_t out, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    // NOTE: perhaps we can move swizzle out of this loop as well
    uint32_t dst_addr = out + swizzle_better<WIDTH * sizeof(nv_bfloat16)>(row, col / num_elems);
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, int GROUP_M>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v6_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  constexpr int MMA_K = 16;
  static_assert(BLOCK_M % NUM_WARP_M == 0);
  static_assert(BLOCK_N % NUM_WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;
  constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  const int grid_m = cdiv(M, BLOCK_M);
  const int grid_n = cdiv(N, BLOCK_N);
  int bid_m, bid_n;

  if constexpr (GROUP_M == 0) {
    // no swizzling
    bid_m = bid / grid_n;
    bid_n = bid % grid_n;
  }
  else {
    // threadblock swizzling to improve L2 cache hit rate
    // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    // each group is [GROUP_M, grid_n], tile from top (small M) to bottom (large M).
    // the last group might be shorter than GROUP_M if grid_m % GROUP_M != 0.
    const int group_size = GROUP_M * grid_n;
    const int group_id = bid / group_size;
    const int group_off_m = group_id * GROUP_M;
    const int group_m = std::min(grid_m - group_off_m, GROUP_M);  // actual group height

    bid_m = group_off_m + ((bid % group_size) % group_m);
    bid_n = (bid % group_size) / group_m;
  }

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  // A is row-major, B is column-major, C is row-major
  A += off_m * K;
  B += off_n * K;
  C += (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

  constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);
  constexpr int AB_size = A_size + B_size;

  // convert shared memory address to 32-bit from the start
  extern __shared__ nv_bfloat16 shm[];
  const uint32_t shm_u32 = cvta_shared(shm);
  const uint32_t A_shm = shm_u32;
  const uint32_t B_shm = A_shm + A_size;

  uint32_t A_regs[NUM_MMA_K][NUM_MMA_M][4];
  uint32_t B_regs[NUM_MMA_K][NUM_MMA_N][2];
  float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K * sizeof(nv_bfloat16)>(A_offm, lane_id / 16);

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
  const uint32_t B_shm_thread = B_shm + swizzle_better<BLOCK_K * sizeof(nv_bfloat16)>(B_offn, (lane_id % 16) / 8);

  const int num_k_iters = cdiv(K, BLOCK_K);

  auto load_AB = [&](int k_iter) {
    // select the correct shared memory buffer
    const int stage_id = k_iter % NUM_STAGES;
    global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shm + stage_id * AB_size, tid);
    global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shm + stage_id * AB_size, tid);

    // A/B pointer tracks position for global->shared load
    A += BLOCK_K;
    B += BLOCK_K;
    cp_async_commit_group();
  };

  auto compute = [&](int k_iter) {
    // A shared->regs
    for (int k = 0; k < NUM_MMA_K; k++)
      for (int m = 0; m < NUM_MMA_M; m++) {
        uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
        A_addr += m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_regs[k][m], A_addr ^ (k * 32));
      }

    // B shared->regs
    for (int k = 0; k < NUM_MMA_K; k++)
      for (int n = 0; n < NUM_MMA_N; n += 2) {
        uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
        B_addr += n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_regs[k][n], B_addr ^ (k * 32));
      }

    // do MMA
    for (int k = 0; k < NUM_MMA_K; k++)
      for (int m = 0; m < NUM_MMA_M; m++)
        for (int n = 0; n < NUM_MMA_N; n++)
          mma_m16n8k16(A_regs[k][m], B_regs[k][n], acc[m][n]);
  };

  // initiate NUM_STAGES-1 stages
  for (int stage = 0; stage < NUM_STAGES - 1; stage++)
    load_AB(stage);

  // loop invariance: there is always NUM_STAGES - 1 prefetch stages in-flight
  for (int k_iter = 0; k_iter < num_k_iters - (NUM_STAGES - 1); k_iter++) {
    // wait for previous MMA to finish using the shared buffer
    __syncthreads();

    // prefetch the next stage. add 1 more stage to the pipeline
    load_AB(k_iter + NUM_STAGES - 1);

    // wait for the 1st stage to finish. remove 1 stage from the pipeline
    // -> restore loop invariance
    cp_async_wait_group<NUM_STAGES - 1>();
    __syncthreads();

    // ldmatrix and mma
    compute(k_iter);
  }

  for (int k_iter = num_k_iters - (NUM_STAGES - 1); k_iter < num_k_iters; k_iter++) {
    // preserve invariance of cp.async commited groups
    cp_async_commit_group();

    // wait cp.async
    cp_async_wait_group<NUM_STAGES - 1>();
    __syncthreads();

    // ldmatrix and mma
    compute(k_iter);
  }

  for (int m = 0; m < NUM_MMA_M; m++)
    for (int n = 0; n < NUM_MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C + ((row + 0) * N + col))[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C + ((row + 8) * N + col))[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v6(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  const int GROUP_M = 0;
  auto kernel = matmul_v6_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES, GROUP_M>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void matmul_v6b(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int NUM_STAGES = 2;

  const int GROUP_M = 8;
  auto kernel = matmul_v6_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES, GROUP_M>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void run_gemm_a100(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr,
                   int M, int K_inner, int N_out)
{
    auto* A   = reinterpret_cast<const nv_bfloat16*>(A_ptr);
    auto* B_T = reinterpret_cast<const nv_bfloat16*>(B_T_ptr);
    auto* C   = reinterpret_cast<nv_bfloat16*>(C_ptr);
    matmul_v6(A, B_T, C, M, N_out, K_inner);
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
            name="bf16_gemm_mma_v6_a100_to_h100_src",
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
        B_T = B.t().contiguous()  # [N, K] row-major for gau-nernst convention
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        _get_ext().run_gemm_a100(
            A.data_ptr(), B_T.data_ptr(), C.data_ptr(), M, K_inner, N)
        return {"C": C}
