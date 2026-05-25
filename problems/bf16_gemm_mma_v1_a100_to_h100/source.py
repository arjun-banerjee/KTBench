"""
source.py — A100 CUDA kernel for bf16_gemm_mma_v1_a100_to_h100.
Source: https://github.com/gau-nernst/learn-cuda/blob/main/02b_matmul_sm80/matmul_v1.cu
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


// matmul_v1.cu (verbatim from GitHub)
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

template <int TB_SIZE, int HEIGHT, int WIDTH, int OUT_STRIDE>
__device__
void global_to_shared(const nv_bfloat16 *in, int in_stride, nv_bfloat16 *out, int tid) {
  // number of elements to do 128-bit/16-byte load
  // e.g. FP32 -> 4 elements, BF16 -> 8 elements.
  using TLoad = uint4;
  constexpr int num_elems = sizeof(TLoad) / sizeof(nv_bfloat16);

  // NOTE: write loop this way to make sure the compiler can fully unroll it.
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    TLoad tmp = reinterpret_cast<const TLoad *>(in + row * in_stride + col)[0];
    reinterpret_cast<TLoad *>(out + row * OUT_STRIDE + col)[0] = tmp;
  }
}

template <int TB_SIZE, int HEIGHT, int WIDTH, int OUT_STRIDE, bool use_swizzle>
__device__
void global_to_shared_async(const nv_bfloat16 *in, int in_stride, nv_bfloat16 *out, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);  // cp.async cp-size = 16

  // convert to shared state space outside of the loop
  // TODO: move this to kernel body
  uint32_t out_addr = cvta_shared(out);

  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    uint32_t dst_addr = out_addr + (row * OUT_STRIDE + col) * sizeof(nv_bfloat16);
    if constexpr (use_swizzle)
      dst_addr = swizzle<OUT_STRIDE * sizeof(nv_bfloat16)>(dst_addr);
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int SHM_STRIDE, bool use_cp_async,
          bool use_swizzle>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v1_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  constexpr int MMA_K = 16;
  static_assert(BLOCK_M % NUM_WARP_M == 0);
  static_assert(BLOCK_N % NUM_WARP_N == 0);
  static_assert(BLOCK_K % MMA_K == 0);
  constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
  constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  static_assert(use_cp_async || !use_swizzle); // use_swizzle=true requires use_cp_async=true
  constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

  // each warp will do (NUM_MMA_M * NUM_MMA_N) MMAs
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // TODO: threadblock swizzling to improve L2 cache hit rate
  const int num_blocks_n = cdiv(N, BLOCK_N);
  const int bid_m = bid / num_blocks_n;
  const int bid_n = bid % num_blocks_n;
  const int offset_m = bid_m * BLOCK_M;
  const int offset_n = bid_n * BLOCK_N;

  const int warp_id_m = warp_id / NUM_WARP_N;
  const int warp_id_n = warp_id % NUM_WARP_N;

  // A is row-major, B is column-major, C is row-major
  A += offset_m * K;
  B += offset_n * K;
  C += (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

  extern __shared__ nv_bfloat16 shm[];
  nv_bfloat16 *A_shared = shm;                               // BLOCK_M * BLOCK_K
  nv_bfloat16 *B_shared = A_shared + (BLOCK_M * SHM_STRIDE); // BLOCK_N * BLOCK_K

  // all registers are 32-bit (4-byte)
  // - we accumulate to FP32, which is exactly 32-bit
  // - our inputs are FP16/BF16, hence each register holds 2 elements
  // - inputs and accumulate are distributed across 32 threads in a warp
  // for m16n8k8, each thread holds
  // - 4 output float
  // - 4 input A FP16/BF16
  // - 2 input B FP16/BF16
  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE;
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE;
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    if constexpr (use_cp_async) {
      global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K, SHM_STRIDE, use_swizzle>(A, K, A_shared, tid);
      global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K, SHM_STRIDE, use_swizzle>(B, K, B_shared, tid);
      cp_async_wait_all();
    } else {
      global_to_shared<TB_SIZE, BLOCK_M, BLOCK_K, SHM_STRIDE>(A, K, A_shared, tid);
      global_to_shared<TB_SIZE, BLOCK_N, BLOCK_K, SHM_STRIDE>(B, K, B_shared, tid);
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_K; k += MMA_K) {
      // for m16n8k8
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688
      //   A\B   [8x8-0]
      // [8x8-0]
      // [8x8-1]
      // where each [8x8] matrix can be loaded from shared memory with ldmatrix
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix

      // for m16n8k16
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
      //                [8x8-0]
      //       A\B      [8x8-1]
      // [8x8-0][8x8-2]
      // [8x8-1][8x8-3]

      // select the tile this warp is responsible for
      const nv_bfloat16 *A_shm_warp = A_shared + (warp_id_m * WARP_M) * SHM_STRIDE + k;
      const nv_bfloat16 *B_shm_warp = B_shared + (warp_id_n * WARP_N) * SHM_STRIDE + k;

      // to use ldmatrix: each thread holds the address of 1 row e.g.
      // - thread 0 holds address of row 0
      // - thread 1 holds address of row 1, and so on
      // when loading multiple matrices, thread0-7 specifies the 1st matrix,
      // thread 8-15 specifies the 2nd matrix, and so on

      // load B to registers
      uint32_t B_reg[NUM_MMA_N][num_B_regs];
      for (int n = 0; n < NUM_MMA_N; n++) {
        // NOTE: we can reduce unnecessary address calculation if we know MMA_K=8 or 16
        // convert generic address to .shared state space address expected by inline PTX
        const nv_bfloat16 *B_ptr = B_shm_warp + (n * MMA_N + (lane_id % 8)) * SHM_STRIDE + (lane_id / 8) * 8;
        uint32_t B_addr = cvta_shared(B_ptr);
        if constexpr (use_swizzle)
          B_addr = swizzle<SHM_STRIDE * sizeof(nv_bfloat16)>(B_addr);
        ldmatrix_x2(B_reg[n], B_addr);
      }

      for (int m = 0; m < NUM_MMA_M; m++) {
        // load A to registers
        uint32_t A_reg[num_A_regs];
        const nv_bfloat16 *A_ptr = A_shm_warp + (m * MMA_M + (lane_id % 16)) * SHM_STRIDE + (lane_id / 16) * 8;
        uint32_t A_addr = cvta_shared(A_ptr);
        if constexpr (use_swizzle)
          A_addr = swizzle<SHM_STRIDE * sizeof(nv_bfloat16)>(A_addr);
        ldmatrix_x4(A_reg, A_addr);

        // call mma
        for (int n = 0; n < NUM_MMA_N; n++)
          mma_m16n8k16(A_reg, B_reg[n], acc[m][n]);
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  // check output layout here
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
  // NOTE: we can do some warp shuffle to get coalesced write
  for (int m = 0; m < NUM_MMA_M; m++)
    for (int n = 0; n < NUM_MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C + (row + 0) * N + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K; // no padding
  const int use_cp_async = false;
  const int use_swizzle = false;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K; // no padding
  const int use_cp_async = true;
  const int use_swizzle = false;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void matmul_v3(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K + 8; // pad shmem to avoid bank conflict
  const int use_cp_async = true;
  const int use_swizzle = false;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void matmul_v4(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;
  const int SHM_STRIDE = BLOCK_K;
  const int use_cp_async = true;
  const int use_swizzle = true;

  auto kernel =
      matmul_v1_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, SHM_STRIDE, use_cp_async, use_swizzle>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * SHM_STRIDE * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void run_gemm_a100(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr,
                   int M, int K_inner, int N_out)
{
    auto* A   = reinterpret_cast<const nv_bfloat16*>(A_ptr);
    auto* B_T = reinterpret_cast<const nv_bfloat16*>(B_T_ptr);
    auto* C   = reinterpret_cast<nv_bfloat16*>(C_ptr);
    matmul_v2(A, B_T, C, M, N_out, K_inner);
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
            name="bf16_gemm_mma_v1_a100_to_h100_src",
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
