"""
source.py — A100 CUDA kernel for bf16_gemm_mma_v5_a100_to_h100.
Source: https://github.com/gau-nernst/learn-cuda/blob/main/02b_matmul_sm80/matmul_v5.cu
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


// matmul_v5.cu (verbatim from GitHub)
#include <assert.h>
#include <cstdint>
#include <cuda_bf16.h>

template <int TB_SIZE, int HEIGHT, int WIDTH>
__device__
void global_to_shared_async(const nv_bfloat16 *in, int in_stride, uint32_t out, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    // NOTE: perhaps we can move swizzle out of this loop as well
    uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(out + (row * WIDTH + col) * sizeof(nv_bfloat16));
    cp_async(dst_addr, in + row * in_stride + col);
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE) // maxThreadsPerBlock
__global__
void matmul_v5_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
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

  // convert shared memory address to 32-bit from the start
  extern __shared__ nv_bfloat16 shm[];
  const uint32_t A_shared = cvta_shared(shm);                                     // BLOCK_M * BLOCK_K
  const uint32_t B_shared = A_shared + (BLOCK_M * BLOCK_K) * sizeof(nv_bfloat16); // BLOCK_N * BLOCK_K

  constexpr int num_acc_regs = MMA_M * MMA_N / WARP_SIZE;
  constexpr int num_A_regs = MMA_M * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  constexpr int num_B_regs = MMA_N * MMA_K * sizeof(nv_bfloat16) / 4 / WARP_SIZE; // 4
  float acc[NUM_MMA_M][NUM_MMA_N][num_acc_regs] = {};

  // pre-compute address used for ldmatrix
  // also pre-compute swizzling
  const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
  const int A_offk = (lane_id / 16) * 8;
  const uint32_t A_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(A_shared + (A_offm * BLOCK_K + A_offk) * sizeof(nv_bfloat16));

  const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
  const int B_offk = ((lane_id % 16) / 8) * 8;
  const uint32_t B_shm_thread = swizzle<BLOCK_K * sizeof(nv_bfloat16)>(B_shared + (B_offn * BLOCK_K + B_offk) * sizeof(nv_bfloat16));

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(A, K, A_shared, tid);
    global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(B, K, B_shared, tid);
    cp_async_wait_all();
    __syncthreads();

    for (int k = 0; k < NUM_MMA_K; k++) {
      // iterate MMA_K=16 -> increment bit5 (32 bytes) -> affects swizzled bits
      // assume we have alignment (bit0-6 are all zeros), increment bit5
      // is equivalent to XOR mma_id_k directly, which is commutative with swizzling
      // -> we can move swizzling outside of this loop
      // the kernel compiles to fewer instructions, but no speedup

      // load B to registers
      uint32_t B_reg[NUM_MMA_N][num_B_regs];
      for (int n = 0; n < NUM_MMA_N; n += 2) {
        const uint32_t B_addr = B_shm_thread + n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(B_reg[n], B_addr ^ (k * MMA_K * sizeof(nv_bfloat16)));
      }

      for (int m = 0; m < NUM_MMA_M; m++) {
        // load A to registers
        uint32_t A_reg[num_A_regs];
        const uint32_t A_addr = A_shm_thread + m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
        ldmatrix_x4(A_reg, A_addr ^ (k * MMA_K * sizeof(nv_bfloat16)));

        // call mma
        for (int n = 0; n < NUM_MMA_N; n++)
          mma_m16n8k16(A_reg, B_reg[n], acc[m][n]);
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K;
  }

  for (int m = 0; m < NUM_MMA_M; m++)
    for (int n = 0; n < NUM_MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;

      float *regs = acc[m][n];
      reinterpret_cast<nv_bfloat162 *>(C + (row + 0) * N + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(C + (row + 8) * N + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

void matmul_v5(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  // 4 warps
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
  const int NUM_WARP_M = 2, NUM_WARP_N = 2;

  auto kernel = matmul_v5_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N>;

  const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
  const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);

  launch_kernel(kernel, grid_size, TB_SIZE, shm_size, A, B, C, M, N, K);
}

void run_gemm_a100(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr,
                   int M, int K_inner, int N_out)
{
    auto* A   = reinterpret_cast<const nv_bfloat16*>(A_ptr);
    auto* B_T = reinterpret_cast<const nv_bfloat16*>(B_T_ptr);
    auto* C   = reinterpret_cast<nv_bfloat16*>(C_ptr);
    // matmul_v5(A, B[N,K], C, M, N, K): B_T is [N_out, K_inner] row-major.
    // kernel computes C = A @ B_T^T = A @ B. ✓
    matmul_v5(A, B_T, C, M, N_out, K_inner);
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
            name="bf16_gemm_mma_v5_a100_to_h100_src",
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
