"""
reference_tgt.py — H100 CUDA kernel for gemm_v1_a100_to_h100.
Source: https://github.com/HamzaElshafie/h100_gemm/blob/main/src/kernels/hopper/gemm_bf16_wgmma_tma.cuh
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

using bf16 = __nv_bfloat16;

// hopper_wgmma_utils.cuh (verbatim from GitHub)
/**
 * @file hopper_wgmma_utils.cuh
 *
 * @brief Host-side utilities for Tensor Memory Accelerator (TMA) on Hopper.
 */
#ifndef HOPPER_WGMMA_UTILS_CUH
#define HOPPER_WGMMA_UTILS_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <type_traits>
#include <cuda_bf16.h> // for __nv_bfloat16
#include <cuda/barrier>


/**
 * @brief Encodes a matrix descriptor value for WGMMA operations.
 *
 * Extracts and encodes the relevant bits from a 64-bit value for use in WGMMA matrix descriptors.
 * Masks the lower 18 bits and shifts right by 4 positions.
 *
 * @cite https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
 *
 * @param x The 64-bit value to encode.
 * @return The encoded descriptor value.
 */
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return ((x) & 0x3FFFF) >> 4;
}

/**
 * @brief Creates a WGMMA matrix descriptor for shared memory access.
 *
 * Constructs a 64-bit descriptor that encodes the layout and access pattern for a matrix
 * stored in shared memory. The descriptor specifies the base address, leading dimension,
 * stride dimension, and swizzle mode for WGMMA matrix multiply operations.
 *
 * @cite https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
 *
 * @param ptr Pointer to the matrix data in shared memory.
 * @return A 64-bit WGMMA matrix descriptor with:
 *         - Bits [13:0]:  Encoded matrix start address
 *         - Bits [29:16]: Leading dimension byte offset (16 bytes)
 *         - Bits [45:32]: Stride dimension byte offset (1024 bytes)
 *         - Bits [62:63]: Swizzle mode (128B swizzle)
 */
__device__ uint64_t make_smem_desc(bf16* ptr) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    // Initialise an empty 64 bit descriptor
    uint64_t desc = 0x0000000000000000;
    // bitwise OR
    // sets bits [13:0] encoded matrix start address
    desc |= matrix_descriptor_encode(address);
    // sets bits [29:16] leading dimension byte offset
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(16)) << 16;
    // sets bits [45: 32] stride dimension byte offset
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32;
    // sets bits [62: 63] swizzle mode
    desc |= 1llu << 62;
    return desc;
}

/**
 * @cite https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
 */
__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/**
 * @cite https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group
 */
__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @cite https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group
 */
template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

/**
 * @brief Increase max register count for the warp group (e.g. for consumer warp groups doing WGMMA).
 * @cite PTX setmaxnreg.inc.sync.aligned
 */
template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

/**
 * @brief Decrease max register count for the warp group (e.g. for producer warp group to free regs).
 * @cite PTX setmaxnreg.dec.sync.aligned
 */
template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

/**
 * @note Accumulator floats per thread (per one WGMMA issue) = (WGMMA_M * WGMMA_N) / NUM_THREADS
 */
template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma32(float d[2][8], bf16* sharedA, bf16*sharedB) {
    uint64_t desc_a = make_smem_desc(&sharedA[0]);
    uint64_t desc_b = make_smem_desc(&sharedB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
        " %16,"
        " %17,"
        " %18, %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @note Accumulator floats per thread (per one WGMMA issue) = (WGMMA_M * WGMMA_N) / NUM_THREADS
 */
template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma64(float d[4][8], bf16 *sharedA, bf16 *sharedB)
{
    uint64_t desc_a = make_smem_desc(&sharedA[0]);
    uint64_t desc_b = make_smem_desc(&sharedB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @note Accumulator floats per thread (per one WGMMA issue) = (WGMMA_M * WGMMA_N) / NUM_THREADS
 */
template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma128(float d[8][8], bf16 *sharedA, bf16 *sharedB) {
    uint64_t desc_a = make_smem_desc(&sharedA[0]);
    uint64_t desc_b = make_smem_desc(&sharedB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
          "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
          "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @note Accumulator floats per thread (per one WGMMA issue) = (WGMMA_M * WGMMA_N) / NUM_THREADS
 */
template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma192(float d[12][8], bf16* sharedA, bf16* sharedB) {
    uint64_t desc_a = make_smem_desc(&sharedA[0]);
    uint64_t desc_b = make_smem_desc(&sharedB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},  "
        " %96,"
        " %97,"
        " %98,    %99,  %100,  %101,  %102;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
          "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
          "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @note Accumulator floats per thread (per one WGMMA issue) = (WGMMA_M * WGMMA_N) / NUM_THREADS
 */
template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma256(float d[16][8], bf16* sharedA, bf16* sharedB) {
    uint64_t desc_a = make_smem_desc(&sharedA[0]);
    uint64_t desc_b = make_smem_desc(&sharedB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
          "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
          "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
          "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
          "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
          "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
          "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * Compile-time dispatcher that selects the correct WGMMA instruction variant based on WGMMA_N and forwards all MMA parameters.
 */
template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N / 16][8], bf16 *sharedA, bf16 *sharedB){
    if constexpr (WGMMA_N == 32) {wgmma32<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sharedA, sharedB);}
    if constexpr (WGMMA_N == 64){wgmma64<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sharedA, sharedB);}
    if constexpr (WGMMA_N == 128){wgmma128<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sharedA, sharedB);}
    if constexpr (WGMMA_N == 192){wgmma192<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sharedA, sharedB);}
    if constexpr (WGMMA_N == 256){wgmma256<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sharedA, sharedB);}
}

#endif


// hopper_tma_utils.h (verbatim from GitHub)
/**
 * @file hopper_tma_utils.h
 *
 * @brief Host-side utilities for Tensor Memory Accelerator (TMA) on Hopper.
 */
#ifndef HOPPER_TMA_UTILS_H
#define HOPPER_TMA_UTILS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <type_traits>
#include <cuda_bf16.h> // for __nv_bfloat16


/**
 * @brief Creates a CUDA tensor map descriptor for TMA operations.
 *
 * This function encodes the layout and properties of a tensor in global memory into a CUtensorMap structure,
 * using the CUDA Driver API. The tensor map is required for Tensor Memory Accelerator (TMA) asynchronous memory operations
 * on Hopper.
 *
 * @cite https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
 *
 * @tparam BlockMajorSize   The tile size in the major (slowest-changing) dimension.
 * @tparam BlockMinorSize   The tile size in the minor (fastest-changing) dimension.
 * @param tensor_map        Pointer to the CUtensorMap structure to be filled.
 * @param tensor_ptr        Pointer to the tensor data in global memory.
 * @param blocks_height     Number of tiles in the major (slowest-changing) dimension. Each tile has size BlockMajorSize.
 * @param blocks_width      Number of tiles in the minor (fastest-changing) dimension. Each tile has size BlockMinorSize.
 */
template <const uint BlockMajorSize, const uint BlockMinorSize>
void create_tensor_map(CUtensorMap *tensor_map, bf16 *tensor_ptr, uint blocks_height, uint blocks_width) {
    // Starting address of memory region described by tensor (casting to void
    // as the tensor map descriptor is type-agnostic.)
    void *gmem_address = static_cast<void *>(tensor_ptr);

    uint num_tiles_major = blocks_height;
    uint num_tiles_minor = blocks_width;

    // full size of the tensor in global memory (API expects the 5D supported
    // tensor ranks to be defined)
    uint64_t global_dim[5] = {
        static_cast<uint64_t>(BlockMinorSize * num_tiles_minor),
        static_cast<uint64_t>(BlockMajorSize * num_tiles_major),
        1, 1, 1};

    // Define the tensor strides (in bytes) along each of the tensor ranks dims - 1
    uint64_t global_strides[5] = {
        sizeof(bf16),
        sizeof(bf16) * BlockMinorSize * num_tiles_minor,
        0, 0, 0};

    // Define the shape of the "box_size" -> the tile shapes a TMA ops will load
    uint32_t box_dim[5] = {
        static_cast<uint32_t>(BlockMinorSize),
        static_cast<uint32_t>(BlockMajorSize),
        1, 1, 1};

    uint32_t elem_strides[5] = {1, 1, 1, 1, 1};

    // Create tensor map
    CU_CHECK(cuTensorMapEncodeTiled(
        tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address,
        global_dim, global_strides + 1, box_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

/**
 * @brief Allocates device memory and creates a CUDA tensor map descriptor for TMA operations.
 *
 * This function allocates device memory for a CUtensorMap descriptor, creates the tensor map on the host
 * using the provided tensor pointer and dimensions, and copies the descriptor to the device. It internally
 * calls create_tensor_map() to perform the host-side descriptor creation.
 *
 * @tparam BlockMajorSize   The tile size in the major (slowest-changing) dimension.
 * @tparam BlockMinorSize   The tile size in the minor (fastest-changing) dimension.
 * @param tensor_ptr        Pointer to the tensor data in global memory.
 * @param blocks_height     Number of tiles in the major (slowest-changing) dimension.
 * @param blocks_width      Number of tiles in the minor (fastest-changing) dimension.
 * @return                  Device pointer to the allocated and initialized CUtensorMap descriptor.
 */
template <const uint BlockMajorSize, const uint BlockMinorSize>
__host__ static inline CUtensorMap*
create_and_allocate_tensor_map(bf16 *tensor_ptr, uint blocks_height, uint blocks_width) {
    CUtensorMap *tensor_map;
    // Allocate device memory for the tensor map descriptor.
    CUDA_CHECK(cudaMalloc((void **)&tensor_map, sizeof(CUtensorMap)));
    // Register the tensorMap in our device memory pointers
    // resources.add_device_ptr(tensor_map);
    // Create on host
    CUtensorMap tensor_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tensor_map_host, tensor_ptr, blocks_height, blocks_width);
    // Copy descriptor to device
    CUDA_CHECK(cudaMemcpy(tensor_map, &tensor_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    return tensor_map;
}

#endif
// gemm_bf16_wgmma_tma.cuh (verbatim from GitHub)
#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template <const uint TILE_SIZE_M, const uint TILE_SIZE_K, const uint TILE_SIZE_N,
          const uint WGMMA_M, const uint WGMMA_K, const uint WGMMA_N, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_bf16_wgmma_tma(const CUtensorMap* __restrict__ tensorMapA, const CUtensorMap* __restrict__ tensorMapB, bf16* __restrict__ C,
    int M, int N, int K, float alpha, float beta) {
    // Allocate SMEM
    __shared__ alignas(128) bf16 sharedA[TILE_SIZE_M * TILE_SIZE_K];
    __shared__ alignas(128) bf16 sharedB[TILE_SIZE_K * TILE_SIZE_N];
    // Initialise thread's accumilator
    // d[4][8] = 32 floats per thread
    float d[WGMMA_N / WGMMA_K][8];
    memset(d, 0, sizeof(d));

    const int num_blocks_k = CEIL_DIV(K, TILE_SIZE_K);
    int num_block_n = blockIdx.x % CEIL_DIV(N, TILE_SIZE_N);
    int num_block_m = blockIdx.x / CEIL_DIV(N, TILE_SIZE_N);

    // SMEM barriers for A and B
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA; 
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        // A single thread initializes the total expected arrival count.
        // barrier expects blockDim.x (=N) arrivals before it is released. This is the countdown counter the
        // async barrier tracks.
        // @cite https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#a-barrier-s-phase-arrival-countdown-completion-and-reset
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; block_k_iter++) {
        // Async loads (Only 1 thread launches the TMA op)
        if (threadIdx.x == 0) {
            // Thread 0 launches async bulk tensor copy operations for both matrices
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sharedA[0], tensorMapA, block_k_iter * TILE_SIZE_K, num_block_m * TILE_SIZE_M, barA);
            // Signal barrier and wait for both loads to complete
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sharedA));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sharedB[0], tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sharedB));
        }
        else {
            // Other threads arrive at barrier to synchronise data loads
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        // All threads wait for async loads to complete
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        // Compute phase using WGMMA tensor cores
        warpgroup_arrive(); // asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[0], &sharedB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[WGMMA_K], &sharedB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[2 * WGMMA_K], &sharedB[2 * WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sharedA[3 * WGMMA_K], &sharedB[3 * WGMMA_K]);
        warpgroup_commit_batch(); // asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        warpgroup_wait<0>();      // asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
    }

    // Store results from accumulator to global memory
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    uint32_t row = warp * 16 + lane / 4;
    bf16 *block_C = C + num_block_n * TILE_SIZE_N * M + num_block_m * TILE_SIZE_M;

    for (int m_it = 0; m_it < TILE_SIZE_M / WGMMA_M; ++m_it) {
        for (int n_it = 0; n_it < TILE_SIZE_N / WGMMA_N; ++n_it) {
            for (int w = 0; w < WGMMA_N / 16; ++w) { // w = {0, 1, 2, 3}
                // (16 * w) selects the base col of the 16 col block
                int col = 16 * w + 2 * (tid % 4);
                #define IDX(i, j) ((j + n_it * WGMMA_N) * M + ((i) + m_it * WGMMA_M))

                // Apply alpha scaling to accumulator results and add beta*C
                block_C[IDX(row, col)] = __float2bfloat16(alpha * d[w][0] + beta * __bfloat162float(block_C[IDX(row, col)]));
                block_C[IDX(row, col + 1)] = __float2bfloat16(alpha * d[w][1] + beta * __bfloat162float(block_C[IDX(row, col + 1)]));
                block_C[IDX(row + 8, col)] = __float2bfloat16(alpha * d[w][2] + beta * __bfloat162float(block_C[IDX(row + 8, col)]));
                block_C[IDX(row + 8, col + 1)] = __float2bfloat16(alpha * d[w][3] + beta * __bfloat162float(block_C[IDX(row + 8, col + 1)]));

                block_C[IDX(row, col + 8)] = __float2bfloat16(alpha * d[w][4] + beta * __bfloat162float(block_C[IDX(row, col + 8)]));
                block_C[IDX(row, col + 9)] = __float2bfloat16(alpha * d[w][5] + beta * __bfloat162float(block_C[IDX(row, col + 9)]));
                block_C[IDX(row + 8, col + 8)] = __float2bfloat16(alpha * d[w][6] + beta * __bfloat162float(block_C[IDX(row + 8, col + 8)]));
                block_C[IDX(row + 8, col + 9)] = __float2bfloat16(alpha * d[w][7] + beta * __bfloat162float(block_C[IDX(row + 8, col + 9)]));

                #undef IDX
            }
        }
    } 
}
static CUtensorMap make_tma_map_host(const void* ptr, uint64_t rows, uint64_t cols,
                                      uint32_t tile_rows, uint32_t tile_cols) {
    void* gmem = const_cast<void*>(ptr);
    uint64_t global_dim[2]    = {cols, rows};
    uint64_t global_stride[1] = {sizeof(bf16) * cols};
    uint32_t box_dim[2]       = {tile_cols, tile_rows};
    uint32_t elem_stride[2]   = {1, 1};
    CUtensorMap tensor_map;
    CU_CHECK(cuTensorMapEncodeTiled(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem,
        global_dim, global_stride, box_dim, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static const int TM_REF=64, TK_REF=64, TN_REF=64, NT_REF=128;

void run_gemm_h100(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr,
                   int M, int K_inner, int N_out)
{
    auto* A   = reinterpret_cast<const bf16*>(A_ptr);
    auto* B_T = reinterpret_cast<const bf16*>(B_T_ptr);
    auto* C   = reinterpret_cast<bf16*>(C_ptr);
    CUtensorMap mapA = make_tma_map_host(A,   M,     K_inner, TM_REF, TK_REF);
    CUtensorMap mapB = make_tma_map_host(B_T, N_out, K_inner, TN_REF, TK_REF);
    CUtensorMap *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_A, &mapA, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, &mapB, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    int nb = (M / TM_REF) * (N_out / TN_REF);
    gemm_bf16_wgmma_tma<TM_REF, TK_REF, TN_REF, 64, 16, 64, NT_REF>
        <<<nb, NT_REF>>>(d_A, d_B, C, M, N_out, K_inner, 1.0f, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
}

"""

_CPP_DECL = """
void run_gemm_h100(int64_t A_ptr, int64_t B_T_ptr, int64_t C_ptr,
                   int M, int K_inner, int N_out);
"""

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="gemm_v1_a100_to_h100_ref",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["run_gemm_h100"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],
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
        # H100 kernel expects B^T in [N, K] row-major layout
        B_T = B.t().contiguous()
        # H100 stores C column-major: C[m,n] at flat offset n*M + m
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        C.zero_()
        _get_ext().run_gemm_h100(
            A.data_ptr(), B_T.data_ptr(), C.data_ptr(), M, K_inner, N)
        return C
