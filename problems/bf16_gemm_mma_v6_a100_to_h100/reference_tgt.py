"""
reference_tgt.py — H100 CUDA kernel for bf16_gemm_mma_v6_a100_to_h100.
Source: https://github.com/HamzaElshafie/h100_gemm/blob/main/src/kernels/hopper/gemm_bf16_pc_pipeline.cuh
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
// gemm_bf16_pc_pipeline.cuh (verbatim from GitHub)
#pragma once

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template <int TILE_SIZE_M, int TILE_SIZE_K, int TILE_SIZE_N, int NUM_STAGES>
struct Smem {
    alignas(128) bf16 A[TILE_SIZE_M * TILE_SIZE_K * NUM_STAGES];
    alignas(128) bf16 B[TILE_SIZE_K * TILE_SIZE_N * NUM_STAGES];

    static constexpr int TILE_M_PAD = TILE_SIZE_M + 8;
    // Epilogue staging tile (padded)
    alignas(128) bf16 C_epi[TILE_M_PAD * TILE_SIZE_N];
};

template <const int TILE_SIZE_M, const int TILE_SIZE_K, const int TILE_SIZE_N,
          const int WGMMA_M, const int WGMMA_N, const int WGMMA_K, const int NUM_THREADS,
          const int NUM_STAGES = 5>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_bf16_pc_pipeline(CUtensorMap* tensorMapA, CUtensorMap* tensorMapB, bf16* C,
    int M, int K, int N, float alpha, float beta) {
        static_assert(WGMMA_N == TILE_SIZE_N, "WGMMA_N must be == TILE_SIZE_N");
        static_assert(TILE_SIZE_M % WGMMA_M == 0, "TILE_SIZE_M must be divisible by WGMMA_M");
        static_assert(TILE_SIZE_K % WGMMA_K == 0, "TILE_SIZE_K must be divisible by WGMMA_K");
        static_assert(TILE_SIZE_N % WGMMA_N == 0, "TILE_SIZE_N must be divisible by WGMMA_N");
        static_assert(NUM_THREADS % 128 == 0, "NUM_THREADS must be divisible by warp group size (128)");
        static_assert(NUM_THREADS >= 256, "Need at least 2 warp groups (1 producer + 1 consumer)");

        // Allocate SMEM
        extern __shared__ __align__(128) uint8_t smem_raw[];
        Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES> &s =
            *reinterpret_cast<Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES>*>(smem_raw);

        constexpr int TILE_M_PAD = Smem<TILE_SIZE_M, TILE_SIZE_K, TILE_SIZE_N, NUM_STAGES>::TILE_M_PAD;

        constexpr int A_stage_size = TILE_SIZE_M * TILE_SIZE_K;
        constexpr int B_stage_size = TILE_SIZE_K * TILE_SIZE_N;

        // How many warp groups in the block?
        constexpr int num_warp_groups = NUM_THREADS / 128;
        constexpr int num_consumer_groups = num_warp_groups - 1; // only 1 producer

        int warp_group_idx = threadIdx.x / 128;
        bool is_producer = (warp_group_idx == 0);

        // How many M rows of the output tile each 'consumer' warp group is responsible for
        // @example: TILE_SIZE_M = 128, num_consumer_groups = 1 -> 128 rows; num_consumer_groups = 2 -> 64 rows each
        constexpr int rows_per_consumer_warp_group = TILE_SIZE_M / num_consumer_groups;

        // Consumer warp group index (0-indexed among consumers only)
        int consumer_warp_group_idx = is_producer ? -1 : (warp_group_idx - 1);

        const int num_blocks_k = CEIL_DIV(K, TILE_SIZE_K);
        int num_block_m = blockIdx.x / CEIL_DIV(N, TILE_SIZE_N);
        int num_block_n = blockIdx.x % CEIL_DIV(N, TILE_SIZE_N);

        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__ barrier full[NUM_STAGES];  // Signals data is ready
        __shared__ barrier empty[NUM_STAGES]; // Signals slot is available

        if (threadIdx.x == 0) {
            for (int i = 0; i < NUM_STAGES; i++) {
                init(&full[i], num_consumer_groups * 128 + 1); // consumers + producer thread 0
                init(&empty[i], num_consumer_groups * 128 + 1);
            }
            cde::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        if (is_producer) {
            constexpr int num_regs_producer = (num_consumer_groups <= 2 ? 24 : 32);
            warpgroup_reg_dealloc<num_regs_producer>();
            // Producer warp group: Issues TMA loads
            if (threadIdx.x == 0) {
                // Fill the pipeline
                for (int stage = 0; stage < NUM_STAGES && stage < num_blocks_k; stage++) {
                    int block_k_iter = stage;
                    
                    // Wait for empty slot (initially all are empty, so this passes immediately)
                    empty[stage].wait(empty[stage].arrive());

                    // Get pointers for this stage in the flat arrays
                    bf16* A_stage = s.A + (stage * A_stage_size);
                    bf16* B_stage = s.B + (stage * B_stage_size);

                    // TMA loads for A and B
                    cde::cp_async_bulk_tensor_2d_global_to_shared(A_stage, tensorMapA, block_k_iter * TILE_SIZE_K, num_block_m * TILE_SIZE_M, full[stage]);
                    cde::cp_async_bulk_tensor_2d_global_to_shared(B_stage, tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, full[stage]);

                    // Signal data is ready
                    barrier::arrival_token token = cuda::device::barrier_arrive_tx(full[stage], 1, A_stage_size * sizeof(bf16) + B_stage_size * sizeof(bf16));
                }

                // Main loop: Continue issuing loads
                for (int block_k_iter = NUM_STAGES; block_k_iter < num_blocks_k; block_k_iter++) {
                    int stage = block_k_iter % NUM_STAGES;
                    
                    // Wait for this stage to be empty before overwriting
                    empty[stage].wait(empty[stage].arrive());

                    // Get pointers for this stage in the flat arrays
                    bf16* A_stage = s.A + (stage * A_stage_size);
                    bf16* B_stage = s.B + (stage * B_stage_size);

                    // Issue next TMA loads
                    cde::cp_async_bulk_tensor_2d_global_to_shared(A_stage, tensorMapA, block_k_iter * TILE_SIZE_K, num_block_m * TILE_SIZE_M, full[stage]);
                    cde::cp_async_bulk_tensor_2d_global_to_shared(B_stage, tensorMapB, block_k_iter * TILE_SIZE_K, num_block_n * TILE_SIZE_N, full[stage]);

                    // Signal data is ready
                    barrier::arrival_token token = cuda::device::barrier_arrive_tx(full[stage], 1, A_stage_size * sizeof(bf16) + B_stage_size * sizeof(bf16));
                }
            }
            
        } else {
            constexpr int num_regs_consumer = (num_consumer_groups == 1 ? 256 : (num_consumer_groups == 2 ? 240 : 160));
            warpgroup_reg_alloc<num_regs_consumer>();
            // Consumer warp groups: Execute WGMMA compute
            // Accumulator registers - declared inside consumer branch only so
            // ptxas doesn't allocate them for the producer warp group
            float d[TILE_SIZE_M / WGMMA_M / num_consumer_groups][WGMMA_N / 16][8];
            memset(d, 0, sizeof(d));

            // Initially signal all empty slots are available
            for (int i = 0; i < NUM_STAGES; i++) {
                barrier::arrival_token token = empty[i].arrive();
            }

            // Main compute loop
            for (int block_k_iter = 0; block_k_iter < num_blocks_k; block_k_iter++) {
                int stage = block_k_iter % NUM_STAGES;
                
                // Get pointers for this stage in the flat arrays
                bf16* A_stage = s.A + (stage * A_stage_size);
                bf16* B_stage = s.B + (stage * B_stage_size);
                
                // Wait for data to be ready
                full[stage].arrive_and_wait();

                // Compute phase using WGMMA
                warpgroup_arrive();
                
                #pragma unroll
                for (int m_iter = 0; m_iter < rows_per_consumer_warp_group / WGMMA_M; m_iter++) {
                    bf16* sharedA_wgmma_tile_base = A_stage + ((consumer_warp_group_idx * rows_per_consumer_warp_group) + (m_iter * WGMMA_M)) * TILE_SIZE_K;
                    
                    #pragma unroll
                    for (int k_iter = 0; k_iter < TILE_SIZE_K / WGMMA_K; k_iter++) {
                        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_iter], &sharedA_wgmma_tile_base[k_iter * WGMMA_K], &B_stage[k_iter * WGMMA_K]);
                    }
                }
                
                warpgroup_commit_batch();
                warpgroup_wait<0>();

                // Signal this slot is now empty and can be reused
                barrier::arrival_token empty_token = empty[stage].arrive();
            }

            int tid  = threadIdx.x % 128;
            int lane = tid % 32;
            int warp = tid / 32;
            uint32_t row = warp * 16 + lane / 4;

            // @note C is column-major
            bf16* block_C = C + (num_block_n * TILE_SIZE_N * M) + (num_block_m * TILE_SIZE_M);

            constexpr int TILE_M_PAD = TILE_SIZE_M + 8;
            #define IDX_GMEM(r, c) ((c) * M + (r))
            #define IDX_SMEM(r, c) ((c) * TILE_M_PAD + (r))

            #pragma unroll
            // Phase 1: alpha-scaled accumulators -> shared staging tile
            for (int m_iter = 0; m_iter < rows_per_consumer_warp_group / WGMMA_M; m_iter++) {
                int row_tile_base_C = (consumer_warp_group_idx * rows_per_consumer_warp_group) + (m_iter * WGMMA_M);
                #pragma unroll
                for (int w = 0; w < WGMMA_N / 16; w++) {
                    int col = 16 * w + 2 * (tid % 4);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col)] = __float2bfloat16(alpha * d[m_iter][w][0]);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col + 1)] = __float2bfloat16(alpha * d[m_iter][w][1]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col)] = __float2bfloat16(alpha * d[m_iter][w][2]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col + 1)] = __float2bfloat16(alpha * d[m_iter][w][3]);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col + 8)] = __float2bfloat16(alpha * d[m_iter][w][4]);
                    s.C_epi[IDX_SMEM(row + row_tile_base_C, col + 9)] = __float2bfloat16(alpha * d[m_iter][w][5]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col + 8)] = __float2bfloat16(alpha * d[m_iter][w][6]);
                    s.C_epi[IDX_SMEM(row + 8 + row_tile_base_C, col + 9)] = __float2bfloat16(alpha * d[m_iter][w][7]);
                }
            }
            __syncthreads();

            // Phase 2: coalesced write to GMEM (alpha*D + beta*C)
            int row4_in_group = lane * 4;
            int group_base_row = consumer_warp_group_idx * rows_per_consumer_warp_group;
            if (row4_in_group < rows_per_consumer_warp_group) {
                int r0 = group_base_row + row4_in_group;
                #pragma unroll
                for (int c = warp; c < TILE_SIZE_N; c += 4) {
                    block_C[IDX_GMEM(r0 + 0, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 0, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 0, c)]));
                    block_C[IDX_GMEM(r0 + 1, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 1, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 1, c)]));
                    block_C[IDX_GMEM(r0 + 2, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 2, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 2, c)]));
                    block_C[IDX_GMEM(r0 + 3, c)] = __float2bfloat16(__bfloat162float(s.C_epi[IDX_SMEM(r0 + 3, c)]) + beta * __bfloat162float(block_C[IDX_GMEM(r0 + 3, c)]));
                }
            }
            #undef IDX_GMEM
            #undef IDX_SMEM
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

static const int TM_REF=128, TK_REF=64, TN_REF=128, NT_REF=256, NS_REF=3;

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
    constexpr int TM_PAD = TM_REF + 8;
    size_t smem = ((size_t)(TM_REF * TK_REF + TK_REF * TN_REF) * NS_REF
                   + (size_t)TM_PAD * TN_REF) * sizeof(bf16);
    auto kernel = gemm_bf16_pc_pipeline<TM_REF, TK_REF, TN_REF, 64, TN_REF, 16, NT_REF, NS_REF>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int nb = (M / TM_REF) * (N_out / TN_REF);
    kernel<<<nb, NT_REF, smem>>>(d_A, d_B, C, M, K_inner, N_out, 1.0f, 0.0f);
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
            name="bf16_gemm_mma_v6_a100_to_h100_ref",
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
        C_buf = torch.empty(M * N, dtype=A.dtype, device=A.device)
        _get_ext().run_gemm_h100(
            A.data_ptr(), B_T.data_ptr(), C_buf.data_ptr(), M, K_inner, N)
        # Reinterpret as [M, N] column-major, then make row-major contiguous
        C = C_buf.as_strided((M, N), (1, M)).contiguous()
        return {"C": C}
