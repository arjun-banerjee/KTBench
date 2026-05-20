"""
Prompt construction.

What IS in the prompt:
  - Full source kernel (source.py text)
  - Target DSL name + short hardware summary
  - ModelNew interface signature (forward() argument names and dtypes)
  - Semantic description from meta.toml
  - Allowed libraries for the target DSL

What is NOT in the prompt:
  - Exact input shapes (these are in test_suite.toml, never passed to the model)
  - Oracle outputs
  - Stress case details
  - Scoring internals
"""

from __future__ import annotations

from ktbench.problem import Problem
from ktbench.registry.hardware import HardwareSpec, get_hw_spec

_HW_SUMMARIES: dict[str, str] = {
    "nvidia_h200_sxm": (
        "NVIDIA H200 SXM — GH100 Hopper die, 132 SMs, 4800 GB/s HBM3e, "
        "989 TFLOP/s FP16 Tensor Core, 90 MB L2, 128 KB shared memory / SM, "
        "warp size 32."
    ),
    "nvidia_a100_sxm": (
        "NVIDIA A100 SXM — GA100 Ampere die, 108 SMs, 2000 GB/s HBM2e, "
        "312 TFLOP/s FP16 Tensor Core, 40 MB L2, 164 KB shared memory / SM, "
        "warp size 32. Supports async copy (cp.async) and TF32."
    ),
    "aws_trainium2": (
        "AWS Trainium2 — 128 NeuronCores-v3, 832 TFLOP/s FP16, 820 GB/s HBM. "
        "Use NKI (Neuron Kernel Interface) with @nki.jit for device kernels. "
        "Scratchpad-based memory model; no global L2 cache."
    ),
    "amd_mi300x": (
        "AMD Instinct MI300X — CDNA3, 304 CUs, 5300 GB/s HBM3, "
        "1307 TFLOP/s FP16, 32 MB L2. Use HIP C++ or Triton. "
        "Wavefront size 64."
    ),
}

_DSL_LIBRARIES: dict[str, str] = {
    "cuda":    "torch.utils.cpp_extension.load_inline, custom __global__ kernels, cuBLAS, cuDNN.",
    "cute":    "CuTe (CUTLASS 3.x): cute::Layout, cute::Tensor, cute::copy, cute::gemm.",
    "triton":  "triton: @triton.jit, tl.load, tl.store, tl.dot, tl.math.*",
    "tilelang":"tilelang: @T.prim_func, T.Kernel, T.alloc_shared, T.gemm.",
    "helion":  "helion: @helion.kernel, hl.load, hl.store.",
    "hip":     "HIP C++: __global__ kernels via torch.utils.cpp_extension (hipcc). HIP runtime API.",
    "nki":     "NKI: @nki.jit, nl.load, nl.store, nl.matmul, nisa.*",
    "pallas":  "JAX Pallas: pl.pallas_call, pl.load, pl.store.",
    "numba":   "Numba CUDA: @numba.cuda.jit, cuda.shared.array, cuda.syncthreads.",
    "mojo":    "Mojo: fn kernels with GPU intrinsics via max.gpu.*",
    "pytorch": "PyTorch eager / torch.compile (any ops).",
}

_INTERFACE_NOTE = """\
## Interface Contract

Your submission must define a class named `ModelNew` with a `forward` method.
The grader will call `ModelNew().forward(*inputs)` where `inputs` is a list of
`torch.Tensor` arguments matching the source kernel's input structure.
Your `forward` must return the same output tensor(s) as the source kernel.

You may define helper functions and compile multiple device kernels internally.
Only `forward` is called by the grader.

**Do not** hardcode shapes, input values, or expected outputs — inputs are
freshly randomized each evaluation run. Your kernel must implement the
correct algorithm for arbitrary valid inputs.
"""


def build_prompt(problem: Problem) -> str:
    """
    Build the model prompt for a translation problem.
    Does NOT include input shapes, test case details, or oracle outputs.
    """
    meta = problem.meta
    hw_key = meta.tgt_hw
    hw_summary = _HW_SUMMARIES.get(hw_key, f"Target hardware: {hw_key}")
    dsl_libs = _DSL_LIBRARIES.get(meta.tgt_dsl, f"Target DSL: {meta.tgt_dsl}")

    lines = [
        f"# Kernel Translation Task",
        f"",
        f"**Problem:** {meta.name}",
        f"**Source DSL:** `{meta.src_dsl}` on `{meta.src_hw}`",
        f"**Target DSL:** `{meta.tgt_dsl}` on `{meta.tgt_hw}`",
        f"**Difficulty:** {meta.difficulty}/5",
        f"**Tags:** {', '.join(meta.tags)}",
        f"",
        f"## Target Hardware",
        f"",
        hw_summary,
        f"",
        f"## Allowed Libraries",
        f"",
        dsl_libs,
        f"",
        _INTERFACE_NOTE,
        f"## Source Kernel (`{meta.src_dsl}`)",
        f"",
        f"Translate the following kernel to `{meta.tgt_dsl}` for `{meta.tgt_hw}`:",
        f"",
        "```python",
        problem.source_src.strip(),
        "```",
        f"",
        f"## Your Translation (`{meta.tgt_dsl}`)",
        f"",
        f"Implement `ModelNew` below.",
        "```python",
        "# Your implementation here",
        "```",
    ]

    return "\n".join(lines)
