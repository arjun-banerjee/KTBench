"""Prompt construction for CUDA A100 → CUDA H100 translation tasks."""

from __future__ import annotations

from ktbench.problem import Problem
from ktbench.source_bundle import collect_prompt_source_files, fence_language

_H100_SUMMARY = (
    "NVIDIA H100 SXM — GH100 Hopper die, 132 SMs, 3350 GB/s HBM3, "
    "989 TFLOP/s BF16 Tensor Core, 50 MB L2, 228 KB shared memory / SM, "
    "warp size 32.\n"
    "KEY ARCHITECTURAL DIFFERENCES FROM A100 (you MUST use these for peak performance):\n"
    "- WGMMA (Warpgroup MMA): replaces mma.sync. Operates on 4-warp warpgroups (128 threads). "
    "Use wgmma.mma_async.sync.aligned PTX intrinsics (e.g. "
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16). "
    "Requires warpgroup-level sync: __syncwarp() is insufficient; use asm volatile('wgmma.fence.sync.aligned;') "
    "and asm volatile('wgmma.commit_group.sync.aligned;') / wgmma.wait_group.\n"
    "- TMA (Tensor Memory Accelerator): replaces cp.async for bulk shared-memory loads. "
    "Create a CUtensorMap descriptor (cuTensorMapEncodeTiled) and use "
    "cp.async.bulk.tensor.2d.shared::cluster.global / barrier to move tiles asynchronously. "
    "Requires cuda.h / cuda_runtime.h and linking against libcuda (-lcuda).\n"
    "- Compile target: must be sm_90a (not sm_90) to enable WGMMA and TMA PTX. "
    "In load_inline use extra_cuda_cflags=[] (leave -arch out; set "
    "TORCH_CUDA_ARCH_LIST=9.0a in the environment instead).\n"
    "- Persistent kernels with Producer/Consumer warpgroups are the recommended pattern: "
    "one warpgroup issues TMA loads; another issues WGMMA; coordinate via mbarrier."
)

_CUDA_LIBRARIES = "torch.utils.cpp_extension.load_inline, custom __global__ kernels, cuBLAS, cuDNN."

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
    """Build the model prompt for a CUDA A100 → CUDA H100 translation problem."""
    meta = problem.meta

    lines = [
        "# Kernel Translation Task",
        "",
        f"**Problem:** {meta.name}",
        f"**Source:** CUDA on NVIDIA A100 SXM",
        f"**Target:** CUDA on NVIDIA H100 SXM",
        f"**Difficulty:** {meta.difficulty}/5",
        f"**Tags:** {', '.join(meta.tags)}",
        "",
        "## Target Hardware",
        "",
        _H100_SUMMARY,
        "",
        "## Allowed Libraries",
        "",
        _CUDA_LIBRARIES,
        "",
        _INTERFACE_NOTE,
        "## Source Kernel (CUDA / A100)",
        "",
        "Translate the following kernel to CUDA for the H100:",
        "",
    ]

    bundle = collect_prompt_source_files(problem.problem_dir)
    if (problem.problem_dir / "source.py").exists():
        lines.extend([
            "### Harness interface (`source.py`)",
            "",
            "The grader calls `ModelNew().forward(*inputs)` as defined here. "
            "Your optimized kernel must preserve this calling convention.",
            "",
            "```python",
            problem.source_src.strip(),
            "```",
            "",
        ])
    if bundle:
        lines.append("### Kernel source files")
        lines.append("")
        for fname, text in bundle:
            lang = fence_language(fname)
            lines.extend([f"#### `{fname}`", "", f"```{lang}", text.strip(), "```", ""])
    elif not (problem.problem_dir / "source.py").exists():
        lines.extend([
            "```python",
            problem.source_src.strip(),
            "```",
            "",
        ])

    lines.extend([
        "## Your Translation (CUDA / H100)",
        "",
        "Implement `ModelNew` below.",
        "```python",
        "# Your implementation here",
        "```",
    ])

    return "\n".join(lines)
