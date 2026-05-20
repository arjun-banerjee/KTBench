# Adding Problems to KTBench

Each problem teaches the model to translate a kernel from one DSL/hardware pair to another. This guide walks through every file you need to write, the invariants that tie them together, and how to verify a new problem end-to-end.

---

## Directory layout

Every problem lives in `problems/{problem_id}/` and contains six files plus one optional:

```
problems/flash_attn_a100_to_h100/
├── meta.toml          # identity: translation axis, tags, difficulty, tolerances
├── test_suite.toml    # shapes to test + stress ranges
├── generator.py       # makes fresh random inputs for each test call
├── source.py          # the input kernel (what the model sees)
├── oracle.py          # ground-truth reference in PyTorch eager
├── reference_tgt.py   # handwritten target-side reference (performance baseline)
├── perf.py            # optional: flops(shapes, dtype) for compute-axis SOL
└── notes.md           # human description and translation gotchas (optional)
```

---

## The central invariant

All three `ModelNew` classes — `source.py`, `oracle.py`, and `reference_tgt.py` — must accept the **same input tensors** that `generator.py` produces:

```
generator.py::make_inputs(shapes, dtype, rng, device) → list[Tensor]
                                    ↓  same tensors  ↓
source.py::ModelNew.forward(*inputs)          # shown to the model
oracle.py::ModelNew.forward(*inputs)          # correctness reference
reference_tgt.py::ModelNew.forward(*inputs)   # performance baseline
```

If your kernel takes `(q, k, v)`, every file must accept `(q, k, v)` in that order.

---

## File reference

### `meta.toml`

Declares the translation axis and scoring parameters.

```toml
problem_id   = "flash_attn_a100_to_h100"
name         = "Flash Attention Forward (A100 CUDA → H100 CUDA)"
src_dsl      = "cuda"                 # DSL of source.py
src_hw       = "nvidia_a100_sxm"      # hardware source.py targets
tgt_dsl      = "cuda"                 # DSL the model must produce
tgt_hw       = "nvidia_h100_sxm"      # hardware the output runs on
tags         = ["attention", "fp16", "tiling"]
difficulty   = 3                      # 1 (trivial) – 5 (research-level)
provenance   = "hand-written"         # or arxiv:XXXX.XXXXX, etc.

# Override default tolerances when accumulation differs across DSLs.
[tolerances.fp16]
atol = 1e-2
rtol = 1e-2
```

**Hardware keys** (from the registry):

| Key | Hardware | FP16 TFLOP/s | DRAM BW | In eval fleet |
|---|---|---|---|---|
| `nvidia_h200_sxm` | H200 SXM | 989 | 4800 GB/s | yes (primary) |
| `nvidia_a100_sxm` | A100 SXM | 312 | 2000 GB/s | yes (secondary) |
| `nvidia_h100_sxm` | H100 SXM | 989 | 3350 GB/s | no |
| `amd_mi300x`      | MI300X   | 1307 | 5300 GB/s | no |
| `aws_trainium2`   | Trainium2 | 832 | 820 GB/s  | TBD |
| `google_tpuv4`    | TPUv4    | 275 | 300 GB/s  | no |

**DSL keys**: `cuda`, `cute`, `triton`, `tilelang`, `helion`, `hip`, `nki`, `pallas`, `numba`, `mojo`, `pytorch`

---

### `generator.py`

Called for every test case and stress trial. Must return a `list` of tensors matching the `forward(*inputs)` signature. All randomness must go through `rng` so that values differ between calls — hardcoded outputs cannot pass.

```python
import numpy as np
import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

def make_inputs(
    shapes: dict,           # keys defined in test_suite.toml
    dtype: str,             # "fp16", "bf16", or "fp32"
    rng: np.random.Generator,
    device: torch.device,
) -> list:
    B, H, N, D = shapes["B"], shapes["H"], shapes["N"], shapes["D"]
    dt = DTYPE_MAP[dtype]
    scale = float(rng.uniform(0.5, 2.0))   # vary scale across trials

    def rand(*shape):
        return torch.from_numpy(
            rng.standard_normal(shape).astype("float32") * scale
        ).to(dtype=dt, device=device)

    return [rand(B, H, N, D), rand(B, H, N, D), rand(B, H, N, D)]  # q, k, v
```

Rules:
- Use `rng` (a `numpy.random.Generator`) for **all** randomness — not `torch.randn`, not `random`.
- Vary scale or distribution parameters with `rng` so that each trial has different statistics.
- Return a plain Python `list` of `torch.Tensor`, not a dict or tuple.

---

### `source.py` — the input kernel

This is what the model reads. Write it as a working `ModelNew` class that runs correctly on the source hardware. For CUDA kernels, use `torch.utils.cpp_extension.load_inline`:

```python
"""
Source kernel: Flash Attention forward in CUDA (A100).
Candidate must translate this to CUDA optimized for the H100.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_fp16.h>
// ... your A100-tuned kernel ...
__global__ void flash_attn_kernel(...) { ... }

void flash_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                        torch::Tensor out) { ... }
"""

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="flash_attn_a100_src",
            cpp_sources="void flash_attn_forward("
                        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA_SRC,
            functions=["flash_attn_forward"],
            with_cuda=True,
            verbose=False,
        )
    return _ext


class ModelNew(nn.Module):
    def forward(self, q, k, v):
        out = torch.empty_like(q)
        out.zero_()                    # always zero the output buffer
        _get_ext().flash_attn_forward(
            q.contiguous(), k.contiguous(), v.contiguous(), out
        )
        return out
```

Important: always call `out.zero_()` before writing to the output buffer. This prevents stale GPU memory from making incorrect kernels appear to pass.

---

### `oracle.py` — ground truth

**Always PyTorch eager.** Numerically correct by definition. Cast to fp32 for accumulation, then cast back to the input dtype:

```python
"""
Ground-truth oracle: Flash Attention in PyTorch SDPA (fp32 accumulation).
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, q, k, v):
        # Use fp32 accumulation for numerical stability.
        out = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        )
        return out.to(q.dtype)
```

The oracle defines what "correct" means. If your oracle has a bug, every candidate will fail. Test it manually before building oracle tensors.

---

### `reference_tgt.py` — handwritten target reference

Your best hand-tuned implementation in the **target** DSL running on the **target** hardware. This is the performance baseline — the leaderboard shows `speedup_vs_ref` as context, though the scored metric is SOL (hardware utilization fraction). Write the strongest version you can.

```python
"""
Handwritten reference: Flash Attention forward in CUDA (H100).
Performance baseline for scoring — candidates are compared against this.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
// H100-specific optimisations: wgmma, TMA, persistent kernels, etc.
...
"""

# ... (same load_inline pattern as source.py) ...

class ModelNew(nn.Module):
    def forward(self, q, k, v):
        out = torch.empty_like(q)
        out.zero_()
        _get_ext().flash_attn_forward_h100(
            q.contiguous(), k.contiguous(), v.contiguous(), out
        )
        return out
```

If you don't have an optimised reference yet, you can temporarily use a PyTorch fallback to unblock development — just replace it before publishing the problem.

---

### `test_suite.toml`

Defines which shapes to test. Values are always freshly randomized — only shapes are fixed:

```toml
# Structured cases: fixed shapes, fresh random values every run.
[[cases]]
id    = "small"
desc  = "small batch, short sequence"
dtype = "fp16"
[cases.shapes]
B = 2
H = 8
N = 128
D = 64

[[cases]]
id    = "medium"
desc  = "typical inference shape"
dtype = "fp16"
[cases.shapes]
B = 4
H = 16
N = 512
D = 128

[[cases]]
id    = "long_seq"
desc  = "long sequence — tests memory footprint"
dtype = "fp16"
[cases.shapes]
B = 1
H = 32
N = 4096
D = 128

[[cases]]
id    = "bf16"
desc  = "bfloat16 dtype"
dtype = "bf16"
[cases.shapes]
B = 2
H = 16
N = 256
D = 128

# Stress: both shapes AND values are random. Shape-specific hacks fail.
[stress]
num_trials     = 30
pass_threshold = 0.90

[stress.shape_ranges]
B = [1, 8]
H = [1, 32]
N = [16, 2048]
D = [16, 128]
```

Guidelines:
- Cover at least 4–6 structured cases spanning a range of sizes and at least one non-power-of-2 shape.
- Include a `bf16` or `fp32` case if the kernel supports multiple dtypes.
- Set `stress.shape_ranges` tightly enough that random shapes stay within what the kernel can handle.

---

### `perf.py` (optional, recommended for compute-axis SOL)

`perf.py` is the one knob that lets KTBench report the compute axis of
SOL accurately. Without it, the eval still reports `memory_util`
(auto-computed from input + output byte totals) but leaves
`compute_util` at -1, so `sol = max(compute_util, memory_util) =
memory_util`. Skip it when your op is bandwidth-bound and you do not
care about a compute-side SOL number; ship one when the op is
arithmetically dense (gemm, attention without flash-style memory
tricks) and the compute axis should drive the final score.

```python
# problems/flash_attn_a100_to_h100/perf.py
def flops(shapes, dtype) -> float:
    B = shapes["B"]
    H = shapes["H"]
    S = shapes["S"]
    D = shapes["D"]
    # Q @ K^T = 2 * B * H * S * S * D, then * V = 2 * B * H * S * S * D,
    # plus softmax (4 ops per element of the S * S score matrix).
    qk  = 2.0 * B * H * S * S * D
    sv  = 2.0 * B * H * S * S * D
    sm  = 4.0 * B * H * S * S
    return qk + sv + sm
```

**Counting convention.** Standard SOL accounting; the same numbers
Nsight / NCU would report for "executed FLOPs" if you ran the kernel
under them.

- `add`, `sub`, `mul`, `div`, `min`, `max`, compare: **1 flop** each.
- Fused multiply-add: **2 flops** (counted as separate mul + add).
- Transcendentals (`exp`, `log`, `rsqrt`, `sqrt`, `sigmoid`, `tanh`):
  **1 flop** each. This is the SOL ceiling convention; some pipelines
  count them as 4-8 because they decompose into a polynomial in
  hardware, but for the SOL denominator you want the *minimum* work
  the kernel had to do.
- Reduction over N elements: **N flops** (one accumulating add per
  element; the last add is the result).
- Element-wise op over a tensor of shape S: **prod(S) flops**.
- Matmul `[M, K] @ [K, N] = [M, N]`: **2 * M * N * K flops** (the
  classic GEMM count: N output elements, each is a K-long dot product
  = K muls + K-1 adds ≈ 2K flops per output).

The signature is `flops(shapes, dtype) -> float`. `shapes` is the
first structured case's shape dict (the same one timing runs against);
`dtype` is its dtype string. Return a float (the count, not a TFLOP
rate). The harness divides by measured runtime and the target HW's
peak TFLOP/s to land on `compute_util` in [0, 1].

Worked examples for the shipped problems:

| Problem | `flops(shapes, dtype)` |
|---|---|
| `softmax_a100_to_h100` | `5 * N * D` (sub + exp + add + div per element + max reduction) |
| `swiglu_activation_a100_to_h100` | `5 * N * D` (4 ops for SiLU + 1 mul by up) |
| `fused_rms_norm_residual_a100_to_h100` | `B * L * (5 * D + 3)` (D squares + D-sum + rsqrt + 2 * D muls + residual) |
| `causal_conv1d_silu_a100_to_h100` | `B * D * L * 11` (4 muls + 3 adds per tap + 4 for SiLU) |
| `chunk_decay_scan_a100_to_h100` | `2 * B * H * L * D + H * D` (mul-add per element + one exp per decay entry) |
| `hadamard_transform_a100_to_h100` | `B * N * 2 * log2(N)` |

**When to skip perf.py.** Memory-bound ops (softmax, swiglu, rmsnorm,
hadamard, most activations) reach SOL on the memory axis well before
the compute axis even at saturation, so `memory_util` alone is the
right ceiling and adding flops just confirms it. Ship `perf.py`
anyway if you want both axes on the leaderboard; the cost is a few
lines per problem and the gain is a `compute_util` column that's
nonzero, which helps spot kernels that are spending real arithmetic
work on the right path.

---

### `notes.md` (optional but recommended)

Free-form documentation for problem authors and evaluators:

```markdown
# Flash Attention Forward: A100 CUDA → H100 CUDA

## Description
...

## Translation Challenges
- **wgmma vs tensor cores**: H100 introduces wgmma instructions...
- **TMA**: H100's Tensor Memory Accelerator allows async copies...

## Known Gotchas
- Head dimension D must be a power of 2 for this implementation.

## Provenance
Adapted from the FlashAttention-2 paper (Dao 2023).
```

---

## Scaffolding a new problem

The `add_problem.py` script creates the directory and empty file stubs:

```bash
python scripts/add_problem.py \
    --id flash_attn_a100_to_h100 \
    --src-dsl cuda --src-hw nvidia_a100_sxm \
    --tgt-dsl cuda --tgt-hw nvidia_h100_sxm \
    --name "Flash Attention Forward (A100 CUDA → H100 CUDA)" \
    --tags attention,fp16,tiling \
    --difficulty 3
```

Then fill in `source.py`, `oracle.py`, `reference_tgt.py`, `generator.py`, and `test_suite.toml`.

---

## Verifying a new problem

### 1. Build oracle tensors (requires a GPU)

```bash
python scripts/build_oracle_tensors.py \
    --problem problems/flash_attn_a100_to_h100 \
    --device 0
```

This runs `oracle.py` over all structured cases and saves the outputs to `oracle_tensors/`. If this step fails, your oracle has a bug.

### 2. Smoke-test with the reference

Use `reference_tgt.py` as the candidate — it should score near 1.0 on correctness and a strong SOL:

```bash
python scripts/run_eval.py \
    --problem problems/flash_attn_a100_to_h100 \
    --candidate problems/flash_attn_a100_to_h100/reference_tgt.py \
    --device 0 --verbose
```

A passing result looks like:
```
correctness_rate : 1.0
stress_pass_rate : 1.0
sol_score        : 0.82
final_score      : 0.82
```

### 3. Smoke-test with the source

Run `source.py` through the evaluator as well. It runs on the source hardware, so its SOL score is against the source HW ceiling — but it should still be numerically correct:

```bash
python scripts/run_eval.py \
    --problem problems/flash_attn_a100_to_h100 \
    --candidate problems/flash_attn_a100_to_h100/source.py \
    --device 0
```

### 4. Test with an LLM agent

```bash
OPENAI_API_KEY=sk-... python scripts/run_agent.py \
    --problem problems/flash_attn_a100_to_h100 \
    --model o3 --reasoning-effort medium \
    --multi-turn --max-turns 10 --device 0 --verbose
```

---

## Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| `generator.py` uses `torch.randn` instead of `rng` | Stress trials all see the same distribution | Replace with `rng.standard_normal(...)` |
| Output buffer not zeroed in `source.py` / `reference_tgt.py` | Candidates pass by reading stale GPU memory | Add `out.zero_()` before writing |
| `oracle.py` accumulates in fp16 | Correctness failures on wide inputs | Cast to fp32 inside the oracle, cast back at return |
| `test_suite.toml` shape keys don't match `generator.py` | `KeyError` during eval | Ensure `shapes["B"]` etc. match `[cases.shapes]` keys |
| Stress ranges include shapes the kernel can't handle | Random stress failures in the reference | Narrow `[stress.shape_ranges]` to safe bounds |
| `reference_tgt.py` is a PyTorch fallback | SOL score near 0 — baseline is too slow | Write an optimised target-side implementation |
