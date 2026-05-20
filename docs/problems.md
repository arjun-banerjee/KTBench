# Writing a KTBench Problem

Each problem lives in `problems/<problem_id>/` and consists of six files.

---

## Directory layout

```
problems/my_problem/
├── meta.toml        # identity, tags, tolerances
├── test_suite.toml  # structured cases + stress config
├── generator.py     # input tensor factory
├── oracle.py        # ground-truth reference (PyTorch eager)
├── source.py        # source kernel the model will see
├── reference_tgt.py # hand-written target kernel (performance baseline)
└── notes.md         # human-readable description and gotchas
```

---

## `meta.toml`

```toml
problem_id = "my_problem"
name       = "Human-readable title"
src_dsl    = "cuda"          # source language: "cuda", "triton", etc.
src_hw     = "nvidia_h200_sxm"
tgt_dsl    = "triton"        # target language the model must produce
tgt_hw     = "nvidia_h200_sxm"
tags       = ["reduction", "fp16"]
difficulty = 2               # 1 (easy) – 5 (very hard)
provenance = "hand-written reference"

# Optional: per-dtype tolerances (defaults are tight)
[tolerances.fp16]
atol = 1e-2
rtol = 1e-2
```

---

## `test_suite.toml`

Define **structured cases** (fixed shapes, random values each run) and an optional **stress block** (random shapes *and* values every trial).

```toml
[[cases]]
id    = "small"
desc  = "basic correctness"
dtype = "fp16"
[cases.shapes]
N = 128
D = 256

[[cases]]
id    = "nonpow2"
desc  = "non-power-of-2 — breaks naive tiling"
dtype = "fp16"
[cases.shapes]
N = 1000
D = 300

# Stress: random shapes AND values — shape-specific hacks fail here
[stress]
num_trials     = 30
pass_threshold = 0.90   # fraction of trials that must pass

[stress.shape_ranges]
N = [1, 4096]
D = [1, 8192]
```

Pick cases that cover: small (sanity), large (performance), edge cases (single row, D=1, non-power-of-2), and any dtype variants the kernel must support.

---

## `generator.py`

Returns a list of input tensors for one test case. Values must be freshly sampled from `rng` — hardcoded tensors defeat the benchmark.

```python
import numpy as np
import torch

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device: torch.device) -> list:
    N, D = shapes["N"], shapes["D"]
    dt = DTYPE_MAP[dtype]
    x = torch.from_numpy(rng.standard_normal((N, D)).astype("float32")).to(dtype=dt, device=device)
    return [x]
```

**Rules:**
- Accept exactly `(shapes, dtype, rng, device)`.
- Use `rng` (NumPy Generator) for all randomness — never `random` or `torch.rand`.
- Return a plain `list`; the harness unpacks it as positional args to `forward`.

---

## `oracle.py`

Ground-truth implementation. Must be numerically correct by definition — this is what candidate outputs are compared against.

```python
import torch, torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
```

Use PyTorch eager (no custom kernels). Accumulate in fp32 when the operation is reduction-heavy.

---

## `source.py`

The kernel the model is given as input — the thing it must translate. Wrap it in the same `ModelNew(nn.Module)` interface.

```python
import torch, torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...  # CUDA / source-DSL implementation
```

---

## `reference_tgt.py`

A correct, performant hand-written implementation in the *target* DSL. The harness measures the candidate's throughput as a fraction of this baseline (Speed-of-Light). It is not shown to the model.

---

## `notes.md`

Describe the algorithm, list translation challenges, and call out any known gotchas. This file is shown to human reviewers but not to the model during eval.

Suggested sections:
- **Description** — what the kernel computes
- **Translation Challenges** — what makes this non-trivial to translate
- **Known Gotchas** — sharp edges a candidate is likely to hit
- **Provenance** — how the problem was created

---

## Checklist before submitting a problem

- [ ] `problem_id` in `meta.toml` matches the directory name
- [ ] At least one structured case per dtype the kernel supports
- [ ] Stress block included with `num_trials >= 20`
- [ ] `generator.py` uses only `rng` for randomness
- [ ] `oracle.py` passes its own cases when run through the harness
- [ ] `reference_tgt.py` compiles and passes all cases on the target hardware
- [ ] `notes.md` written (especially the gotchas section)
