"""
Input generator for wkv_recurrence_a100_to_h100.

Inputs:
  r, k, v  [B, H, T, N]  fp16/bf16  — receptance, key, value
  w        [H, N]        fp32       — per-head log-decay (always ≤ 0)
  u        [H, N]        fp16/bf16  — bonus (first-token gate)
"""

import numpy as np
import torch

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device: torch.device) -> list:
    B, H, T, N = shapes["B"], shapes["H"], shapes["T"], shapes["N"]
    dt = DTYPE_MAP[dtype]

    scale = float(rng.uniform(0.3, 1.0))
    r = torch.from_numpy(rng.standard_normal((B, H, T, N)).astype("float32") * scale).to(dtype=dt, device=device)
    k = torch.from_numpy(rng.standard_normal((B, H, T, N)).astype("float32") * scale).to(dtype=dt, device=device)
    v = torch.from_numpy(rng.standard_normal((B, H, T, N)).astype("float32") * scale).to(dtype=dt, device=device)

    # w: log-decay, always negative → decay = exp(w) ∈ (0,1)
    w = torch.from_numpy(-rng.uniform(0.01, 2.0, (H, N)).astype("float32")).to(dtype=torch.float32, device=device)
    u = torch.from_numpy(rng.standard_normal((H, N)).astype("float32") * 0.5).to(dtype=dt, device=device)

    return [r, k, v, w, u]
