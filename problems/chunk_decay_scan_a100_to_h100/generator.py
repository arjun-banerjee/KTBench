"""
Input generator for chunk_decay_scan_a100_to_h100.

Inputs:
  x         [B, H, L, D]  fp16/bf16  — token features
  log_decay  [H, D]       fp32       — per-head, per-dim log decay rate
                                       (kept in fp32 for accuracy; always ≤ 0)
"""

import numpy as np
import torch

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device: torch.device) -> list:
    B, H, L, D = shapes["B"], shapes["H"], shapes["L"], shapes["D"]
    dt = DTYPE_MAP[dtype]

    scale = float(rng.uniform(0.5, 1.5))
    x = torch.from_numpy(rng.standard_normal((B, H, L, D)).astype("float32") * scale).to(dtype=dt, device=device)

    # log_decay: always negative so that decay = exp(log_decay) ∈ (0, 1).
    # Range [-2, -0.01] gives decay from ~0.13 to ~0.99 — covers both fast and slow fading.
    log_decay = torch.from_numpy(
        -rng.uniform(0.01, 2.0, (H, D)).astype("float32")
    ).to(dtype=torch.float32, device=device)

    return [x, log_decay]
