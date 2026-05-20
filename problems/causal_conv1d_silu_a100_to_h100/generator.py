"""
Input generator for causal_conv1d_silu_a100_to_h100.

Inputs:
  x      [B, D, L]  — float16/bfloat16 activations
  weight [D, 4]     — float16/bfloat16 per-channel filter (width=4)

Values are freshly sampled every call so hardcoded outputs cannot pass.
"""

import numpy as np
import torch

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}

kWidth = 4


def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device: torch.device) -> list:
    B, D, L = shapes["B"], shapes["D"], shapes["L"]
    dt = DTYPE_MAP[dtype]

    scale = float(rng.uniform(0.5, 2.0))
    x = torch.from_numpy(rng.standard_normal((B, D, L)).astype("float32") * scale).to(dtype=dt, device=device)
    # Filter weights: small range to keep activations reasonable through SiLU
    w = torch.from_numpy(rng.standard_normal((D, kWidth)).astype("float32") * 0.5).to(dtype=dt, device=device)

    return [x, w]
