"""
Input generator for fused_rms_norm_residual_a100_to_h100.

Inputs:
  x [B, L, D]   — fp16/bf16 activations (main stream)
  r [B, L, D]   — fp16/bf16 residual
  w [D]         — fp16/bf16 affine scale (RMSNorm weight)
"""

import numpy as np
import torch

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device: torch.device) -> list:
    B, L, D = shapes["B"], shapes["L"], shapes["D"]
    dt = DTYPE_MAP[dtype]

    scale = float(rng.uniform(0.5, 2.0))
    x = torch.from_numpy(rng.standard_normal((B, L, D)).astype("float32") * scale).to(dtype=dt, device=device)
    r = torch.from_numpy(rng.standard_normal((B, L, D)).astype("float32") * scale).to(dtype=dt, device=device)
    # Weight near 1 so output magnitude stays sensible
    w = torch.from_numpy((rng.standard_normal((D,)).astype("float32") * 0.1 + 1.0)).to(dtype=dt, device=device)

    return [x, r, w]
