"""
Input generator for hadamard_transform_a100_to_h100.

Input: x [B, N]  fp16/bf16   (N must be a power of 2; enforced here)
Output: y [B, N] fp16/bf16   = (1/sqrt(N)) * H_N @ x

For the stress case, N is sampled from the allowed range and rounded to the
nearest power of two so the kernel can always be launched.
"""

import numpy as np
import torch

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _nearest_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device: torch.device) -> list:
    B = shapes["B"]
    N = _nearest_pow2(int(shapes["N"]))  # snap to power of 2

    dt = DTYPE_MAP[dtype]
    scale = float(rng.uniform(0.5, 2.0))
    x = torch.from_numpy(rng.standard_normal((B, N)).astype("float32") * scale).to(dtype=dt, device=device)

    return [x]
