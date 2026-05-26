"""Input generator for flash_attn_bwd_a100_to_h100."""
import numpy as np
import torch


def make_inputs(shapes, dtype, rng, device):
    B, H, N, D = shapes["B"], shapes["H"], shapes["N"], shapes["D"]
    if D not in (32, 64, 96, 128, 256):
        D = 128 if D > 96 else ((D + 31) // 32) * 32
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    scale = float(rng.uniform(0.5, 1.0))

    def rand(*shape):
        return torch.from_numpy(
            rng.standard_normal(shape).astype("float32") * scale
        ).to(dtype=dt, device=device)

    return [rand(B, H, N, D), rand(B, H, N, D), rand(B, H, N, D), rand(B, H, N, D)]
