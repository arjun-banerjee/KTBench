"""Generator for bf16_gemv_a100_to_h100.

y = W @ x  (BF16) — x[K], W[N,K] → y[N]
"""
import numpy as np
import torch


def make_inputs(shapes, dtype, rng, device):
    # Snap to alignment: N multiple of 4, K multiple of 256
    N = max(4, ((shapes["N"] + 3) // 4) * 4)
    K = max(256, ((shapes["K"] + 255) // 256) * 256)

    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    scale = float(rng.uniform(0.5, 2.0))
    x = torch.from_numpy(
        rng.standard_normal(K).astype("float32") * scale
    ).to(dtype=dt, device=device)
    W = torch.from_numpy(
        rng.standard_normal((N, K)).astype("float32") * scale
    ).to(dtype=dt, device=device)
    return [x, W]
