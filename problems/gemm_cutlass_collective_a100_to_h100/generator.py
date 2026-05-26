"""Input generator for gemm_cutlass_collective_a100_to_h100."""
import numpy as np
import torch


def make_inputs(shapes, dtype, rng, device):
    M, N, K = shapes["M"], shapes["N"], shapes["K"]
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    scale = float(rng.uniform(0.5, 1.0))

    def rand(*shape):
        return torch.from_numpy(
            rng.standard_normal(shape).astype("float32") * scale
        ).to(dtype=dt, device=device)

    return [rand(M, K), rand(K, N)]
