"""Input generator for gemm_pc_pipeline_a100_to_h100."""
import numpy as np
import torch


def _round_up(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


def make_inputs(shapes, dtype, rng, device):
    M = _round_up(shapes["M"], 128)
    K = _round_up(shapes["K"], 128)
    N = _round_up(shapes["N"], 128)
    dt = torch.bfloat16 if dtype == "fp16" else torch.float16
    scale = float(rng.uniform(0.5, 1.0))
    A = torch.from_numpy(rng.standard_normal((M, K)).astype("float32") * scale).to(dtype=dt, device=device)
    B = torch.from_numpy(rng.standard_normal((K, N)).astype("float32") * scale).to(dtype=dt, device=device)
    return [A, B]
