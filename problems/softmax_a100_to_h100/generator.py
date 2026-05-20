"""
Input generator for softmax_h200_to_triton.

Softmax forward: input [N, D] float → output [N, D] float.

Values are always freshly sampled from rng — hardcoded outputs cannot pass.
"""

import numpy as np
import torch

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def make_inputs(
    shapes: dict,
    dtype: str,
    rng: np.random.Generator,
    device: torch.device,
) -> list:
    N = shapes["N"]
    D = shapes["D"]
    dt = DTYPE_MAP[dtype]

    # Random values in a range that exercises numerical stability.
    # Using standard_normal and scaling avoids degenerate all-zero / all-inf cases.
    # Scale varies per call via rng so no two trials have the same distribution.
    scale = float(rng.uniform(0.5, 3.0))
    x_np = rng.standard_normal((N, D)).astype("float32") * scale
    x = torch.from_numpy(x_np).to(dtype=dt, device=device)

    return [x]
