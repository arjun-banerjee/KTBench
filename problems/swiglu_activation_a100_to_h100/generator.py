"""Input generator for swiglu_activation_a100_to_h100.

Input x: [N, 2*D] (gate concatenated with up along last dim).
Output (constructed by ModelNew): [N, D].

Values are sampled fresh every trial; a random per-call scale exercises
numerical stability across magnitudes.
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
    # KTBench's loader wraps structured-case scalars in single-element
    # lists for forward compatibility with per-case shape sweeps; stress
    # passes raw ints. Accept both.
    def _scalar(v):
        return v[0] if isinstance(v, list) else v

    N = _scalar(shapes["N"])
    D = _scalar(shapes["D"])
    dt = DTYPE_MAP[dtype]

    scale = float(rng.uniform(0.5, 3.0))
    x_np = rng.standard_normal((N, 2 * D)).astype("float32") * scale
    x = torch.from_numpy(x_np).to(dtype=dt, device=device)
    return [x]
