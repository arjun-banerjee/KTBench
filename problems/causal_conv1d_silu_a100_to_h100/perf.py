"""Flop count for depthwise causal conv1d (width 4) + SiLU on [B, D, L].

Per output element: kW=4 multiplies (weight * tap) + 3 adds (kW-1
accumulations) + 4 SiLU ops (negate, exp, add 1, div, mul) = 11 flops.
"""

_K_WIDTH = 4


def flops(shapes, dtype) -> float:
    B = shapes["B"]
    D = shapes["D"]
    L = shapes["L"]
    per_elt = 2 * _K_WIDTH + 4  # mul+add per tap, then 4 for silu
    return float(B) * D * L * per_elt
