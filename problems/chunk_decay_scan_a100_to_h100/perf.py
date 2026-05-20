"""Flop count for chunked exponential-decay scan on [B, H, L, D].

Recurrence: state = state * decay + x[t]; one output per timestep.
Per (b, h, l, d) element: 1 mul + 1 add = 2 flops.
Plus one exp per (h, d) entry to materialise the decay tile.
"""


def flops(shapes, dtype) -> float:
    B = shapes["B"]
    H = shapes["H"]
    L = shapes["L"]
    D = shapes["D"]
    scan = 2.0 * B * H * L * D
    decay = float(H) * D
    return scan + decay
