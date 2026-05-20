"""Flop count for the Fast Walsh-Hadamard transform on [B, N], N=2^k.

log2(N) butterfly passes. Each pass touches every element with one add
or one subtract, so each pass is N flops per row (one op per element).
Total per row: 2 * N * log2(N) flops (counting each butterfly's add and
subtract as separate ops).
"""

import math


def flops(shapes, dtype) -> float:
    B = shapes["B"]
    N = shapes["N"]
    # snap N to power of two to match the generator
    n = 1 << max(int(math.log2(N)), 0)
    if n < 2:
        return 0.0
    return float(B) * n * 2.0 * math.log2(n)
