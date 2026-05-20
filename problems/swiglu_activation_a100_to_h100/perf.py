"""Flop count for SwiGLU: silu(gate) * up on [N, 2D] -> [N, D].

silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x))) -> 4 flops (negate,
exp, add 1, div). Then one multiply by `up`. Total 5 flops per element
of the output tensor.
"""


def flops(shapes, dtype) -> float:
    N = shapes["N"]
    D = shapes["D"]
    return 5.0 * N * D
