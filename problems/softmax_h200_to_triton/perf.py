"""Flop count for online softmax over rows of shape [N, D].

Per row: D elements * (sub + exp + add + div) + D-element max reduction
= 5 flops per element.
"""


def flops(shapes, dtype) -> float:
    N = shapes["N"]
    D = shapes["D"]
    return 5.0 * N * D
