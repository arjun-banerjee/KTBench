"""Flop count for online softmax over rows of shape [N, D].

Per row, the kernel does, on each of D elements: one subtract (x - max),
one exp, one add into the running sum, and one divide to normalise.
Plus a max reduction across the D elements (D-1 comparisons, treated as
D for simplicity). That is 5 flops per element.
"""


def flops(shapes, dtype) -> float:
    N = shapes["N"]
    D = shapes["D"]
    return 5.0 * N * D
