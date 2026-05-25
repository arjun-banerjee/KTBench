"""Flop count for BF16 GEMV: y = W @ x, W[N,K], x[K] → y[N].

Each output element y[i] = dot(W[i], x) = K multiply-adds = 2*K flops.
Total: 2 * N * K flops.
"""


def flops(shapes, dtype) -> float:
    N = shapes["N"]
    K = shapes["K"]
    return 2.0 * N * K
