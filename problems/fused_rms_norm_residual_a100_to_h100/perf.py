"""Flop count for fused RMSNorm + residual on [B, L, D].

Per row of D elements:
  D squares (x * x)
  D-1 adds for the sum reduction (count as D)
  1 div for the mean
  1 add for the eps
  1 rsqrt
  D muls to scale (x * rsqrt)
  D muls to apply the weight
  D adds for the residual

Total per row: 4 * D + 3 + D = 5 * D + 3. Across [B, L] rows:
"""


def flops(shapes, dtype) -> float:
    B = shapes["B"]
    L = shapes["L"]
    D = shapes["D"]
    return float(B) * L * (5.0 * D + 3.0)
