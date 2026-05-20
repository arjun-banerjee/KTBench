"""Flop count for the WKV token mixing recurrence on [B, H, T, N].

WKV's standard formulation maintains numerator a and denominator b and
emits `num / den` per token after blending with an exponential decay
and a bonus-weighted current token. Per (b, h, t, n) the kernel does
roughly:

  - 2 exp (decay, bonus weighting)
  - 4 muls (decay * a, decay * b, exp * v, exp * 1)
  - 4 adds (running sum updates and combining current token)
  - 1 div (num / den) for the output

Counted conservatively as 11 flops per element. The constant matters
only for SOL; a candidate that uses a different mathematically
equivalent recurrence (e.g. log-domain) lands on the same denominator.
"""


def flops(shapes, dtype) -> float:
    B = shapes["B"]
    H = shapes["H"]
    T = shapes["T"]
    N = shapes["N"]
    return 11.0 * B * H * T * N
