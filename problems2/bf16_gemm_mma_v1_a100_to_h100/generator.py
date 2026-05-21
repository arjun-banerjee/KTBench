"""Generator for bf16_gemm_mma_v1_a100_to_h100."""
import torch


def generate(shapes: dict, dtype: str) -> dict:
    M = shapes["M"]
    K = shapes["K"]
    N = shapes["N"]
    assert M % 128 == 0 and K % 64 == 0 and N % 128 == 0, \
        f"M={M}, K={K}, N={N} must be multiples of 128/64/128"
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=dt, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=dt, device="cuda") * 0.1
    return {"A": A, "B": B}
