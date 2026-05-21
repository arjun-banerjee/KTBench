"""Oracle for bf16_gemm_warptile_a100_to_h100: C = A @ B in BF16."""
import torch


def run(inputs: dict) -> dict:
    A, B = inputs["A"], inputs["B"]
    C = torch.mm(A.float(), B.float()).to(A.dtype)
    return {"C": C}
