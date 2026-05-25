"""Hardware registry — A100 and H100 only."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    key: str
    fp16_tflops: float
    bf16_tflops: float
    fp32_tflops: float
    dram_bw_gbs: float
    l2_cache_mb: float
    sm_count: int


_REGISTRY: dict[str, HardwareSpec] = {
    "nvidia_a100_sxm": HardwareSpec(
        key="nvidia_a100_sxm",
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        fp32_tflops=19.5,
        dram_bw_gbs=2000.0,
        l2_cache_mb=40.0,
        sm_count=108,
    ),
    "nvidia_h100_sxm": HardwareSpec(
        key="nvidia_h100_sxm",
        fp16_tflops=989.0,
        bf16_tflops=989.0,
        fp32_tflops=67.0,
        dram_bw_gbs=3350.0,
        l2_cache_mb=50.0,
        sm_count=132,
    ),
}


def get_hw_spec(key: str) -> HardwareSpec:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown hardware {key!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def compute_sol(
    runtime_ms: float,
    hw_spec: HardwareSpec,
    *,
    flops: float | None = None,
    bytes_accessed: float | None = None,
) -> dict:
    """
    Compute SOL (Speed-of-Light) score.
    SOL = max(compute_util, memory_util) where each is in [0, 1].
    """
    result: dict = {"sol_score": -1.0, "bottleneck": "unknown"}

    if runtime_ms <= 0:
        return result
    if flops is None and bytes_accessed is None:
        return result

    runtime_s = runtime_ms / 1000.0

    if flops is not None and hw_spec.fp16_tflops > 0:
        achieved_tflops = (flops / 1e12) / runtime_s
        compute_util = min(achieved_tflops / hw_spec.fp16_tflops, 1.0)
        result["achieved_tflops"] = round(achieved_tflops, 4)
        result["compute_util"] = round(compute_util, 4)
    else:
        compute_util = 0.0

    if bytes_accessed is not None and hw_spec.dram_bw_gbs > 0:
        achieved_bw_gbs = (bytes_accessed / 1e9) / runtime_s
        memory_util = min(achieved_bw_gbs / hw_spec.dram_bw_gbs, 1.0)
        result["achieved_bandwidth_gbs"] = round(achieved_bw_gbs, 4)
        result["memory_util"] = round(memory_util, 4)
    else:
        memory_util = 0.0

    sol = max(compute_util, memory_util)
    result["sol_score"] = round(sol, 4)
    result["bottleneck"] = "compute" if compute_util >= memory_util else "memory"
    return result
