"""
Hardware registry.

Each entry specifies the theoretical peak throughput for SOL computation.
SOL = achieved_throughput / hw_peak  (0..1, bounded by physics).

Official eval fleet:  nvidia_h200_sxm (primary), nvidia_a100_sxm (secondary),
                      aws_trainium2 (NKI, availability TBD).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HardwareSpec:
    key: str
    vendor: str                      # "nvidia" | "amd" | "aws" | "google"
    fp16_tflops: float               # theoretical peak FP16 dense TFLOP/s
    bf16_tflops: float
    fp32_tflops: float
    dram_bw_gbs: float               # peak HBM bandwidth GB/s
    l2_cache_mb: float
    sm_count: int                    # SMs / CUs / NeuronCores
    compatible_dsls: tuple[str, ...] # DSLs that can target this HW
    in_eval_fleet: bool = False
    notes: str = ""


_REGISTRY: dict[str, HardwareSpec] = {}


def _reg(spec: HardwareSpec) -> HardwareSpec:
    _REGISTRY[spec.key] = spec
    return spec


# ── Official eval fleet ───────────────────────────────────────────────────────

H200_SXM = _reg(HardwareSpec(
    key="nvidia_h200_sxm",
    vendor="nvidia",
    fp16_tflops=989.0,
    bf16_tflops=989.0,
    fp32_tflops=67.0,
    dram_bw_gbs=4800.0,
    l2_cache_mb=50.0,
    sm_count=132,
    compatible_dsls=("cuda", "cute", "triton", "tilelang", "helion", "numba", "mojo"),
    in_eval_fleet=True,
    notes="GH100 die, HBM3e — primary eval target; up to 8× per node",
))

A100_SXM = _reg(HardwareSpec(
    key="nvidia_a100_sxm",
    vendor="nvidia",
    fp16_tflops=312.0,
    bf16_tflops=312.0,
    fp32_tflops=19.5,
    dram_bw_gbs=2000.0,
    l2_cache_mb=40.0,
    sm_count=108,
    compatible_dsls=("cuda", "cute", "triton", "tilelang", "helion", "numba", "mojo"),
    in_eval_fleet=True,
    notes="SXM4 Ampere — secondary eval target; up to 2× per node",
))

TRAINIUM2 = _reg(HardwareSpec(
    key="aws_trainium2",
    vendor="aws",
    fp16_tflops=832.0,
    bf16_tflops=832.0,
    fp32_tflops=104.0,
    dram_bw_gbs=820.0,
    l2_cache_mb=0.0,    # scratchpad-based, not L2
    sm_count=128,       # NeuronCores
    compatible_dsls=("nki",),
    in_eval_fleet=False,   # TBD
    notes="Trainium2; NKI only; eval fleet availability TBD",
))

# ── Extended registry (problem authoring; not all in eval fleet) ─────────────

_reg(HardwareSpec(
    key="nvidia_h100_sxm",
    vendor="nvidia",
    fp16_tflops=989.0,
    bf16_tflops=989.0,
    fp32_tflops=67.0,
    dram_bw_gbs=3350.0,
    l2_cache_mb=50.0,
    sm_count=132,
    compatible_dsls=("cuda", "cute", "triton", "tilelang", "helion", "numba", "mojo"),
    in_eval_fleet=False,
    notes="SXM5 Hopper — same compute as H200, less bandwidth",
))

_reg(HardwareSpec(
    key="amd_mi300x",
    vendor="amd",
    fp16_tflops=1307.0,
    bf16_tflops=1307.0,
    fp32_tflops=163.0,
    dram_bw_gbs=5300.0,
    l2_cache_mb=32.0,
    sm_count=304,
    compatible_dsls=("hip", "triton", "numba"),
    in_eval_fleet=False,
    notes="CDNA3; not in fleet — problems supported, eval requires external HW",
))

_reg(HardwareSpec(
    key="google_tpuv4",
    vendor="google",
    fp16_tflops=275.0,
    bf16_tflops=275.0,
    fp32_tflops=137.5,
    dram_bw_gbs=300.0,
    l2_cache_mb=0.0,
    sm_count=4,         # MXU cores
    compatible_dsls=("pallas",),
    in_eval_fleet=False,
    notes="TPUv4 — Pallas problems deferred to v2",
))


# ── API ───────────────────────────────────────────────────────────────────────

def get_hw_spec(key: str) -> HardwareSpec:
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown hardware key {key!r}. Known keys: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_hw_keys(fleet_only: bool = False) -> list[str]:
    return [k for k, v in _REGISTRY.items() if not fleet_only or v.in_eval_fleet]


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
    At least one of flops or bytes_accessed should be provided; with
    neither, both utilisation axes default to 0 and the result is the
    sentinel `sol_score = -1` so downstream callers (score.compute_final_score,
    antihack's noop gate) can fall back without misreading 0 as a real
    measurement.
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
