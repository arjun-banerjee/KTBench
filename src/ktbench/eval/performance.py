"""
Performance measurement: timing, SOL, memory, kernel launches, energy.

All measurements compare the *candidate* against *reference_tgt* on the same
target hardware in the same eval run. There is no comparison to the source kernel.

Timing uses CUDA events (cold-cache, L2-flushed) for NVIDIA/AMD.
SOL is computed from timing + HW spec sheet — bounded by physics, not gameable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ktbench.registry.hardware import HardwareSpec, compute_sol


# ── L2 cache flushing ─────────────────────────────────────────────────────────

def _flush_l2(device: torch.device) -> None:
    """Thrash L2 cache with a large write to ensure cold-cache timing."""
    # 256 MB write — larger than any L2 in the eval fleet (H200: 90 MB, A100: 40 MB)
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(0)
    del dummy


# ── CUDA event timer ──────────────────────────────────────────────────────────

def time_model(
    model: Any,
    inputs: list,
    device: torch.device,
    n_warmup: int = 5,
    n_trials: int = 20,
) -> dict:
    """
    Time model.forward(*inputs) using CUDA events.

    Each trial flushes L2 cache first (cold-cache measurement).
    Returns timing stats in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for timing")

    torch.cuda.synchronize(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model.forward(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
    torch.cuda.synchronize(device)

    elapsed: list[float] = []
    with torch.no_grad():
        for _ in range(n_trials):
            _flush_l2(device)
            trial_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
            torch.cuda.synchronize(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model.forward(*trial_inputs)
            end.record()
            torch.cuda.synchronize(device)

            elapsed.append(start.elapsed_time(end))  # ms

    arr = np.array(elapsed)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "n_trials": n_trials,
        "hardware": torch.cuda.get_device_name(device),
    }


# ── Memory measurement ────────────────────────────────────────────────────────

def measure_memory(model: Any, inputs: list, device: torch.device) -> dict:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        model.forward(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
    torch.cuda.synchronize(device)

    peak = torch.cuda.max_memory_allocated(device)
    return {
        "peak_memory_bytes": peak,
        "peak_memory_mb": round(peak / (1024 * 1024), 2),
    }


# ── Kernel launch count ───────────────────────────────────────────────────────

def measure_kernel_launches(model: Any, inputs: list, device: torch.device) -> dict:
    try:
        with torch.no_grad():
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                model.forward(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
        torch.cuda.synchronize(device)

        cuda_events = [
            {"name": e.key, "cuda_time_us": e.cuda_time_total, "calls": e.count}
            for e in prof.function_events
            if e.cuda_time_total > 0 and not e.key.startswith("cudaDevice")
        ]
        cuda_events.sort(key=lambda x: x["cuda_time_us"], reverse=True)
        return {
            "num_kernels": len(cuda_events),
            "kernel_breakdown": cuda_events[:10],
            "total_cuda_time_us": sum(e["cuda_time_us"] for e in cuda_events),
        }
    except Exception as e:
        return {"num_kernels": -1, "error": str(e)}


# ── Energy (NVIDIA only via NVML) ─────────────────────────────────────────────

def measure_energy(
    model: Any,
    inputs: list,
    device: torch.device,
    n_trials: int = 50,
) -> dict:
    """Measure energy in millijoules. Returns empty dict on non-NVIDIA hardware."""
    if "nvidia" not in torch.cuda.get_device_name(device).lower():
        return {}
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)

        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        with torch.no_grad():
            for _ in range(n_trials):
                model.forward(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
        torch.cuda.synchronize(device)
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        energy_mj = (end_energy - start_energy) / n_trials
        power_info = pynvml.nvmlDeviceGetPowerUsage(handle)
        return {
            "energy_mj": round(energy_mj, 4),
            "avg_power_w": round(power_info / 1000.0, 2),
        }
    except Exception:
        return {}


# ── Combined performance measurement ─────────────────────────────────────────

@dataclass
class PerformanceResult:
    # Candidate
    runtime: dict = field(default_factory=dict)        # timing stats
    memory: dict = field(default_factory=dict)
    kernel_launches: dict = field(default_factory=dict)
    energy: dict = field(default_factory=dict)
    sol: dict = field(default_factory=dict)

    # Reference (reference_tgt.py on same HW)
    ref_runtime: dict = field(default_factory=dict)
    ref_memory: dict = field(default_factory=dict)
    ref_kernel_launches: dict = field(default_factory=dict)
    ref_energy: dict = field(default_factory=dict)

    # Derived
    speedup_vs_ref: float = -1.0
    memory_ratio: float = -1.0
    fusion_ratio: float = -1.0
    energy_ratio: float = -1.0

    def summary(self) -> dict:
        return {
            "runtime_mean_ms": self.runtime.get("mean_ms", -1),
            "ref_runtime_mean_ms": self.ref_runtime.get("mean_ms", -1),
            "speedup_vs_ref": round(self.speedup_vs_ref, 4),
            "sol_score": self.sol.get("sol_score", -1),
            "sol_bottleneck": self.sol.get("bottleneck", "unknown"),
            "compute_util": self.sol.get("compute_util", -1),
            "memory_util": self.sol.get("memory_util", -1),
            "peak_memory_bytes": self.memory.get("peak_memory_bytes", -1),
            "memory_ratio": round(self.memory_ratio, 4),
            "num_kernels": self.kernel_launches.get("num_kernels", -1),
            "ref_num_kernels": self.ref_kernel_launches.get("num_kernels", -1),
            "fusion_ratio": round(self.fusion_ratio, 4),
            "energy_mj": self.energy.get("energy_mj", -1),
            "energy_ratio": round(self.energy_ratio, 4),
        }


def measure_performance(
    cand_model: Any,
    ref_model: Any,
    inputs: list,
    hw_spec: HardwareSpec,
    device: torch.device,
    n_warmup: int = 5,
    n_trials: int = 20,
    *,
    flops: float | None = None,
    bytes_accessed: float | None = None,
) -> PerformanceResult:
    """
    Time candidate AND reference on the same hardware with the same inputs.
    Compute SOL relative to HW ceiling.

    flops / bytes_accessed: if provided, used for exact SOL computation.
    If neither is available, timing-only stats are recorded and sol_score = -1.
    """
    result = PerformanceResult()

    result.runtime = time_model(cand_model, inputs, device, n_warmup, n_trials)
    result.ref_runtime = time_model(ref_model, inputs, device, n_warmup, n_trials)

    cand_ms = result.runtime["mean_ms"]
    ref_ms = result.ref_runtime["mean_ms"]
    if cand_ms > 0 and ref_ms > 0:
        result.speedup_vs_ref = ref_ms / cand_ms

    result.sol = compute_sol(
        cand_ms, hw_spec,
        flops=flops,
        bytes_accessed=bytes_accessed,
    )

    result.memory = measure_memory(cand_model, inputs, device)
    result.ref_memory = measure_memory(ref_model, inputs, device)
    cand_mem = result.memory.get("peak_memory_bytes", 0)
    ref_mem = result.ref_memory.get("peak_memory_bytes", 0)
    result.memory_ratio = cand_mem / ref_mem if ref_mem > 0 else -1.0

    result.kernel_launches = measure_kernel_launches(cand_model, inputs, device)
    result.ref_kernel_launches = measure_kernel_launches(ref_model, inputs, device)
    cand_k = result.kernel_launches.get("num_kernels", 0)
    ref_k = result.ref_kernel_launches.get("num_kernels", 0)
    result.fusion_ratio = ref_k / cand_k if cand_k > 0 else -1.0

    result.energy = measure_energy(cand_model, inputs, device, n_trials=50)
    result.ref_energy = measure_energy(ref_model, inputs, device, n_trials=50)
    cand_e = result.energy.get("energy_mj", 0)
    ref_e = result.ref_energy.get("energy_mj", 0)
    result.energy_ratio = cand_e / ref_e if ref_e > 0 else -1.0

    return result
