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


def _ideal_bytes_accessed(inputs: list, output: Any) -> float:
    """Sum the bytes of every input and output tensor.

    This is the "speed of light" memory denominator for SOL: the
    minimum amount of DRAM traffic a kernel could do while still
    producing the right answer (read each input once, write each
    output once). Real kernels do more, so memory_util in (0, 1] is
    the fraction of that ideal the kernel achieved. A kernel that
    moves *fewer* bytes than this would have to be skipping work,
    which the antihack gate already catches via the utilisation
    floor.
    """
    total = 0.0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.element_size() * t.numel()
    outputs = output if isinstance(output, (list, tuple)) else [output]
    for t in outputs:
        if isinstance(t, torch.Tensor):
            total += t.element_size() * t.numel()
    return total


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
    Time candidate AND reference (if available) on the same hardware.
    ref_model may be None when the reference kernel is unavailable; in that
    case speedup_vs_ref and ref_* fields are left at their defaults (-1/empty)
    but candidate SOL is still computed.

    flops / bytes_accessed: if provided, used for exact SOL computation.
    If bytes_accessed is missing, it is estimated automatically from
    input + output tensor sizes (the ideal SOL denominator). This means
    SOL is always reported; SOL is only -1 if timing itself failed.
    """
    result = PerformanceResult()

    result.runtime = time_model(cand_model, inputs, device, n_warmup, n_trials)

    if ref_model is not None:
        result.ref_runtime = time_model(ref_model, inputs, device, n_warmup, n_trials)

    cand_ms = result.runtime["mean_ms"]
    ref_ms = result.ref_runtime.get("mean_ms", 0)
    if cand_ms > 0 and ref_ms > 0:
        result.speedup_vs_ref = ref_ms / cand_ms

    if bytes_accessed is None:
        with torch.no_grad():
            cand_out = cand_model.forward(*inputs)
        bytes_accessed = _ideal_bytes_accessed(inputs, cand_out)

    result.sol = compute_sol(
        cand_ms, hw_spec,
        flops=flops,
        bytes_accessed=bytes_accessed,
    )

    result.memory = measure_memory(cand_model, inputs, device)
    if ref_model is not None:
        result.ref_memory = measure_memory(ref_model, inputs, device)
    cand_mem = result.memory.get("peak_memory_bytes", 0)
    ref_mem = result.ref_memory.get("peak_memory_bytes", 0)
    result.memory_ratio = cand_mem / ref_mem if ref_mem > 0 else -1.0

    result.kernel_launches = measure_kernel_launches(cand_model, inputs, device)
    if ref_model is not None:
        result.ref_kernel_launches = measure_kernel_launches(ref_model, inputs, device)
    cand_k = result.kernel_launches.get("num_kernels", 0)
    ref_k = result.ref_kernel_launches.get("num_kernels", 0)
    result.fusion_ratio = ref_k / cand_k if cand_k > 0 else -1.0

    result.energy = measure_energy(cand_model, inputs, device, n_trials=50)
    if ref_model is not None:
        result.ref_energy = measure_energy(ref_model, inputs, device, n_trials=50)
    cand_e = result.energy.get("energy_mj", 0)
    ref_e = result.ref_energy.get("energy_mj", 0)
    result.energy_ratio = cand_e / ref_e if ref_e > 0 else -1.0

    return result
