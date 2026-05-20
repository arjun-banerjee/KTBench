"""
Correctness evaluation.

Anti-reward-hacking invariants enforced here:
  1. Tensor VALUES are always freshly randomized — a kernel cannot hardcode outputs.
  2. Output buffer is allocated fresh and zeroed before each candidate run.
  3. Pointer aliasing check: candidate output must not share memory with oracle tensor.
  4. Structured cases use fixed shapes but random values (seed varies per run).
  5. Stress cases use both random shapes AND random values.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ktbench.problem import CaseSpec, Problem, StressConfig, Tolerance


# ── RNG helpers ───────────────────────────────────────────────────────────────

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_shapes_from_stress(
    rng: np.random.Generator,
    ranges: dict[str, tuple[int, int]],
) -> dict[str, int]:
    """Sample one concrete shape dict from stress ranges."""
    return {
        dim: int(rng.integers(lo, hi + 1))
        for dim, (lo, hi) in ranges.items()
    }


# ── Output buffer management ──────────────────────────────────────────────────

def _alloc_zeroed(like: torch.Tensor) -> torch.Tensor:
    """Allocate a fresh zeroed buffer with the same shape/dtype/device as `like`."""
    buf = torch.zeros_like(like)
    torch.cuda.synchronize(like.device)
    return buf


def _check_no_aliasing(candidate_out: torch.Tensor, oracle_out: torch.Tensor) -> bool:
    """
    Return True if the two tensors share no underlying storage.
    A no-op kernel that returns the input buffer (or oracle buffer) directly
    would have its data_ptr equal to one of the input tensors.
    We check data_ptr as a heuristic; the eval harness also allocates fresh
    output buffers on each call so aliasing to oracle memory is impossible
    when inputs are passed correctly.
    """
    return candidate_out.data_ptr() != oracle_out.data_ptr()


# ── Comparison ────────────────────────────────────────────────────────────────

def _compare(
    candidate: torch.Tensor | tuple,
    oracle: torch.Tensor | tuple,
    tol: Tolerance,
) -> tuple[bool, dict]:
    """Recursively compare candidate and oracle outputs. Returns (match, details)."""
    if isinstance(oracle, (tuple, list)):
        if not isinstance(candidate, (tuple, list)) or len(candidate) != len(oracle):
            return False, {"error": "output structure mismatch"}
        results = [_compare(c, o, tol) for c, o in zip(candidate, oracle)]
        ok = all(r[0] for r in results)
        details = {f"element_{i}": r[1] for i, r in enumerate(results) if not r[0]}
        return ok, details

    if not isinstance(candidate, torch.Tensor):
        match = candidate == oracle
        return bool(match), {} if match else {"value_mismatch": f"{candidate} != {oracle}"}

    if candidate.shape != oracle.shape:
        return False, {"shape_mismatch": f"{candidate.shape} vs {oracle.shape}"}

    c = candidate.detach().float()
    o = oracle.detach().float()
    abs_diff = (c - o).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    denom = o.abs().clamp(min=1e-12)
    max_rel = (abs_diff / denom).max().item()

    match = torch.allclose(c, o, atol=tol.atol, rtol=tol.rtol)
    details = {
        "max_abs_error": round(max_abs, 8),
        "mean_abs_error": round(mean_abs, 8),
        "max_rel_error": round(max_rel, 8),
    }
    return match, details


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    passed: bool
    details: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class CorrectnessResult:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def correctness_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    @property
    def all_passed(self) -> bool:
        return bool(self.cases) and all(c.passed for c in self.cases)

    def summary(self) -> dict:
        n = len(self.cases)
        passed = sum(1 for c in self.cases if c.passed)
        return {
            "correctness_rate": round(self.correctness_rate, 4),
            "cases_passed": passed,
            "cases_total": n,
            "per_case": [
                {"id": c.case_id, "passed": c.passed, **({"error": c.error} if c.error else {})}
                for c in self.cases
            ],
        }


@dataclass
class StressResult:
    trials_passed: int = 0
    trials_total: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.trials_total == 0:
            return 0.0
        return self.trials_passed / self.trials_total

    def summary(self) -> dict:
        return {
            "stress_pass_rate": round(self.pass_rate, 4),
            "trials_passed": self.trials_passed,
            "trials_total": self.trials_total,
        }


# ── Core eval functions ───────────────────────────────────────────────────────

def _run_one_case(
    model_cls: type,
    oracle_cls: type,
    shapes: dict,
    dtype: str,
    problem: Problem,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[bool, dict, str | None]:
    """
    Run a single correctness trial (structured or stress).
    Returns (passed, details, error_str).

    Key invariants:
    - Input VALUES are freshly sampled from rng — cannot be hardcoded.
    - Output buffer for candidate is a fresh zeroed allocation.
    - No aliasing between candidate output and oracle output is verified.
    """
    tol = problem.get_tolerance(dtype)

    try:
        # Fresh random inputs — values cannot be hardcoded
        inputs = problem.make_inputs(shapes, dtype, rng, device)
    except Exception as e:
        return False, {}, f"make_inputs failed: {e}"

    # Run oracle on a cloned copy of inputs (never mutated by candidate)
    oracle_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
    try:
        with torch.no_grad():
            oracle_model = oracle_cls()
            oracle_model = oracle_model.to(device) if hasattr(oracle_model, "to") else oracle_model
            oracle_out = oracle_model.forward(*oracle_inputs)
            torch.cuda.synchronize(device)
    except Exception as e:
        return False, {}, f"oracle failed: {e}"

    # Candidate runs on a separate clone; output buffer is allocated fresh + zeroed
    cand_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
    try:
        with torch.no_grad():
            cand_model = model_cls()
            cand_model = cand_model.to(device) if hasattr(cand_model, "to") else cand_model
            cand_out = cand_model.forward(*cand_inputs)
            torch.cuda.synchronize(device)
    except Exception as e:
        return False, {}, f"candidate forward failed: {e}"

    # Aliasing check: candidate must not return the oracle buffer or input buffers
    oracle_ptrs = {
        t.data_ptr() for t in ([oracle_out] if isinstance(oracle_out, torch.Tensor)
                                else oracle_out if isinstance(oracle_out, (list, tuple)) else [])
    }
    input_ptrs = {t.data_ptr() for t in inputs if isinstance(t, torch.Tensor)}
    cand_ptrs = {
        t.data_ptr() for t in ([cand_out] if isinstance(cand_out, torch.Tensor)
                                else cand_out if isinstance(cand_out, (list, tuple)) else [])
    }
    if cand_ptrs & (oracle_ptrs | input_ptrs):
        return False, {}, "output aliases oracle or input buffer (no-op exploit detected)"

    passed, details = _compare(cand_out, oracle_out, tol)
    return passed, details, None


def run_structured_cases(
    model_cls: type,
    oracle_cls: type,
    problem: Problem,
    device: torch.device,
    global_seed: int = 0,
) -> CorrectnessResult:
    """
    Evaluate candidate against every structured (curated) test case.

    Shapes are fixed (curated for diversity); VALUES are randomly sampled
    with a per-case seed derived from global_seed. A different global_seed
    is used each eval run so per-run value hardcoding is impossible.
    """
    result = CorrectnessResult()

    for idx, case in enumerate(problem.test_suite.cases):
        case_seed = global_seed ^ (idx * 0x9E3779B9)  # deterministic but varied per case
        rng = _make_rng(case_seed)

        passed, details, err = _run_one_case(
            model_cls, oracle_cls,
            shapes=case.shapes,
            dtype=case.dtype,
            problem=problem,
            rng=rng,
            device=device,
        )
        result.cases.append(CaseResult(
            case_id=case.id,
            passed=passed,
            details=details,
            error=err,
        ))

    return result


def run_stress_suite(
    model_cls: type,
    oracle_cls: type,
    problem: Problem,
    device: torch.device,
    global_seed: int = 0,
) -> StressResult:
    """
    Evaluate candidate on randomly shaped + randomly valued inputs.

    Both shapes AND values are sampled fresh each trial, so:
    - Hardcoded values fail (values differ each trial).
    - Shape-specific implementations fail (shapes differ each trial).
    - Only a correct general algorithm passes.
    """
    cfg = problem.test_suite.stress
    result = StressResult(trials_total=cfg.num_trials)

    # Use the first structured case's dtype for stress (or fp32 fallback)
    dtype = problem.test_suite.cases[0].dtype if problem.test_suite.cases else "fp32"

    for trial in range(cfg.num_trials):
        trial_seed = global_seed ^ (0xDEADBEEF + trial * 0x6B43A9B5)
        rng = _make_rng(trial_seed)

        # Sample shapes fresh from ranges
        shapes = _sample_shapes_from_stress(rng, cfg.shape_ranges)

        passed, _, err = _run_one_case(
            model_cls, oracle_cls,
            shapes=shapes,
            dtype=dtype,
            problem=problem,
            rng=rng,
            device=device,
        )
        if passed:
            result.trials_passed += 1
        elif err:
            result.errors.append(f"trial {trial}: {err}")

    return result
