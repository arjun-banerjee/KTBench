"""
Top-level evaluation harness: eval_translation().

Pipeline:
  [1] Static analysis (antihack.py)       — blocks hard exploits before any GPU use
  [2] Compilation check                   — DSL-specific load
  [3] Correctness: structured test suite  — fixed shapes, random values
  [4] Utilization gate                    — block suspected no-ops
  [5] Stress testing                      — random shapes AND random values
  [6] Performance timing                  — candidate vs reference_tgt on same HW
  [7] Score                               — correctness × stress × SOL

Score = correctness_rate × stress_pass_rate × sol_score
speedup_vs_ref is displayed but not part of the gated score.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ktbench.eval.antihack import (
    StaticCheckResult,
    check_excessive_speedup,
    static_check,
    utilization_gate,
)
from ktbench.eval.correctness import (
    CorrectnessResult,
    StressResult,
    run_stress_suite,
    run_structured_cases,
)
from ktbench.eval.performance import PerformanceResult, measure_performance
from ktbench.problem import Problem
from ktbench.registry.dsl import DSLLoadError, load_model_class
from ktbench.registry.hardware import get_hw_spec


def _reference_at_fault(ref_model: Any, inputs: list, original_exc: Exception) -> bool:
    """Best-effort attribution for a perf-measurement failure.

    Try calling the reference once with the same inputs. If that raises,
    the reference is broken. If it succeeds, the failure was on the
    candidate side (or downstream timing / nsight / memory measurement).
    Used to make the error message in the trace name the right actor.
    """
    try:
        with torch.no_grad():
            ref_model.forward(*inputs)
        return False
    except Exception:
        return True
from ktbench.score import compute_final_score


@dataclass
class TranslationResult:
    # ── Identity ──────────────────────────────────────────────────────────────
    problem_id: str = ""
    tgt_dsl: str = ""
    tgt_hw: str = ""
    hardware_name: str = ""    # torch.cuda.get_device_name()

    # ── Pipeline stage outcomes ───────────────────────────────────────────────
    static_check: StaticCheckResult | None = None
    compiled: bool = False
    compile_error: str | None = None

    correctness: CorrectnessResult | None = None
    stress: StressResult | None = None
    performance: PerformanceResult | None = None

    # ── Anti-hack flags ───────────────────────────────────────────────────────
    utilization_blocked: bool = False
    utilization_reason: str = ""
    excessive_speedup_flagged: bool = False
    excessive_speedup_reason: str = ""

    # ── Final scores ──────────────────────────────────────────────────────────
    correctness_rate: float = 0.0
    stress_pass_rate: float = 0.0
    sol_score: float = 0.0
    speedup_vs_ref: float = -1.0
    final_score: float = 0.0

    # ── Timing metadata ───────────────────────────────────────────────────────
    eval_wall_time_s: float = -1.0

    def summary(self) -> dict:
        out: dict = {
            "problem_id": self.problem_id,
            "tgt_dsl": self.tgt_dsl,
            "tgt_hw": self.tgt_hw,
            "hardware": self.hardware_name,
            "compiled": self.compiled,
            "correctness_rate": round(self.correctness_rate, 4),
            "stress_pass_rate": round(self.stress_pass_rate, 4),
            "sol_score": round(self.sol_score, 4),
            "speedup_vs_ref": round(self.speedup_vs_ref, 4),
            "final_score": round(self.final_score, 4),
            "utilization_blocked": self.utilization_blocked,
            "excessive_speedup_flagged": self.excessive_speedup_flagged,
        }
        if self.compile_error:
            out["compile_error"] = self.compile_error
        if self.static_check:
            out["static_errors"] = self.static_check.errors
            out["static_warnings"] = self.static_check.warnings
        if self.correctness:
            out["correctness_detail"] = self.correctness.summary()
        if self.stress:
            out["stress_detail"] = self.stress.summary()
        if self.performance:
            out["performance"] = self.performance.summary()
        return out


def eval_translation(
    candidate_src: str,
    problem: Problem,
    device: int | torch.device = 0,
    *,
    global_seed: int | None = None,
    n_warmup: int = 5,
    n_timing_trials: int = 20,
    utilization_floor_pct: float = 0.05,
    excessive_speedup_threshold: float = 50.0,
    # Optional: if known, improves SOL accuracy
    flops: float | None = None,
    bytes_accessed: float | None = None,
    verbose: bool = False,
) -> TranslationResult:
    """
    Evaluate a candidate kernel translation against its problem spec.

    candidate_src: complete Python source string implementing ModelNew.
    problem: Problem object loaded via ktbench.problem.load_problem().
    device: CUDA device index or torch.device.
    global_seed: RNG seed base for input generation. Randomized per run if None.
    """
    t_start = time.monotonic()

    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")

    if global_seed is None:
        global_seed = int(time.time_ns() & 0xFFFFFFFF)

    result = TranslationResult(
        problem_id=problem.meta.problem_id,
        tgt_dsl=problem.meta.tgt_dsl,
        tgt_hw=problem.meta.tgt_hw,
        hardware_name=torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
    )

    hw_spec = get_hw_spec(problem.meta.tgt_hw)

    # ── [1] Static analysis ──────────────────────────────────────────────────
    if verbose:
        print("[1/6] Static analysis")
    sc = static_check(candidate_src, problem.meta.tgt_dsl)
    result.static_check = sc
    if not sc.passed:
        if verbose:
            print(f"  BLOCKED: {sc.errors}")
        result.eval_wall_time_s = time.monotonic() - t_start
        result.final_score = 0.0
        return result

    # ── [2] Compilation ──────────────────────────────────────────────────────
    if verbose:
        print("[2/6] Compilation")
    try:
        cand_cls = load_model_class(candidate_src, problem.meta.tgt_dsl)
        result.compiled = True
    except DSLLoadError as e:
        result.compile_error = str(e)
        if verbose:
            print(f"  compile failed: {e}")
        result.eval_wall_time_s = time.monotonic() - t_start
        return result

    # Load oracle and reference classes (always present on eval machine)
    try:
        oracle_cls = load_model_class(problem.oracle_src, "pytorch")
    except DSLLoadError as e:
        result.compile_error = f"oracle load failed: {e}"
        result.eval_wall_time_s = time.monotonic() - t_start
        return result

    try:
        ref_cls = load_model_class(problem.reference_tgt_src, problem.meta.tgt_dsl)
    except DSLLoadError as e:
        result.compile_error = f"reference_tgt load failed: {e}"
        result.eval_wall_time_s = time.monotonic() - t_start
        return result

    # ── [3] Correctness: structured cases ────────────────────────────────────
    if verbose:
        print("[3/6] Structured correctness cases")
    correctness = run_structured_cases(
        cand_cls, oracle_cls, problem, device, global_seed=global_seed
    )
    result.correctness = correctness
    result.correctness_rate = correctness.correctness_rate

    if not correctness.all_passed:
        if verbose:
            print(f"  FAIL: {correctness.correctness_rate:.0%} cases passed")
        result.eval_wall_time_s = time.monotonic() - t_start
        result.final_score = 0.0
        return result
    if verbose:
        print(f"  PASS: all {len(correctness.cases)} cases")

    # ── [4] Stress testing ───────────────────────────────────────────────────
    if verbose:
        print("[4/6] Stress testing (random shapes + values)")
    stress = run_stress_suite(
        cand_cls, oracle_cls, problem, device, global_seed=global_seed
    )
    result.stress = stress
    result.stress_pass_rate = stress.pass_rate
    if verbose:
        print(f"  stress: {stress.trials_passed}/{stress.trials_total} passed "
              f"({stress.pass_rate:.0%})")

    # ── [5] Performance timing ───────────────────────────────────────────────
    if verbose:
        print("[5/6] Performance measurement")

    # Use the first structured case's inputs for timing
    first_case = problem.test_suite.cases[0]
    import numpy as np
    timing_rng = np.random.default_rng(global_seed ^ 0xCAFEBABE)
    timing_inputs = problem.make_inputs(
        first_case.shapes, first_case.dtype, timing_rng, device
    )

    # If the problem ships a perf.py with a flops formula, use it.
    # Callers can still pass flops= explicitly to override.
    if flops is None and problem.flops_fn is not None:
        try:
            flops = float(problem.flops_fn(first_case.shapes, first_case.dtype))
        except Exception as e:
            if verbose:
                print(f"  warn: problem.flops_fn raised {type(e).__name__}: {e}; "
                      f"falling back to memory-util-only SOL")

    cand_model = cand_cls()
    if hasattr(cand_model, "to"):
        cand_model = cand_model.to(device)

    ref_model = ref_cls()
    if hasattr(ref_model, "to"):
        ref_model = ref_model.to(device)

    # Wrap perf measurement so a broken reference baseline does not
    # surface as an opaque "unexpected error" attributed to the
    # candidate. The candidate already passed correctness + stress
    # at this point; the perf path can only fail because the
    # reference_tgt is broken (JIT compile, runtime error, OOM) or
    # the candidate misbehaves under repeated invocation. Either way
    # the diagnostic should name the failing actor.
    try:
        perf = measure_performance(
            cand_model, ref_model, timing_inputs, hw_spec, device,
            n_warmup=n_warmup, n_trials=n_timing_trials,
            flops=flops, bytes_accessed=bytes_accessed,
        )
    except Exception as e:
        ref_failed = _reference_at_fault(ref_model, timing_inputs, e)
        which = "reference_tgt" if ref_failed else "candidate"
        result.compile_error = (
            f"{which} failed during performance measurement: "
            f"{type(e).__name__}: {e}"
        )
        result.sol_score = -1.0
        result.final_score = 0.0
        result.eval_wall_time_s = time.monotonic() - t_start
        return result
    result.performance = perf
    result.sol_score = perf.sol.get("sol_score", 0.0)
    result.speedup_vs_ref = perf.speedup_vs_ref

    # ── Utilization gate ─────────────────────────────────────────────────────
    gate_passes, gate_reason = utilization_gate(perf.sol, utilization_floor_pct)
    if not gate_passes:
        result.utilization_blocked = True
        result.utilization_reason = gate_reason
        if verbose:
            print(f"  BLOCKED by utilization gate: {gate_reason}")
        result.correctness_rate = correctness.correctness_rate
        result.final_score = 0.0
        result.eval_wall_time_s = time.monotonic() - t_start
        return result

    # ── Excessive speedup flag ────────────────────────────────────────────────
    flagged, flag_reason = check_excessive_speedup(
        perf.speedup_vs_ref, excessive_speedup_threshold
    )
    result.excessive_speedup_flagged = flagged
    result.excessive_speedup_reason = flag_reason
    if flagged and verbose:
        print(f"  FLAG: {flag_reason}")

    # ── [6] Final score ──────────────────────────────────────────────────────
    if verbose:
        print("[6/6] Computing final score")

    result.final_score = compute_final_score(
        correctness_rate=result.correctness_rate,
        stress_pass_rate=result.stress_pass_rate,
        sol_score=result.sol_score,
    )

    result.eval_wall_time_s = time.monotonic() - t_start
    if verbose:
        print(f"  final_score={result.final_score:.4f}  "
              f"(correctness={result.correctness_rate:.2f} × "
              f"stress={result.stress_pass_rate:.2f} × "
              f"SOL={result.sol_score:.4f})")

    return result
