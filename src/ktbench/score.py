"""
Scoring.

Final score = correctness_rate × stress_pass_rate × sol_score

All three factors are in [0, 1]:
  correctness_rate  — fraction of structured cases passing (all-or-nothing gate in practice)
  stress_pass_rate  — fraction of stress trials passing
  sol_score         — fraction of target HW theoretical ceiling achieved (physics-bounded)

A no-op kernel that passes structured tests:  1 × 0 × 0 = 0
A correct kernel that hardcodes one shape:    1 × ~0 × ~0 = 0
A correct, robust, hardware-saturating kernel: → close to 1

speedup_vs_ref is displayed on the leaderboard but is NOT part of the gated score
because it can be manipulated by providing a slow reference_tgt.
"""

from __future__ import annotations


def compute_final_score(
    correctness_rate: float,
    stress_pass_rate: float,
    sol_score: float,
) -> float:
    """
    Compute the primary ranking score.

    sol_score = -1 means SOL was not measured (no flops/bytes_accessed provided
    and Nsight not available). In that case we fall back to a timing-only partial
    score so the candidate is not penalized for missing profiling infrastructure.
    """
    if sol_score < 0:
        # Timing-only fallback: score is correctness × stress only, capped at 0.5
        # so hardware-saturating kernels are always ranked higher.
        return round(min(correctness_rate * stress_pass_rate * 0.5, 0.5), 6)

    return round(correctness_rate * stress_pass_rate * sol_score, 6)


def leaderboard_row(
    problem_id: str,
    model_name: str,
    correctness_rate: float,
    stress_pass_rate: float,
    sol_score: float,
    speedup_vs_ref: float,
    final_score: float,
    *,
    occupancy_pct: float = -1,
    memory_ratio: float = -1,
    fusion_ratio: float = -1,
    energy_ratio: float = -1,
    utilization_blocked: bool = False,
    excessive_speedup_flagged: bool = False,
    hardware: str = "",
) -> dict:
    """Produce a flat dict suitable for a leaderboard CSV / JSON row."""
    return {
        "problem_id": problem_id,
        "model": model_name,
        "hardware": hardware,
        # Primary ranking
        "final_score": round(final_score, 6),
        # Component scores
        "correctness_rate": round(correctness_rate, 4),
        "stress_pass_rate": round(stress_pass_rate, 4),
        "sol_score": round(sol_score, 4),
        # Context (not gated)
        "speedup_vs_ref": round(speedup_vs_ref, 4),
        "occupancy_pct": round(occupancy_pct, 2),
        "memory_ratio": round(memory_ratio, 4),
        "fusion_ratio": round(fusion_ratio, 4),
        "energy_ratio": round(energy_ratio, 4),
        # Flags
        "utilization_blocked": utilization_blocked,
        "excessive_speedup_flagged": excessive_speedup_flagged,
    }
