"""
Anti-reward-hacking checks.

Layer 1 — Static analysis (AST + regex):
  Runs before any execution. Flags patterns that are definitionally exploits:
  try/except fallback, pass-through, sys.exit, grader imports, stack introspection.

Layer 2 — Utilization gate (post-performance):
  A kernel with near-zero GPU utilization cannot be doing real work.
  Blocks the performance score if SOL_compute AND SOL_dram are both below the floor.

Layer 3 — Excessive speedup flag:
  If candidate is >50× faster than the handwritten reference, flag for audit.
  Does not hard-block (legitimate extreme speedups exist) but requires review.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


# ── Layer 1: Static analysis ──────────────────────────────────────────────────

@dataclass
class StaticCheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)    # hard blocks
    warnings: list[str] = field(default_factory=list)  # logged but don't block


# Patterns that are unconditional exploits
_HARD_BLOCK_PATTERNS: list[tuple[str, str]] = [
    # Process manipulation
    (r"\bos\.kill\b",              "os.kill() — process manipulation"),
    (r"\bos\.killpg\b",            "os.killpg() — process group kill"),
    (r"\bsubprocess\b",            "subprocess usage — process spawning"),
    (r"\bsignal\.raise_signal\b",  "signal.raise_signal()"),
    # Stack / frame introspection
    (r"\bsys\._getframe\b",        "sys._getframe() — stack introspection"),
    (r"\binspect\.stack\b",        "inspect.stack() — stack introspection"),
    (r"\binspect\.currentframe\b", "inspect.currentframe()"),
    (r"\bctypes\b",                "ctypes usage"),
    # Grader / eval module access
    (r"\bktbench\b",               "importing ktbench eval internals"),
    (r"\bkernelbench\b",           "importing kernelbench eval internals"),
    # Sys exit / exception to kill evaluator
    (r"\bsys\.exit\b",             "sys.exit() — evaluator kill attempt"),
    (r"\bos\._exit\b",             "os._exit() — evaluator kill attempt"),
    # File system snooping (oracle tensors, test_suite)
    (r"oracle_tensors",            "direct oracle_tensor path reference"),
    (r"test_suite\.toml",          "direct test_suite.toml reference"),
]

# Patterns that indicate likely exploits but may have rare legitimate uses
_SOFT_WARN_PATTERNS: list[tuple[str, str]] = [
    (r"\btry\s*:",     "try/except block — possible fallback bypass"),
    (r"\bexcept\s*[:\w]", "except clause — possible fallback bypass"),
    (r"\bpass\b",      "bare pass — possible inheritance bypass"),
]

# DSL-specific structural requirements: the submitted code must show real work
_DSL_REQUIREMENTS: dict[str, list[tuple[str, str]]] = {
    "cuda": [
        (r"__global__",          "missing __global__ kernel definition"),
        (r"load_inline|cpp_extension", "missing load_inline/cpp_extension"),
    ],
    "hip": [
        (r"__global__",          "missing __global__ kernel definition"),
        (r"load_inline|cpp_extension", "missing load_inline/cpp_extension"),
    ],
    "triton": [
        (r"@triton\.jit",        "missing @triton.jit decorator"),
        (r"\btl\.",              "no tl.* operations found"),
    ],
    "tilelang": [
        (r"@T\.prim_func",       "missing @T.prim_func decorator"),
    ],
    "nki": [
        (r"@nki\.jit",           "missing @nki.jit decorator"),
        (r"\bnl\.",              "no nl.* operations found"),
    ],
    "pallas": [
        (r"@functools\.partial|pl\.pallas_call", "missing pallas_call"),
    ],
}


def static_check(src: str, dsl: str) -> StaticCheckResult:
    """
    Run static analysis on candidate source code.
    Returns StaticCheckResult with errors (hard blocks) and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Strip Python comments for pattern matching (not AST)
    stripped = _strip_comments(src)

    for pattern, label in _HARD_BLOCK_PATTERNS:
        if re.search(pattern, stripped):
            errors.append(f"BLOCKED: {label}")

    for pattern, label in _SOFT_WARN_PATTERNS:
        if re.search(pattern, stripped):
            warnings.append(f"WARNING: {label}")

    # DSL structural requirements
    reqs = _DSL_REQUIREMENTS.get(dsl.lower(), [])
    for pattern, label in reqs:
        if not re.search(pattern, stripped):
            errors.append(f"MISSING: {label}")

    # AST sanity: must be parseable Python (for all DSLs using Python wrapper)
    if dsl.lower() not in ("mojo",):
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e}")

    return StaticCheckResult(passed=len(errors) == 0, errors=errors, warnings=warnings)


def _strip_comments(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if "#" in line:
            line = line[: line.index("#")]
        lines.append(line)
    return "\n".join(lines)


# ── Layer 2: Utilization gate ─────────────────────────────────────────────────

def utilization_gate(sol_stats: dict, floor_pct: float = 2.0) -> tuple[bool, str]:
    """
    Return (passes, reason).
    A kernel passes the gate if at least one throughput axis (compute or memory)
    exceeds floor_pct percent of the hardware ceiling.

    Near-zero utilization on both axes is physically impossible for a kernel
    doing real work — it indicates a no-op or trivially hardcoded result.
    """
    if not sol_stats or sol_stats.get("sol_score", -1) < 0:
        return True, "sol not measured; gate skipped"

    compute_pct = sol_stats.get("compute_util", 0.0) * 100.0
    memory_pct = sol_stats.get("memory_util", 0.0) * 100.0

    if compute_pct >= floor_pct or memory_pct >= floor_pct:
        return True, f"compute={compute_pct:.1f}% memory={memory_pct:.1f}%"

    return False, (
        f"suspected_noop: compute={compute_pct:.1f}% memory={memory_pct:.1f}% "
        f"both below floor {floor_pct}%"
    )


# ── Layer 3: Excessive speedup flag ──────────────────────────────────────────

def check_excessive_speedup(
    speedup_vs_ref: float,
    threshold: float = 50.0,
) -> tuple[bool, str]:
    """
    Flag (but do not block) if candidate is suspiciously faster than the reference.
    Returns (flagged, reason).
    """
    if speedup_vs_ref <= 0:
        return False, ""
    if speedup_vs_ref > threshold:
        return True, (
            f"excessive speedup {speedup_vs_ref:.1f}× vs reference "
            f"(threshold {threshold}×) — manual audit recommended"
        )
    return False, ""
