"""
Anti-reward-hacking checks for CUDA submissions.

Layer 1 — Static analysis (AST + regex): blocks hard exploits before any GPU use.
Layer 2 — Utilization gate (post-performance): blocks near-zero GPU utilization.
Layer 3 — Excessive speedup flag: flags suspiciously fast candidates for audit.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class StaticCheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_HARD_BLOCK_PATTERNS: list[tuple[str, str]] = [
    (r"\bos\.kill\b",              "os.kill() — process manipulation"),
    (r"\bos\.killpg\b",            "os.killpg() — process group kill"),
    (r"\bsubprocess\b",            "subprocess usage — process spawning"),
    (r"\bsignal\.raise_signal\b",  "signal.raise_signal()"),
    (r"\bsys\._getframe\b",        "sys._getframe() — stack introspection"),
    (r"\binspect\.stack\b",        "inspect.stack() — stack introspection"),
    (r"\binspect\.currentframe\b", "inspect.currentframe()"),
    (r"\bctypes\b",                "ctypes usage"),
    (r"\bktbench\b",               "importing ktbench eval internals"),
    (r"\bkernelbench\b",           "importing kernelbench eval internals"),
    (r"\bsys\.exit\b",             "sys.exit() — evaluator kill attempt"),
    (r"\bos\._exit\b",             "os._exit() — evaluator kill attempt"),
    (r"oracle_tensors",            "direct oracle_tensor path reference"),
    (r"test_suite\.toml",          "direct test_suite.toml reference"),
]

_SOFT_WARN_PATTERNS: list[tuple[str, str]] = [
    (r"\btry\s*:",         "try/except block — possible fallback bypass"),
    (r"\bexcept\s*[:\w]",  "except clause — possible fallback bypass"),
    (r"\bpass\b",          "bare pass — possible inheritance bypass"),
]

_CUDA_REQUIREMENTS: list[tuple[str, str]] = [
    (r"__global__",                   "missing __global__ kernel definition"),
    (r"load_inline|cpp_extension",    "missing load_inline/cpp_extension"),
    (r"<<<",                          "no kernel launch found (<<<) — __global__ must be invoked"),
]


def static_check(src: str, dsl: str) -> StaticCheckResult:
    errors: list[str] = []
    warnings: list[str] = []

    stripped = _strip_comments(src)

    for pattern, label in _HARD_BLOCK_PATTERNS:
        if re.search(pattern, stripped):
            errors.append(f"BLOCKED: {label}")

    for pattern, label in _SOFT_WARN_PATTERNS:
        if re.search(pattern, stripped):
            warnings.append(f"WARNING: {label}")

    for pattern, label in _CUDA_REQUIREMENTS:
        if not re.search(pattern, stripped):
            errors.append(f"MISSING: {label}")

    # Thin load_inline wrappers with companion .cu/.h files don't need <<<
    if re.search(r"load_inline", stripped):
        if re.search(r"Path\(__file__\)\.parent|_DIR\s*=", stripped) and not re.search(r"<<<", stripped):
            errors = [e for e in errors if "__global__" not in e and "kernel launch" not in e]

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


def utilization_gate(sol_stats: dict, floor_pct: float = 2.0) -> tuple[bool, str]:
    """Return (passes, reason). Blocks near-zero GPU utilization."""
    if not sol_stats or sol_stats.get("sol_score", -1) < 0:
        return True, "sol not measured; gate skipped"

    compute_pct = sol_stats.get("compute_util", 0.0) * 100.0
    memory_pct  = sol_stats.get("memory_util",  0.0) * 100.0

    if compute_pct >= floor_pct or memory_pct >= floor_pct:
        return True, f"compute={compute_pct:.1f}% memory={memory_pct:.1f}%"

    return False, (
        f"suspected_noop: compute={compute_pct:.1f}% memory={memory_pct:.1f}% "
        f"both below floor {floor_pct}%"
    )


def check_excessive_speedup(speedup_vs_ref: float, threshold: float = 50.0) -> tuple[bool, str]:
    """Flag (but do not block) if candidate is suspiciously faster than the reference."""
    if speedup_vs_ref <= 0:
        return False, ""
    if speedup_vs_ref > threshold:
        return True, (
            f"excessive speedup {speedup_vs_ref:.1f}× vs reference "
            f"(threshold {threshold}×) — manual audit recommended"
        )
    return False, ""
