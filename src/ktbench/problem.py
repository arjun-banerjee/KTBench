"""
Problem data models and TOML loading.

Directory layout for one problem:
  problems/{problem_id}/
    meta.toml           – translation axis, tags, difficulty, tolerances
    test_suite.toml     – case specs (fixed shapes) + stress ranges
    generator.py        – make_inputs(shapes, dtype, rng, device) -> list[Tensor]
    source.py           – kernel to translate (src_dsl, ModelNew interface)
    oracle.py           – ground-truth reference (PyTorch eager or validated impl)
    reference_tgt.py    – handwritten reference in tgt_dsl (performance baseline)
    oracle_tensors/     – pre-stored oracle outputs for structured cases
      {case_id}.safetensors
      manifest.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# ── Tolerance ─────────────────────────────────────────────────────────────────

DTYPE_DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "fp32": (1e-4, 1e-4),   # (atol, rtol)
    "fp16": (1e-2, 1e-2),
    "bf16": (1e-2, 1e-2),
    "int32": (0.0, 0.0),
    "int64": (0.0, 0.0),
    "bool": (0.0, 0.0),
}


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


# ── Test suite ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CaseSpec:
    """A single structured (curated) correctness case with fixed shape."""
    id: str
    desc: str
    shapes: dict[str, int]         # named dimension → concrete size
    dtype: str                     # "fp16" | "bf16" | "fp32"


@dataclass(frozen=True)
class StressConfig:
    num_trials: int
    pass_threshold: float                     # fraction that must pass
    shape_ranges: dict[str, tuple[int, int]]  # dim_name → (min, max) inclusive


@dataclass(frozen=True)
class TestSuite:
    cases: tuple[CaseSpec, ...]
    stress: StressConfig


def _load_test_suite(path: Path) -> TestSuite:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cases = []
    for c in raw.get("cases", []):
        # Pass shape scalars through as-is (every generator on disk
        # reads `shapes["N"]` as an int). The earlier list-wrap was a
        # placeholder for per-case shape sweeps that never landed and
        # only ever broke the consumers.
        shapes = dict(c.get("shapes", {}))
        cases.append(CaseSpec(
            id=c["id"],
            desc=c.get("desc", ""),
            shapes=shapes,
            dtype=c.get("dtype", "fp32"),
        ))

    stress_raw = raw.get("stress", {})
    ranges_raw = stress_raw.get("shape_ranges", {})
    shape_ranges = {}
    for dim, val in ranges_raw.items():
        if isinstance(val, list) and len(val) == 2:
            shape_ranges[dim] = (int(val[0]), int(val[1]))
        else:
            shape_ranges[dim] = (int(val), int(val))

    stress = StressConfig(
        num_trials=int(stress_raw.get("num_trials", 30)),
        pass_threshold=float(stress_raw.get("pass_threshold", 0.9)),
        shape_ranges=shape_ranges,
    )

    return TestSuite(cases=tuple(cases), stress=stress)


# ── Meta ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProblemMeta:
    problem_id: str
    name: str
    src_dsl: str
    src_hw: str
    tgt_dsl: str
    tgt_hw: str
    tags: tuple[str, ...]
    difficulty: int            # 1–5
    provenance: str
    # Per-problem tolerance overrides; fallback to dtype defaults
    tolerances: dict[str, Tolerance] = field(default_factory=dict)


def _load_meta(path: Path) -> ProblemMeta:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    tols: dict[str, Tolerance] = {}
    for dtype, spec in raw.get("tolerances", {}).items():
        tols[dtype] = Tolerance(atol=float(spec["atol"]), rtol=float(spec["rtol"]))

    return ProblemMeta(
        problem_id=raw["problem_id"],
        name=raw.get("name", raw["problem_id"]),
        src_dsl=raw["src_dsl"],
        src_hw=raw["src_hw"],
        tgt_dsl=raw["tgt_dsl"],
        tgt_hw=raw["tgt_hw"],
        tags=tuple(raw.get("tags", [])),
        difficulty=int(raw.get("difficulty", 1)),
        provenance=raw.get("provenance", ""),
        tolerances=tols,
    )


# ── Generator interface ───────────────────────────────────────────────────────

GeneratorFn = Callable[..., list[Any]]


def _load_generator(problem_dir: Path) -> GeneratorFn:
    """
    Load generator.py from the problem directory.
    Must define: make_inputs(shapes, dtype, rng, device) -> list[torch.Tensor]
    """
    gen_path = problem_dir / "generator.py"
    if not gen_path.exists():
        raise FileNotFoundError(f"generator.py not found in {problem_dir}")

    spec = importlib.util.spec_from_file_location("_ktbench_gen", gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "make_inputs"):
        raise AttributeError(f"generator.py in {problem_dir} must define make_inputs()")
    return module.make_inputs


FlopsFn = Callable[..., float]


def _load_perf(problem_dir: Path) -> Optional[FlopsFn]:
    """Load an optional perf.py from the problem directory.

    perf.py must define ``flops(shapes, dtype) -> float`` returning the
    number of floating-point operations the op performs on inputs of
    the given shape and dtype. The convention is standard SOL counting:
    add / sub / mul / div / cmp = 1 flop each; FMA = 2 flops (counted
    as separate mul + add); transcendentals (exp / log / rsqrt / sqrt /
    sigmoid / tanh) = 1 flop each; a reduction over N elements = N
    flops. Element-wise op over a tensor of shape S = prod(S) flops.

    When perf.py is absent the eval still computes ``memory_util`` from
    input + output byte counts and reports ``compute_util = -1``. Add
    a perf.py whenever ``sol = max(compute_util, memory_util)`` should
    reflect the compute axis (gemm, attention without tile-flash
    tricks, anything matmul-shaped at high arithmetic intensity).
    """
    perf_path = problem_dir / "perf.py"
    if not perf_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("_ktbench_perf", perf_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "flops"):
        return None
    return module.flops


# ── Problem ───────────────────────────────────────────────────────────────────

@dataclass
class Problem:
    meta: ProblemMeta
    test_suite: TestSuite
    make_inputs: GeneratorFn    # make_inputs(shapes, dtype, rng, device) -> list[Tensor]
    problem_dir: Path
    flops_fn: Optional[FlopsFn] = None  # optional perf.py::flops(shapes, dtype) -> float

    # Source text (loaded lazily)
    _source_src: str | None = field(default=None, repr=False)
    _oracle_src: str | None = field(default=None, repr=False)
    _reference_tgt_src: str | None = field(default=None, repr=False)

    @property
    def source_src(self) -> str:
        if self._source_src is None:
            self._source_src = (self.problem_dir / "source.py").read_text()
        return self._source_src

    @property
    def oracle_src(self) -> str:
        if self._oracle_src is None:
            self._oracle_src = (self.problem_dir / "oracle.py").read_text()
        return self._oracle_src

    @property
    def reference_tgt_src(self) -> str:
        if self._reference_tgt_src is None:
            self._reference_tgt_src = (self.problem_dir / "reference_tgt.py").read_text()
        return self._reference_tgt_src

    def get_tolerance(self, dtype: str) -> Tolerance:
        if dtype in self.meta.tolerances:
            return self.meta.tolerances[dtype]
        atol, rtol = DTYPE_DEFAULT_TOLERANCES.get(dtype, (1e-4, 1e-4))
        return Tolerance(atol=atol, rtol=rtol)

    def oracle_tensor_path(self, case_id: str) -> Path:
        return self.problem_dir / "oracle_tensors" / f"{case_id}.safetensors"

    def has_oracle_tensors(self) -> bool:
        manifest = self.problem_dir / "oracle_tensors" / "manifest.json"
        return manifest.exists()


def load_problem(problem_dir: str | Path) -> Problem:
    d = Path(problem_dir)
    meta = _load_meta(d / "meta.toml")
    suite = _load_test_suite(d / "test_suite.toml")
    gen = _load_generator(d)
    flops_fn = _load_perf(d)
    return Problem(
        meta=meta,
        test_suite=suite,
        make_inputs=gen,
        problem_dir=d,
        flops_fn=flops_fn,
    )
