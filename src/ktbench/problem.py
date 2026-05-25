"""
Problem data models and central config loading.

All problems are CUDA A100 → CUDA H100 translations.
Configuration lives in  configs/problems.toml  (one entry per problem).
Each problem directory contains only code files:
  source.py          — kernel to translate (ModelNew interface)
  reference_tgt.py   — handwritten H100 reference (performance baseline)
  generator.py       — make_inputs(shapes, dtype, rng, device) -> list[Tensor]
  perf.py            — optional: flops(shapes, dtype) -> float
  *.cu / *.h / *.cuh — optional companion CUDA source files
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

if __import__("sys").version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Fixed for all problems
SRC_DSL = "cuda"
SRC_HW  = "nvidia_a100_sxm"
TGT_DSL = "cuda"
TGT_HW  = "nvidia_h100_sxm"

_REPO_ROOT      = Path(__file__).parent.parent.parent
_CENTRAL_CONFIG = _REPO_ROOT / "configs" / "problems.toml"

DTYPE_DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "fp32":  (1e-4, 1e-4),
    "fp16":  (1e-2, 1e-2),
    "bf16":  (1e-2, 1e-2),
    "int32": (0.0, 0.0),
    "int64": (0.0, 0.0),
    "bool":  (0.0, 0.0),
}


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass(frozen=True)
class CaseSpec:
    id: str
    desc: str
    shapes: dict[str, int]
    dtype: str


@dataclass(frozen=True)
class StressConfig:
    num_trials: int
    pass_threshold: float
    shape_ranges: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class TestSuite:
    cases: tuple[CaseSpec, ...]
    stress: StressConfig


@dataclass(frozen=True)
class ProblemMeta:
    problem_id: str
    name: str
    src_dsl: str = SRC_DSL
    src_hw:  str = SRC_HW
    tgt_dsl: str = TGT_DSL
    tgt_hw:  str = TGT_HW
    tags: tuple[str, ...] = field(default_factory=tuple)
    difficulty: int = 1
    provenance: str = ""
    tolerances: dict[str, Tolerance] = field(default_factory=dict)


GeneratorFn = Callable[..., list[Any]]
FlopsFn     = Callable[..., float]


@dataclass
class Problem:
    meta: ProblemMeta
    test_suite: TestSuite
    make_inputs: GeneratorFn
    problem_dir: Path
    flops_fn: Optional[FlopsFn] = None

    _source_src: str | None = field(default=None, repr=False)
    _reference_tgt_src: str | None = field(default=None, repr=False)

    @property
    def source_src(self) -> str:
        if self._source_src is None:
            self._source_src = (self.problem_dir / "source.py").read_text()
        return self._source_src

    @property
    def reference_tgt_src(self) -> str:
        if self._reference_tgt_src is None:
            self._reference_tgt_src = (self.problem_dir / "reference_tgt.py").read_text()
        return self._reference_tgt_src

    @property
    def effective_ref_dsl(self) -> str:
        return TGT_DSL

    def get_tolerance(self, dtype: str) -> Tolerance:
        if dtype in self.meta.tolerances:
            return self.meta.tolerances[dtype]
        atol, rtol = DTYPE_DEFAULT_TOLERANCES.get(dtype, (1e-4, 1e-4))
        return Tolerance(atol=atol, rtol=rtol)

    def has_oracle_tensors(self) -> bool:
        return (self.problem_dir / "oracle_tensors" / "manifest.json").exists()


# ── Central config ────────────────────────────────────────────────────────────

_config_cache: list[dict] | None = None


def _get_all_problem_entries() -> list[dict]:
    global _config_cache
    if _config_cache is None:
        with open(_CENTRAL_CONFIG, "rb") as f:
            _config_cache = tomllib.load(f).get("problems", [])
    return _config_cache


def _entry_for_id(problem_id: str) -> dict:
    for entry in _get_all_problem_entries():
        if entry["id"] == problem_id:
            return entry
    raise KeyError(f"Problem {problem_id!r} not found in {_CENTRAL_CONFIG}")


def _parse_meta(entry: dict) -> ProblemMeta:
    tols: dict[str, Tolerance] = {}
    for dtype, spec in entry.get("tolerances", {}).items():
        tols[dtype] = Tolerance(atol=float(spec["atol"]), rtol=float(spec["rtol"]))
    return ProblemMeta(
        problem_id=entry["id"],
        name=entry.get("name", entry["id"]),
        tags=tuple(entry.get("tags", [])),
        difficulty=int(entry.get("difficulty", 1)),
        provenance=entry.get("provenance", ""),
        tolerances=tols,
    )


def _parse_test_suite(entry: dict) -> TestSuite:
    cases = []
    for c in entry.get("cases", []):
        cases.append(CaseSpec(
            id=c["id"],
            desc=c.get("desc", ""),
            shapes=dict(c.get("shapes", {})),
            dtype=c.get("dtype", "fp32"),
        ))
    s = entry.get("stress", {})
    ranges_raw = s.get("shape_ranges", {})
    shape_ranges: dict[str, tuple[int, int]] = {}
    for dim, val in ranges_raw.items():
        if isinstance(val, list) and len(val) == 2:
            shape_ranges[dim] = (int(val[0]), int(val[1]))
        else:
            shape_ranges[dim] = (int(val), int(val))
    stress = StressConfig(
        num_trials=int(s.get("num_trials", 30)),
        pass_threshold=float(s.get("pass_threshold", 0.9)),
        shape_ranges=shape_ranges,
    )
    return TestSuite(cases=tuple(cases), stress=stress)


def _load_generator(problem_dir: Path) -> GeneratorFn:
    gen_path = problem_dir / "generator.py"
    if not gen_path.exists():
        raise FileNotFoundError(f"generator.py not found in {problem_dir}")
    spec = importlib.util.spec_from_file_location("_ktbench_gen", gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_inputs"):
        raise AttributeError(f"generator.py in {problem_dir} must define make_inputs()")
    return module.make_inputs


def _load_perf(problem_dir: Path) -> Optional[FlopsFn]:
    perf_path = problem_dir / "perf.py"
    if not perf_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_ktbench_perf", perf_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "flops", None)


def load_problem(problem_dir: str | Path) -> Problem:
    """Load a problem by its directory path. Metadata is read from configs/problems.toml."""
    d = Path(problem_dir).resolve()
    problem_id = d.name
    entry = _entry_for_id(problem_id)
    meta  = _parse_meta(entry)
    suite = _parse_test_suite(entry)
    gen   = _load_generator(d)
    flops = _load_perf(d)
    return Problem(meta=meta, test_suite=suite, make_inputs=gen, problem_dir=d, flops_fn=flops)
