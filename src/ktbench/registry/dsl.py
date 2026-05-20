"""
DSL registry: per-backend loading, compilation, and validation helpers.

Each DSL knows:
  - whether it needs a tempfile (JIT decorators don't work inside exec())
  - which vendors it supports
  - how to load a ModelNew class from source text
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class DSLSpec:
    name: str
    uses_tempfile: bool       # triton/tilelang/nki/pallas need file-based import
    vendors: tuple[str, ...]  # "nvidia" | "amd" | "aws" | "google"
    entry_point: str = "ModelNew"


_REGISTRY: dict[str, DSLSpec] = {
    "cuda":           DSLSpec("cuda",           uses_tempfile=False, vendors=("nvidia",)),
    "cute":           DSLSpec("cute",           uses_tempfile=False, vendors=("nvidia",)),
    "triton":         DSLSpec("triton",         uses_tempfile=True,  vendors=("nvidia", "amd")),
    "tilelang":       DSLSpec("tilelang",       uses_tempfile=True,  vendors=("nvidia",)),
    "helion":         DSLSpec("helion",         uses_tempfile=True,  vendors=("nvidia",)),
    "hip":            DSLSpec("hip",            uses_tempfile=False, vendors=("amd",)),
    "nki":            DSLSpec("nki",            uses_tempfile=True,  vendors=("aws",)),
    "pallas":         DSLSpec("pallas",         uses_tempfile=True,  vendors=("google",)),
    "numba":          DSLSpec("numba",          uses_tempfile=True,  vendors=("nvidia", "amd")),
    "mojo":           DSLSpec("mojo",           uses_tempfile=True,  vendors=("nvidia",)),
    "pytorch":        DSLSpec("pytorch",        uses_tempfile=False, vendors=("nvidia", "amd", "aws", "google")),
}


def get_dsl_spec(dsl: str) -> DSLSpec:
    key = dsl.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown DSL {dsl!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def list_dsls() -> list[str]:
    return sorted(_REGISTRY)


# ── Loading ───────────────────────────────────────────────────────────────────

class DSLLoadError(RuntimeError):
    pass


def load_model_class(src: str, dsl: str, build_dir: str | None = None) -> type:
    """
    Compile and import `src` (candidate code string) for the given DSL.
    Returns the ModelNew class.
    Raises DSLLoadError on syntax error, missing entry point, or compile failure.
    """
    spec = get_dsl_spec(dsl)

    if spec.uses_tempfile:
        return _load_via_tempfile(src, spec.entry_point)
    else:
        return _load_via_exec(src, spec.entry_point, build_dir)


def _load_via_tempfile(src: str, entry_point: str) -> type:
    """Write src to a temp .py file and import it. Required for JIT-decorated DSLs.

    The tempfile is kept on disk for the lifetime of the process: Triton (and
    other JIT-compiled DSLs) call ``inspect.getsource()`` on demand at first
    kernel launch — well after this function returns. Unlinking immediately
    causes "could not get source code" at every kernel call. We schedule
    cleanup via ``atexit`` instead so traces still get cleaned up between
    test runs.
    """
    import atexit

    fd, path = tempfile.mkstemp(suffix=".py", prefix="ktbench_candidate_")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    atexit.register(lambda p=path: os.path.exists(p) and os.unlink(p))

    module_name = f"_ktbench_tmp_{os.path.basename(path)[:-3]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise DSLLoadError(f"Execution failed: {e}") from e
    cls = getattr(module, entry_point, None)
    if cls is None:
        raise DSLLoadError(f"{entry_point} not found in submitted code")
    return cls


def _load_via_exec(src: str, entry_point: str, build_dir: str | None) -> type:
    """exec() the source in an isolated namespace. Used for CUDA/HIP inline compilation."""
    try:
        compile(src, "<candidate>", "exec")
    except SyntaxError as e:
        raise DSLLoadError(f"SyntaxError: {e}") from e

    ns: dict = {}
    if build_dir:
        src = f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = {build_dir!r}\n" + src

    try:
        exec(src, ns)  # noqa: S102
    except Exception as e:
        raise DSLLoadError(f"Runtime error during load: {e}") from e

    cls = ns.get(entry_point)
    if cls is None:
        raise DSLLoadError(f"{entry_point} not defined in submitted code")
    return cls
