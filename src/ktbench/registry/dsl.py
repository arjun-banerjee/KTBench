"""DSL registry — CUDA only."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class DSLSpec:
    name: str
    uses_tempfile: bool
    entry_point: str = "ModelNew"


_REGISTRY: dict[str, DSLSpec] = {
    "cuda": DSLSpec("cuda", uses_tempfile=False),
}


def get_dsl_spec(dsl: str) -> DSLSpec:
    key = dsl.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown DSL {dsl!r}. Only 'cuda' is supported.")
    return _REGISTRY[key]


class DSLLoadError(RuntimeError):
    pass


def load_model_class(src: str, dsl: str, build_dir: str | None = None) -> type:
    """Compile and import src for the given DSL. Returns the ModelNew class."""
    get_dsl_spec(dsl)  # validates
    return _load_via_exec(src, "ModelNew", build_dir)


def _load_via_exec(src: str, entry_point: str, build_dir: str | None) -> type:
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
