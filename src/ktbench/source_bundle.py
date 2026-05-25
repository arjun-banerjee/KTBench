"""Collect raw CUDA/C++ source files for problem prompts."""

from __future__ import annotations

from pathlib import Path

# Harness / eval artifacts — never shown in the translation prompt.
_PROMPT_SKIP_NAMES = frozenset({
    "generator.py",
    "oracle.py",
    "reference_tgt.py",
    "perf.py",
    "meta.toml",
    "test_suite.toml",
    "notes.md",
})

# Standalone benchmarks or incomplete target stubs (prompt lists them via source.cu).
_PROMPT_SKIP_STEMS = frozenset({
    "reference_tgt",
    "matmul_main",
})

_PROMPT_EXTENSIONS = (".h", ".hpp", ".cuh", ".cu")


def collect_prompt_source_files(problem_dir: Path) -> list[tuple[str, str]]:
    """
    Return ``(filename, text)`` pairs to embed in the model prompt.

    Includes companion ``.h`` / ``.cuh`` / ``.cu`` next to ``source.py``.
    Excludes reference targets and harness Python modules.
    """
    problem_dir = Path(problem_dir)
    if not problem_dir.is_dir():
        return []

    files: list[Path] = []
    for path in sorted(problem_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in _PROMPT_SKIP_NAMES:
            continue
        if path.suffix not in _PROMPT_EXTENSIONS:
            continue
        if path.stem in _PROMPT_SKIP_STEMS:
            continue
        files.append(path)

    # Headers before translation units; source.cu before other .cu files.
    def sort_key(p: Path) -> tuple:
        ext_rank = {".h": 0, ".hpp": 0, ".cuh": 1, ".cu": 2}[p.suffix]
        cu_rank = 0 if p.name == "source.cu" else 1
        return (ext_rank, cu_rank, p.name)

    files.sort(key=sort_key)
    return [(p.name, p.read_text()) for p in files]


def fence_language(filename: str) -> str:
    if filename.endswith(".py"):
        return "python"
    if filename.endswith((".cu", ".cuh", ".h", ".hpp")):
        return "cpp"
    return ""
