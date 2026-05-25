"""Problem loading and filtering."""

from __future__ import annotations

from pathlib import Path

from ktbench.problem import Problem, _REPO_ROOT, _get_all_problem_entries, load_problem


def load_all_problems(
    *,
    tags: list[str] | None = None,
    max_difficulty: int | None = None,
) -> list[Problem]:
    """
    Load all problems from configs/problems.toml with optional filtering.
    All problems are CUDA A100 → CUDA H100.
    """
    problems = []
    for entry in _get_all_problem_entries():
        problem_dir = _REPO_ROOT / entry["dir"]
        if not problem_dir.exists():
            print(f"[dataset] skipping {entry['id']}: directory not found at {problem_dir}")
            continue
        try:
            p = load_problem(problem_dir)
        except Exception as e:
            print(f"[dataset] skipping {entry['id']}: {e}")
            continue

        m = p.meta
        if max_difficulty is not None and m.difficulty > max_difficulty:
            continue
        if tags and not all(t in m.tags for t in tags):
            continue

        problems.append(p)

    return problems
