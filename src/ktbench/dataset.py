"""Problem loading and filtering."""

from __future__ import annotations

from pathlib import Path

from ktbench.problem import Problem, load_problem


def load_all_problems(
    problems_dir: str | Path = "problems",
    *,
    src_dsl: str | None = None,
    tgt_dsl: str | None = None,
    src_hw: str | None = None,
    tgt_hw: str | None = None,
    tags: list[str] | None = None,
    max_difficulty: int | None = None,
) -> list[Problem]:
    """
    Load all problems from problems_dir, with optional filtering.
    Each subdirectory containing a meta.toml is treated as a problem.
    """
    root = Path(problems_dir)
    problems = []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "meta.toml").exists():
            continue
        try:
            p = load_problem(d)
        except Exception as e:
            print(f"[dataset] skipping {d.name}: {e}")
            continue

        m = p.meta
        if src_dsl and m.src_dsl != src_dsl:
            continue
        if tgt_dsl and m.tgt_dsl != tgt_dsl:
            continue
        if src_hw and m.src_hw != src_hw:
            continue
        if tgt_hw and m.tgt_hw != tgt_hw:
            continue
        if max_difficulty and m.difficulty > max_difficulty:
            continue
        if tags and not all(t in m.tags for t in tags):
            continue

        problems.append(p)

    return problems
