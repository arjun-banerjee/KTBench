"""
run_sweep.py — Benchmark multiple frontier models on KTBench problems.

Runs every (problem × model) combination and writes a leaderboard JSON.
Uses Inspect AI under the hood, which handles OpenAI, Anthropic, Google,
and xAI natively.

API keys (set in environment before running):
    OPENAI_API_KEY       — openai/*, e.g. openai/o3, openai/gpt-4o
    ANTHROPIC_API_KEY    — anthropic/*, e.g. anthropic/claude-opus-4-7
    GOOGLE_API_KEY       — google/*, e.g. google/gemini-2.5-pro
    XAI_API_KEY          — xai/*, e.g. xai/grok-3

Quick start (all four frontier models on one problem):
  python scripts/run_sweep.py \\
    --problems problems/softmax_h200_to_triton \\
    --models openai/o3 anthropic/claude-opus-4-7 google/gemini-2.5-pro xai/grok-3 \\
    --device 0 \\
    --out results/sweep.json

Multiple problems:
  python scripts/run_sweep.py \\
    --problems problems/softmax_h200_to_triton problems/matmul_h200_to_triton \\
    --models openai/gpt-4o anthropic/claude-sonnet-4-6 \\
    --out results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))


def _build_leaderboard(eval_logs) -> list[dict]:
    """Extract a flat list of result dicts from Inspect AI EvalLog objects."""
    rows = []
    for log in eval_logs:
        model_name = str(log.eval.model)
        for sample_log in (log.samples or []):
            problem_id = sample_log.id or "unknown"
            score_val  = 0.0
            explanation = "no result"
            if sample_log.scores:
                # ktbench_scorer is the only scorer; grab its Score
                s = next(iter(sample_log.scores.values()), None)
                if s is not None:
                    score_val   = float(s.value) if s.value is not None else 0.0
                    explanation = s.explanation or ""
            rows.append({
                "model":       model_name,
                "problem":     problem_id,
                "final_score": round(score_val, 4),
                "detail":      explanation,
            })
    return sorted(rows, key=lambda r: (-r["final_score"], r["model"], r["problem"]))


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results.")
        return
    header = f"{'MODEL':<40}  {'PROBLEM':<40}  {'SCORE':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<40}  {r['problem']:<40}  {r['final_score']:>6.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep multiple frontier models over KTBench problems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--problems", nargs="+", required=True,
                        help="Problem directory paths")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model strings, e.g. openai/o3 anthropic/claude-opus-4-7")
    parser.add_argument("--device",    type=int,   default=0)
    parser.add_argument("--seed",      type=int,   default=None)
    parser.add_argument("--n-timing",  type=int,   default=20)
    parser.add_argument("--max-turns", type=int,   default=30,
                        help="Max messages per session (default 30)")
    parser.add_argument("--log-dir",   default="results/inspect_logs",
                        help="Inspect AI log directory (default: results/inspect_logs)")
    parser.add_argument("--out",       default=None,
                        help="Write leaderboard JSON to this file")
    args = parser.parse_args()

    # Lazy import so the module is importable even without inspect_ai installed.
    try:
        from integrations.inspect_ai import run_ktbench_eval
    except ImportError as e:
        sys.exit(
            f"[run_sweep] inspect-ai is not installed.\n"
            f"  pip install -e '.[inspect]'\n"
            f"  {e}"
        )

    print(f"[run_sweep] Problems : {args.problems}")
    print(f"[run_sweep] Models   : {args.models}")

    eval_logs = run_ktbench_eval(
        problem_paths=args.problems,
        models=args.models,
        device=args.device,
        seed=args.seed,
        n_timing=args.n_timing,
        max_messages=args.max_turns,
        log_dir=args.log_dir,
    )

    rows = _build_leaderboard(eval_logs)

    print("\n--- Leaderboard ---")
    _print_table(rows)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\n[run_sweep] Leaderboard JSON: {args.out}")


if __name__ == "__main__":
    main()
