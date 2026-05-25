"""
Evaluate a single candidate against one problem.

Usage:
    python scripts/run_eval.py \
        --problem problems/softmax_a100_to_h100 \
        --candidate my_kernel.py \
        --device 0 \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from ktbench import eval_translation, load_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",   required=True, help="Path to problem directory")
    parser.add_argument("--candidate", required=True, help="Path to candidate .py file")
    parser.add_argument("--device",    type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed",      type=int, default=None, help="Global RNG seed")
    parser.add_argument("--n-timing",  type=int, default=20, help="Timing trials")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--json-out",  default=None, help="Write result JSON to this path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    import os

    problem_path = Path(args.problem).resolve()
    os.environ["KTBENCH_PROBLEM_PATH"] = str(problem_path)
    problem = load_problem(problem_path)
    candidate_src = Path(args.candidate).read_text()

    result = eval_translation(
        candidate_src,
        problem,
        device=args.device,
        global_seed=args.seed,
        n_timing_trials=args.n_timing,
        verbose=args.verbose,
    )

    summary = result.summary()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"Result written to {args.json_out}")
    else:
        print(json.dumps(summary, indent=2))

    exit_code = 0 if result.compiled and result.correctness_rate == 1.0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
