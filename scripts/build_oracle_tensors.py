"""
Pre-compute and store oracle tensors for structured correctness cases.

Runs oracle.py on each case in test_suite.toml (with fresh random inputs)
and saves outputs as safetensors files with a manifest.

Usage:
    python scripts/build_oracle_tensors.py \
        --problem problems/softmax_h200_to_triton \
        --device 0 \
        --seed 12345
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from safetensors.torch import save_file

from ktbench.problem import load_problem
from ktbench.registry.dsl import load_model_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--seed",    type=int, default=0xABC123,
                        help="Seed for input generation. Each case gets a derived seed.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    problem = load_problem(args.problem)
    out_dir = problem.problem_dir / "oracle_tensors"
    out_dir.mkdir(exist_ok=True)

    oracle_cls = load_model_class(problem.oracle_src, "pytorch")
    oracle_model = oracle_cls()
    if hasattr(oracle_model, "to"):
        oracle_model = oracle_model.to(device)

    manifest = []
    for idx, case in enumerate(problem.test_suite.cases):
        out_path = out_dir / f"{case.id}.safetensors"
        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {case.id} (already exists; use --overwrite)")
            manifest.append({"case_id": case.id, "file": out_path.name, "skipped": True})
            continue

        case_seed = args.seed ^ (idx * 0x9E3779B9)
        rng = np.random.default_rng(case_seed)

        inputs = problem.make_inputs(case.shapes, case.dtype, rng, device)

        with torch.no_grad():
            oracle_out = oracle_model.forward(*inputs)
        torch.cuda.synchronize(device)

        # Normalise to list of tensors
        if isinstance(oracle_out, torch.Tensor):
            tensors = {"output_0": oracle_out.cpu()}
        elif isinstance(oracle_out, (list, tuple)):
            tensors = {f"output_{i}": t.cpu() for i, t in enumerate(oracle_out)
                       if isinstance(t, torch.Tensor)}
        else:
            print(f"  [warn] {case.id}: oracle returned non-tensor {type(oracle_out)}, skipping")
            continue

        save_file(tensors, str(out_path))
        print(f"  [saved] {case.id} → {out_path.name}  "
              f"shapes={[tuple(t.shape) for t in tensors.values()]}")

        manifest.append({
            "case_id": case.id,
            "file": out_path.name,
            "seed": case_seed,
            "shapes": case.shapes,
            "dtype": case.dtype,
            "outputs": {k: list(v.shape) for k, v in tensors.items()},
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": manifest}, indent=2))
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    main()
