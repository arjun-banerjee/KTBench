"""
Scaffold a new problem directory.

Usage:
    python scripts/add_problem.py \
        --id softmax_h200_to_triton \
        --src-dsl cuda --src-hw nvidia_h200_sxm \
        --tgt-dsl triton --tgt-hw nvidia_h200_sxm \
        --name "Online Softmax (CUDA H200 → Triton)" \
        --tags softmax,reduction,fp16 \
        --difficulty 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

META_TEMPLATE = """\
problem_id   = "{problem_id}"
name         = "{name}"
src_dsl      = "{src_dsl}"
src_hw       = "{src_hw}"
tgt_dsl      = "{tgt_dsl}"
tgt_hw       = "{tgt_hw}"
tags         = {tags}
difficulty   = {difficulty}
provenance   = ""

# Optional per-problem tolerance overrides.
# Defaults: fp32 atol/rtol=1e-4, fp16/bf16 atol/rtol=1e-2
# [tolerances.fp16]
# atol = 1e-2
# rtol = 1e-2
"""

TEST_SUITE_TEMPLATE = """\
# Structured correctness cases.
# Shapes are FIXED (curated for diversity); VALUES are always randomly sampled.
# A kernel cannot pass by hardcoding outputs because values differ each run.

[[cases]]
id    = "small"
desc  = "small baseline case"
dtype = "fp16"
[cases.shapes]
# TODO: fill in named dimensions for your kernel
N = 128
D = 64

[[cases]]
id    = "large"
desc  = "large case"
dtype = "fp16"
[cases.shapes]
N = 4096
D = 256

[[cases]]
id    = "nonpow2"
desc  = "non-power-of-2 dimensions"
dtype = "fp16"
[cases.shapes]
N = 1000
D = 100

[[cases]]
id    = "single"
desc  = "degenerate single row"
dtype = "fp16"
[cases.shapes]
N = 1
D = 64

[[cases]]
id    = "bf16"
desc  = "bfloat16 dtype"
dtype = "bf16"
[cases.shapes]
N = 512
D = 128

# Stress: random shapes AND random values — both hardcoded values
# AND shape-specific implementations will fail here.
[stress]
num_trials      = 30
pass_threshold  = 0.90

[stress.shape_ranges]
# TODO: set ranges appropriate for your kernel
N = [64, 8192]
D = [32, 512]
"""

GENERATOR_TEMPLATE = """\
\"\"\"
Input generator for {problem_id}.

make_inputs(shapes, dtype, rng, device) -> list[torch.Tensor]

  shapes: dict of dimension name -> concrete int (sampled from test_suite.toml)
  dtype:  str e.g. "fp16", "bf16", "fp32"
  rng:    np.random.Generator — use this for ALL randomness, never torch.manual_seed
  device: torch.device

Values are freshly sampled on every call — do not hardcode them.
\"\"\"

import numpy as np
import torch

DTYPE_MAP = {{
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}}


def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device: torch.device) -> list:
    N = shapes["N"]
    D = shapes["D"]
    dt = DTYPE_MAP[dtype]

    # TODO: generate the tensors your kernel expects.
    # Use rng (numpy) for random values, then convert to torch.
    x = torch.from_numpy(rng.standard_normal((N, D)).astype("float32")).to(dtype=dt, device=device)
    return [x]
"""

SOURCE_TEMPLATE = """\
\"\"\"
Source kernel to translate.
Implements ModelNew in {src_dsl}.
\"\"\"
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the source kernel in {src_dsl}
        raise NotImplementedError
"""

ORACLE_TEMPLATE = """\
\"\"\"
Ground-truth reference implementation (PyTorch eager).
Used to verify candidate outputs and generate oracle tensors.
\"\"\"
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the correct reference in pure PyTorch
        raise NotImplementedError
"""

REFERENCE_TGT_TEMPLATE = """\
\"\"\"
Handwritten reference implementation in {tgt_dsl} on {tgt_hw}.
This is the performance baseline — candidates are scored relative to this.
\"\"\"
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement in {tgt_dsl}, tuned for {tgt_hw}
        raise NotImplementedError
"""

NOTES_TEMPLATE = """\
# {name}

## Description
TODO: describe what this kernel does.

## Translation Challenges
TODO: describe the key challenges for this translation axis.

## Known Gotchas
- TODO

## Provenance
TODO: where did the source kernel come from?
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",         required=True)
    parser.add_argument("--src-dsl",    required=True)
    parser.add_argument("--src-hw",     required=True)
    parser.add_argument("--tgt-dsl",    required=True)
    parser.add_argument("--tgt-hw",     required=True)
    parser.add_argument("--name",       default="")
    parser.add_argument("--tags",       default="")
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--problems-dir", default="problems")
    args = parser.parse_args()

    problem_id = args.id
    name = args.name or problem_id.replace("_", " ").title()
    tags_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    tags_toml = "[" + ", ".join(f'"{t}"' for t in tags_list) + "]"

    out_dir = Path(args.problems_dir) / problem_id
    if out_dir.exists():
        print(f"ERROR: {out_dir} already exists", file=sys.stderr)
        sys.exit(1)

    (out_dir / "oracle_tensors").mkdir(parents=True)

    ctx = dict(
        problem_id=problem_id,
        name=name,
        src_dsl=args.src_dsl,
        src_hw=args.src_hw,
        tgt_dsl=args.tgt_dsl,
        tgt_hw=args.tgt_hw,
        tags=tags_toml,
        difficulty=args.difficulty,
    )

    (out_dir / "meta.toml").write_text(META_TEMPLATE.format(**ctx))
    (out_dir / "test_suite.toml").write_text(TEST_SUITE_TEMPLATE)
    (out_dir / "generator.py").write_text(GENERATOR_TEMPLATE.format(**ctx))
    (out_dir / "source.py").write_text(SOURCE_TEMPLATE.format(**ctx))
    (out_dir / "oracle.py").write_text(ORACLE_TEMPLATE.format(**ctx))
    (out_dir / "reference_tgt.py").write_text(REFERENCE_TGT_TEMPLATE.format(**ctx))
    (out_dir / "notes.md").write_text(NOTES_TEMPLATE.format(**ctx))

    print(f"Scaffolded {out_dir}/")
    print("  Next steps:")
    print("  1. Fill in source.py, oracle.py, reference_tgt.py")
    print("  2. Update generator.py to match your kernel's inputs")
    print("  3. Update test_suite.toml shape dimensions")
    print("  4. Run: python scripts/build_oracle_tensors.py --problem", out_dir)


if __name__ == "__main__":
    main()
