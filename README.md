# KTBench — Kernel Translation Benchmark

KTBench evaluates whether an LLM can translate a GPU kernel from one DSL and hardware target to another — correctly and efficiently. Given a working implementation in `(src_dsl, src_hw)`, the model must produce a semantically equivalent implementation in `(tgt_dsl, tgt_hw)` that saturates the target hardware.

**Official eval fleet:** up to 8× NVIDIA H200 SXM (primary), up to 2× NVIDIA A100 SXM (secondary), AWS Trainium2 (NKI, TBD).

---

## Why KTBench

Existing kernel benchmarks (KernelBench, robust-kbench) evaluate optimization from a PyTorch reference. They have well-documented reward hacking problems: models learn to exploit the evaluator rather than write real kernels. KTBench is designed around four principles to prevent this:

1. **Tensor values are always freshly randomized.** A kernel cannot pass by hardcoding outputs — the values differ every run.
2. **Stress cases randomize both shapes and values.** Shape-specific hacks fail on 30 trials with randomly sampled dimensions.
3. **Performance is SOL fraction, not speedup ratio.** Speed-of-Light = `achieved_throughput / hw_peak`. It is bounded by physics and cannot be gamed by slowing the baseline.
4. **The eval subprocess cannot observe the grader.** Stack introspection, monkey-patching, and process-kill attempts are blocked statically and at the process-group level.

---

## Installation

```bash
git clone https://github.com/arjun-banerjee/KTBench.git
cd KTBench
pip install -e ".[triton]"         # add [nki] for Trainium support
pip install -e ".[triton,llm]"     # also install openai SDK for the LLM agent
```

Requires Python 3.11+, PyTorch 2.3+, CUDA 12.x.

---

## Quick Start

### Evaluate a candidate kernel

```bash
python scripts/run_eval.py \
    --problem problems/softmax_h200_to_triton \
    --candidate my_kernel.py \
    --device 0 \
    --verbose
```

The script runs the full 6-stage pipeline and prints a JSON result summary. Exit code 0 if the candidate compiles and passes all correctness cases; 1 otherwise.

### Generate a translation with an LLM

```bash
# OpenAI (Responses API — supports reasoning models like o3)
OPENAI_API_KEY=sk-... python scripts/run_agent.py \
    --problem problems/softmax_h200_to_triton \
    --model o3 --reasoning-effort medium \
    --out candidate.py --eval --device 0

# Azure OpenAI
AZURE_OPENAI_API_KEY=... python scripts/run_agent.py \
    --problem problems/softmax_h200_to_triton \
    --model gpt-4.1 --provider azure \
    --base-url https://my-resource.cognitiveservices.azure.com/openai/v1/ \
    --out candidate.py --eval

# xAI Grok (Chat Completions)
XAI_API_KEY=... python scripts/run_agent.py \
    --problem problems/softmax_h200_to_triton \
    --model grok-3 --provider grok \
    --out candidate.py --eval
```

`--eval` runs the full 6-stage pipeline after generation and exits 0 on success.
Omit `--eval` to just write the candidate without running the evaluator.

### Add a new problem

```bash
python scripts/add_problem.py \
    --id flash_attn_h200_to_hip \
    --src-dsl cuda --src-hw nvidia_h200_sxm \
    --tgt-dsl hip  --tgt-hw amd_mi300x \
    --name "Flash Attention 2 Forward (H200 CUDA → MI300X HIP)" \
    --tags attention,fp16,tiling \
    --difficulty 4
```

This scaffolds `problems/flash_attn_h200_to_hip/` with template files. Fill in `source.py`, `oracle.py`, `reference_tgt.py`, and `generator.py`, then build oracle tensors:

```bash
python scripts/build_oracle_tensors.py \
    --problem problems/flash_attn_h200_to_hip \
    --device 0
```

### Use as a library

```python
from ktbench import load_problem, eval_translation, build_prompt
from ktbench.llm import make_client, TranslationAgent

problem = load_problem("problems/softmax_h200_to_triton")

# Build the prompt manually (e.g. to call your own model)
prompt = build_prompt(problem)

# Or use the built-in agent (requires openai package + API key)
client = make_client(api_key_env="OPENAI_API_KEY")
agent  = TranslationAgent(client=client, model="gpt-4o", problem=problem)
candidate_src = agent.generate()

# Evaluate a candidate
result = eval_translation(candidate_src, problem, device=0, verbose=True)

print(result.final_score)      # correctness × stress_pass_rate × SOL
print(result.speedup_vs_ref)   # vs handwritten reference_tgt.py
print(result.summary())        # full metric dict
```

---

## Evaluation Pipeline

```
candidate_src
    │
    ▼
[1] Static analysis          — AST + regex; blocks try/except fallbacks,
    (antihack.py)              process kills, stack introspection, grader imports
    │  fail → score 0
    ▼
[2] Compilation              — DSL-specific load (exec or tempfile)
    (registry/dsl.py)
    │  fail → compile_fail
    ▼
[3] Structured correctness   — fixed shapes, FRESHLY RANDOMIZED values
    (correctness.py)           must pass ALL curated cases
    │  fail → score 0
    ▼
[4] Stress testing           — random shapes + random values, 30 trials
    (correctness.py)           must pass ≥90% to unlock performance score
    │
    ▼
[5] Performance timing       — candidate AND reference_tgt.py timed on same HW
    (performance.py)           CUDA events, L2-flushed, 20 trials each
    │
    ▼
[6] Utilization gate         — SOL_compute OR SOL_dram must exceed 2%
    (antihack.py)              blocks suspected no-ops from receiving perf score
    │
    ▼
Final score = correctness_rate × stress_pass_rate × SOL_tgt
```

---

## Scoring

| Component | Definition | Range |
|---|---|---|
| `correctness_rate` | Fraction of structured cases passing | {0, 1} (all-or-nothing gate) |
| `stress_pass_rate` | Fraction of 30 stress trials passing | [0, 1] |
| `sol_score` | `achieved_throughput / hw_peak` on target HW | [0, 1] physics-bounded |
| **`final_score`** | `correctness × stress × SOL` | [0, 1] |

`speedup_vs_ref` (vs. handwritten `reference_tgt.py`) is displayed on the leaderboard as context but is **not** part of the gated score — it can be inflated by providing a slow reference.

All metrics are reported individually:

| Metric | Description |
|---|---|
| `correctness_rate` | Structured case pass rate |
| `stress_pass_rate` | Stress trial pass rate |
| `sol_score` | HW utilization fraction |
| `speedup_vs_ref` | Runtime vs. handwritten reference |
| `occupancy_pct` | GPU occupancy % |
| `memory_ratio` | Peak memory vs. reference |
| `fusion_ratio` | Kernel launches vs. reference |
| `energy_ratio` | Energy vs. reference (NVIDIA only) |
| `final_score` | Primary ranking key |

---

## Problem Format

Each problem lives in `problems/{problem_id}/`:

```
problems/softmax_h200_to_triton/
├── meta.toml           # translation axis, tags, difficulty, tolerance overrides
├── test_suite.toml     # structured case shapes + stress ranges
├── generator.py        # make_inputs(shapes, dtype, rng, device) -> list[Tensor]
├── source.py           # source kernel (src_dsl) — what the model must translate
├── oracle.py           # ground-truth reference (PyTorch eager)
├── reference_tgt.py    # handwritten reference in tgt_dsl — performance baseline
├── notes.md            # human description and translation gotchas
└── oracle_tensors/     # pre-stored oracle outputs (built by build_oracle_tensors.py)
    ├── small.safetensors
    └── manifest.json
```

### `meta.toml`

```toml
problem_id   = "softmax_h200_to_triton"
name         = "Online Softmax (CUDA H200 → Triton H200)"
src_dsl      = "cuda"
src_hw       = "nvidia_h200_sxm"
tgt_dsl      = "triton"
tgt_hw       = "nvidia_h200_sxm"
tags         = ["softmax", "reduction", "fp16"]
difficulty   = 2          # 1–5

[tolerances.fp16]         # override default; optional
atol = 1e-2
rtol = 1e-2
```

### `test_suite.toml`

```toml
# Fixed shapes, but VALUES are always freshly randomized — hardcoding fails.
[[cases]]
id    = "small"
dtype = "fp16"
[cases.shapes]
N = 128
D = 256

[[cases]]
id    = "nonpow2"
dtype = "fp16"
[cases.shapes]
N = 1000
D = 300

# Stress: both shapes AND values are random — shape-specific hacks fail.
[stress]
num_trials     = 30
pass_threshold = 0.90

[stress.shape_ranges]
N = [1, 4096]
D = [1, 8192]
```

### `generator.py`

```python
import numpy as np
import torch

def make_inputs(shapes, dtype, rng, device):
    # rng is np.random.Generator — use it for ALL randomness.
    # Values differ every call; never hardcode them.
    N, D = shapes["N"], shapes["D"]
    x = torch.from_numpy(
        rng.standard_normal((N, D)).astype("float32")
    ).to(dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype],
         device=device)
    return [x]
```

### Candidate interface

Your submission must define `ModelNew` with a `forward` method:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _kernel(x_ptr, y_ptr, N, D, BLOCK_D: tl.constexpr):
    ...

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
        return y
```

Only `ModelNew().forward(*inputs)` is called by the grader. Helper functions and multiple compiled kernels are allowed internally.

---

## Supported DSLs and Hardware

### DSL Registry

| DSL | Vendors | Load method |
|---|---|---|
| `cuda` | NVIDIA | `load_inline` |
| `cute` | NVIDIA (Ampere+) | `load_inline` |
| `triton` | NVIDIA, AMD | tempfile + JIT |
| `tilelang` | NVIDIA | tempfile + JIT |
| `helion` | NVIDIA | tempfile + JIT |
| `hip` | AMD | `load_inline` |
| `nki` | AWS Trainium | tempfile + JIT |
| `pallas` | TPU | tempfile + JIT |
| `numba` | NVIDIA, AMD | tempfile + JIT |
| `mojo` | NVIDIA | tempfile + JIT |
| `pytorch` | all | exec |

### Hardware Registry (official eval fleet)

| Key | Count | FP16 TFLOP/s | DRAM BW | Notes |
|---|---|---|---|---|
| `nvidia_h200_sxm` | up to 8 | 989 | 4800 GB/s | Primary target |
| `nvidia_a100_sxm` | up to 2 | 312 | 2000 GB/s | Secondary |
| `aws_trainium2` | TBD | 832 | 820 GB/s | NKI only |

---

## Anti-Reward-Hacking Design

| Exploit | Mechanism | KTBench fix |
|---|---|---|
| `torch.empty()` stale GPU memory | Allocator reuses memory containing prior reference output | Output buffer zeroed + pointer aliasing check before every comparison |
| Hardcoded output values | Return fixed tensor regardless of input | Values freshly sampled from `rng` each call; different every run |
| Shape-specific hack | Hardcode logic for one shape | 30 stress trials with randomly sampled shapes from `stress.shape_ranges` |
| Speedup ratio gaming | Slow the baseline to inflate ratio | SOL = `achieved / hw_peak`; denominator is the hardware ceiling, not a runtime |
| Grader monkey-patching | Patch timing or comparison functions | Candidate runs in isolated subprocess; grader namespace unreachable |
| Stack introspection | Walk call stack to find pre-computed tensors | `inspect.stack`, `sys._getframe`, `ctypes` blocked by static checker |
| `pkill` the evaluator | Kill the parent eval process | Subprocess started in new process group; SIGKILL kills the group, scored as failure |
| Oracle file snooping | Read `oracle_tensors/` from filesystem | Static checker blocks direct path references; oracle tensors not mounted in candidate container |

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/run_agent.py` | Generate a translation via LLM then optionally evaluate |
| `scripts/run_eval.py` | Evaluate one candidate against one problem |
| `scripts/add_problem.py` | Scaffold a new problem directory |
| `scripts/build_oracle_tensors.py` | Pre-compute oracle outputs for structured cases |

### `run_agent.py`

```
python scripts/run_agent.py --problem PATH --model MODEL [options]

  --problem            Path to problem directory
  --model              Model name, e.g. gpt-4o, o3, gpt-4.1, grok-3

Provider (pick one preset or use --provider custom for full control):
  --provider           openai (default) | azure | grok | custom
  --api-key-env        Env var for API key (default per provider)
  --base-url           API base URL (required for azure)
  --api-kind           responses | chat  (default per provider)
  --reasoning-effort   minimal | low | medium | high  (Responses API only)

Output:
  --out FILE           Write candidate to FILE (default: print to stdout)
  --retries N          API retry budget (default: 3)

Evaluation (optional):
  --eval               Run the full evaluator on the generated candidate
  --device N           CUDA device index (default: 0)
  --seed N             RNG seed for evaluation
  --json-out FILE      Write evaluation JSON to FILE
  --verbose
```

**Provider table:**

| `--provider` | API key env var | API kind | Base URL |
|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | Responses | (OpenAI default) |
| `azure` | `AZURE_OPENAI_API_KEY` | Responses | required via `--base-url` |
| `grok` | `XAI_API_KEY` | Chat | `https://api.x.ai/v1` |
| `custom` | set via `--api-key-env` | set via `--api-kind` | set via `--base-url` |

### `run_eval.py`

```
python scripts/run_eval.py --problem PATH --candidate FILE [options]

  --problem     Path to problem directory
  --candidate   Path to candidate .py file
  --device      CUDA device index (default: 0)
  --seed        Global RNG seed (default: random per run)
  --n-timing    Number of timing trials (default: 20)
  --verbose     Print pipeline stage output
  --json-out    Write result JSON to file instead of stdout
```

### `add_problem.py`

```
python scripts/add_problem.py --id ID --src-dsl DSL --src-hw HW \
                               --tgt-dsl DSL --tgt-hw HW [options]

  --id           Problem identifier (used as directory name)
  --src-dsl      Source DSL (cuda, triton, hip, ...)
  --src-hw       Source hardware key (nvidia_h200_sxm, ...)
  --tgt-dsl      Target DSL
  --tgt-hw       Target hardware key
  --name         Human-readable name
  --tags         Comma-separated tags
  --difficulty   1–5 (default: 2)
  --problems-dir Base problems directory (default: problems/)
```

### `build_oracle_tensors.py`

```
python scripts/build_oracle_tensors.py --problem PATH [options]

  --problem    Path to problem directory
  --device     CUDA device index (default: 0)
  --seed       Base RNG seed for oracle input generation
  --overwrite  Regenerate even if tensors already exist
```

---

## Repository Structure

```
KTBench/
├── problems/                    # problem library
│   └── {problem_id}/
│       ├── meta.toml
│       ├── test_suite.toml
│       ├── generator.py
│       ├── source.py
│       ├── oracle.py
│       ├── reference_tgt.py
│       ├── notes.md
│       └── oracle_tensors/
├── src/ktbench/
│   ├── llm/
│   │   ├── client.py            # make_client() — OpenAI / Azure / Grok
│   │   ├── utils.py             # retry backoff, usage dict, code extraction
│   │   └── agent.py             # TranslationAgent — prompt → LLM → ModelNew src
│   ├── registry/
│   │   ├── hardware.py          # HW specs and SOL computation
│   │   └── dsl.py               # DSL loading per backend
│   ├── eval/
│   │   ├── harness.py           # eval_translation() — top-level entry point
│   │   ├── correctness.py       # randomized correctness suite
│   │   ├── performance.py       # SOL timing, memory, launches, energy
│   │   ├── antihack.py          # static checker, utilization gate
│   │   └── isolation.py         # subprocess wrapper + watchdog
│   ├── problem.py               # problem data models and TOML loading
│   ├── dataset.py               # problem loading and filtering
│   ├── prompt.py                # prompt construction (no shapes exposed)
│   └── score.py                 # final score and leaderboard row
├── scripts/
│   ├── run_agent.py             # LLM → candidate → (optional) eval
│   ├── run_eval.py
│   ├── add_problem.py
│   └── build_oracle_tensors.py
└── configs/
    └── eval_defaults.toml
```
