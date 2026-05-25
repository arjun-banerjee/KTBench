# KTBench

Benchmark for evaluating whether an LLM can translate a CUDA GPU kernel from **NVIDIA A100 → NVIDIA H100** — correctly and efficiently.

Given a working A100 CUDA kernel, the model must produce a semantically equivalent H100 CUDA kernel that exploits H100-specific hardware (WGMMA, TMA, producer-consumer pipelines). Performance is measured as **Speed-of-Light fraction** (`achieved_throughput / hw_peak`), which is bounded by physics and cannot be gamed by slowing the baseline.

---

## Installation

```bash
git clone https://github.com/arjun-banerjee/KTBench.git
cd KTBench
pip install -e "."             # core eval only
pip install -e ".[llm]"        # + openai SDK for the agent loop
pip install -e ".[inspect]"    # + Inspect AI for multi-model sweeps
```

Requires Python 3.11+, PyTorch 2.3+, CUDA 12.x.

---

## Problem Format

### Directory layout

Each problem lives in `problems/{problem_id}/`:

```
problems/softmax_a100_to_h100/
├── source.py        — A100 CUDA kernel to translate (defines ModelNew interface)
├── reference_tgt.py — handwritten H100 CUDA reference (performance baseline)
├── generator.py     — make_inputs(shapes, dtype, rng, device) → list[Tensor]
└── perf.py          — optional: flops(shapes, dtype) → float
```

All metadata and test configuration lives in **one place**: `configs/problems.toml`. There are no per-problem `meta.toml` or `test_suite.toml` files.

### `configs/problems.toml`

Every problem is one `[[problems]]` entry:

```toml
[[problems]]
id         = "softmax_a100_to_h100"
name       = "Online Softmax (CUDA A100 → CUDA H100)"
dir        = "problems/softmax_a100_to_h100"
difficulty = 2                                       # 1–5
tags       = ["softmax", "reduction", "fp16", "hw-translation"]
provenance = "kernel from <source> (license)"

# Per-dtype tolerance overrides (optional — defaults: fp16/bf16 atol=rtol=1e-2, fp32 1e-4)
tolerances = {fp16 = {atol = 0.01, rtol = 0.01}, bf16 = {atol = 0.01, rtol = 0.01}}

# Structured cases: fixed shapes, freshly randomised values every run.
cases = [
  {id = "small",   desc = "small square",  dtype = "fp16", shapes = {N = 128,  D = 256}},
  {id = "large_D", desc = "wide rows",     dtype = "fp16", shapes = {N = 512,  D = 4096}},
  {id = "nonpow2", desc = "non-power-of-2",dtype = "fp16", shapes = {N = 1000, D = 300}},
]

# Stress: shapes AND values are random — shape-specific hacks fail here.
stress = {num_trials = 30, pass_threshold = 0.90, shape_ranges = {N = [1, 4096], D = [1, 8192]}}
```

**Field reference:**

| Field | Required | Description |
|---|---|---|
| `id` | yes | Must match the problem's directory name |
| `name` | yes | Human-readable title |
| `dir` | yes | Relative path to the problem directory |
| `difficulty` | yes | 1–5 |
| `tags` | yes | List of strings, e.g. `["gemm", "bfloat16"]` |
| `provenance` | yes | Source of the A100 kernel and license |
| `tolerances` | no | Per-dtype `{atol, rtol}` overrides |
| `cases` | yes | At least 3 structured cases with varied shapes |
| `stress` | yes | `num_trials`, `pass_threshold`, `shape_ranges` |

### `source.py`

The A100 CUDA kernel to translate. Must define `ModelNew` with a `forward` method that the grader calls as `ModelNew().forward(*inputs)`:

```python
import torch, torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
__global__ void my_kernel(...) { ... }
"""
_mod = load_inline(name="my_kernel", cuda_sources=[_CUDA_SRC], ...)

class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _mod.my_kernel(x, out)
        return out
```

### `reference_tgt.py`

A hand-optimised H100 CUDA implementation of the same operation. Same `ModelNew` interface. Used as the performance baseline — the grader times both candidate and reference on the same hardware and reports `speedup_vs_ref` as context.

### `generator.py`

```python
import numpy as np, torch

def make_inputs(shapes: dict, dtype: str, rng: np.random.Generator, device) -> list:
    # Use rng for ALL randomness. Values differ every call.
    N, D = shapes["N"], shapes["D"]
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    x = torch.from_numpy(rng.standard_normal((N, D)).astype("float32")).to(dtype=dt, device=device)
    return [x]
```

Returns a flat `list` of `torch.Tensor` objects. The grader calls `forward(*inputs)`.

### `perf.py` (optional)

```python
def flops(shapes: dict, dtype: str) -> float:
    # Return total floating-point operations for one forward pass.
    # Convention: add/sub/mul/div = 1 flop each; FMA = 2; transcendentals = 1.
    N, D = shapes["N"], shapes["D"]
    return float(5 * N * D)  # example: 5 ops per element
```

When present, the grader uses this to compute `compute_util` (compute SOL). Without it, only `memory_util` (memory SOL) is computed.

### Adding a new problem

1. Create the directory: `mkdir problems/{problem_id}/`
2. Write `source.py`, `reference_tgt.py`, `generator.py` (and optionally `perf.py`)
3. Add an entry to `configs/problems.toml` following the format above
4. Verify it loads: `python -c "from ktbench import load_problem; load_problem('problems/{problem_id}')"`

---

## Running Evaluations

### Evaluate one candidate against one problem

```bash
python scripts/run_eval.py \
    --problem problems/softmax_a100_to_h100 \
    --candidate my_kernel.py \
    --device 0 \
    --verbose
```

Prints a JSON result summary. Exit code 0 if the candidate compiles and passes all correctness cases; 1 otherwise. Pass `--json-out result.json` to write to a file.

```
python scripts/run_eval.py --help

  --problem     Path to problem directory
  --candidate   Path to candidate .py file
  --device      CUDA device index (default: 0)
  --seed        RNG seed (default: random)
  --n-timing    Timing trials (default: 20)
  --verbose     Print pipeline stage output
  --json-out    Write result JSON to file
```

### Sweep multiple models across multiple problems

Requires `pip install -e ".[inspect]"` and API keys in the environment.

```bash
OPENAI_API_KEY=sk-... \
ANTHROPIC_API_KEY=sk-ant-... \
python scripts/run_sweep.py \
    --problems problems/softmax_a100_to_h100 problems/bf16_gemv_a100_to_h100 \
    --models openai/o3 anthropic/claude-opus-4-7 \
    --device 0 \
    --out results/sweep.json
```

To run every problem in the benchmark:

```bash
python -c "
from ktbench.dataset import load_all_problems
print(' '.join(str(p.problem_dir) for p in load_all_problems()))
" | xargs -I{} echo {}   # prints all 18 problem paths
```

Then pass them all to `--problems`.

```
python scripts/run_sweep.py --help

  --problems    One or more problem directory paths
  --models      Model strings: openai/o3, anthropic/claude-opus-4-7, google/gemini-2.5-pro, xai/grok-3
  --device      CUDA device index (default: 0)
  --seed        RNG seed
  --n-timing    Timing trials per submission (default: 20)
  --max-turns   Message cap per session (default: 30)
  --log-dir     Inspect AI log directory (default: results/inspect_logs)
  --out         Write leaderboard JSON to this file
```

**API keys per provider:**

| Model prefix | Env var | Example model string |
|---|---|---|
| `openai/` | `OPENAI_API_KEY` | `openai/o3` |
| `anthropic/` | `ANTHROPIC_API_KEY` | `anthropic/claude-opus-4-7` |
| `google/` | `GOOGLE_API_KEY` | `google/gemini-2.5-pro` |
| `xai/` | `XAI_API_KEY` | `xai/grok-3` |
| `openai/` (Azure) | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | `openai/my-deployment` |

**Azure OpenAI** — use the `openai/` prefix with your deployment name and point `OPENAI_BASE_URL` at your Azure endpoint. The sweep script auto-detects Azure (via `azure.com` in the URL) and disables the Responses API, falling back to Chat Completions:

```bash
OPENAI_API_KEY=your-azure-key \
OPENAI_BASE_URL=https://my-resource.openai.azure.com/openai/ \
python scripts/run_sweep.py \
    --problems problems/flash_attn_mma_a100_to_h100 \
    --models openai/my-deployment-name \
    --device 0 \
    --out results/sweep.json
```

You can also put credentials in `.env` at the repo root — the script loads it automatically.

### Use as a library

```python
from ktbench import load_problem, eval_translation, build_prompt
from ktbench.dataset import load_all_problems

# Load one problem
problem = load_problem("problems/softmax_a100_to_h100")

# Load all 18 problems (reads configs/problems.toml)
problems = load_all_problems()
hard = load_all_problems(max_difficulty=4)
gemm = load_all_problems(tags=["gemm"])

# Build the prompt
prompt = build_prompt(problem)

# Evaluate a candidate
with open("my_kernel.py") as f:
    candidate_src = f.read()

result = eval_translation(candidate_src, problem, device=0, verbose=True)
print(result.final_score)        # correctness × stress × SOL
print(result.speedup_vs_ref)     # vs reference_tgt.py
print(result.summary())          # full metric dict
```

---

## Updating Results / Leaderboard

The leaderboard is driven by `results/sweep.json`. To update it, re-run the sweep and commit the new file:

```bash
# Run the sweep (all 18 problems, 4 models)
PROBLEM_PATHS=$(python -c "
from ktbench.dataset import load_all_problems
print(' '.join(str(p.problem_dir) for p in load_all_problems()))
")

OPENAI_API_KEY=sk-... \
ANTHROPIC_API_KEY=sk-ant-... \
GOOGLE_API_KEY=... \
XAI_API_KEY=... \
python scripts/run_sweep.py \
    --problems $PROBLEM_PATHS \
    --models openai/o3 anthropic/claude-opus-4-7 google/gemini-2.5-pro xai/grok-3 \
    --device 0 \
    --out results/sweep.json

# Commit the updated results
git add results/sweep.json
git commit -m "update leaderboard results"
git push
```

`results/sweep.json` is a flat list of result objects:

```json
[
  {
    "model": "anthropic/claude-opus-4-7",
    "problem": "softmax_a100_to_h100",
    "final_score": 0.8312,
    "detail": "correctness=100% stress=100% sol=0.8312"
  },
  ...
]
```

Raw Inspect AI logs are written to `results/inspect_logs/` and are not committed.

---

## Publishing the Website

The website is served from the `gh-pages` branch. The publish script reads `traces/*.jsonl`, rebuilds `runs.json`, and pushes to gh-pages.

### From inspect_ai results (recommended)

If you ran evaluations with `scripts/run_parallel.sh`, convert the `.eval` logs to ensemble traces first:

```bash
# Convert results/inspect_logs/**/*.eval → traces/*.jsonl
python website/convert_eval_to_traces.py

# Publish
GHPAGES_WORKTREE=/scratch/abaner/.ktbench-publish/gh-pages-worktree \
    python website/publish.py
```

Options for the converter:

```bash
python website/convert_eval_to_traces.py --dry-run          # preview without writing
python website/convert_eval_to_traces.py --log-dir path/to/inspect_logs
python website/convert_eval_to_traces.py --out-dir my_traces/
```

### From ensemble runs

If you ran scenarios directly through ensemble:

```bash
ensemble run integrations/ensemble/world.toml --traces traces/
GHPAGES_WORKTREE=/scratch/abaner/.ktbench-publish/gh-pages-worktree \
    python website/publish.py
```

### Publish options

```bash
python website/publish.py               # publish once (uses GHPAGES_WORKTREE env var)
python website/publish.py --watch 60    # re-publish every 60 s
python website/publish.py --dry-run     # preview without pushing
python website/publish.py --traces DIR  # use a different traces directory
```

The script reads every `traces/*.jsonl`, rebuilds `runs.json`, copies each `{slug}/trace.jsonl` into the gh-pages worktree, and pushes. The viewer HTML and static assets already on gh-pages are not modified.

**First-time setup** (if the `gh-pages` branch does not yet exist):

```bash
git checkout --orphan gh-pages
git reset --hard
git commit --allow-empty -m "init gh-pages"
git push -u origin gh-pages
git checkout -
```

Then run `python website/publish.py`.

---

## Evaluation Pipeline

```
candidate_src
    │
    ▼
[1] Static analysis     — blocks try/except fallbacks, process kills,
    antihack.py           stack introspection, grader imports
    │ fail → score 0
    ▼
[2] Compilation         — exec() the source, instantiate ModelNew
    registry/dsl.py
    │ fail → compile_error
    ▼
[3] Structured cases    — fixed shapes, freshly randomised values
    correctness.py        must pass ALL curated cases
    │ fail → score 0
    ▼
[4] Stress testing      — random shapes + random values, 30 trials
    correctness.py        must pass ≥ 90% to unlock perf score
    │
    ▼
[5] Performance timing  — candidate AND reference_tgt timed on same HW
    performance.py        CUDA events, L2-flushed, 20 trials each
    │
    ▼
[6] Utilisation gate    — SOL_compute OR SOL_dram must exceed floor
    antihack.py           blocks no-ops from receiving a perf score

Final score = correctness_rate × stress_pass_rate × SOL
```

---

## Scoring

| Metric | Definition | Range |
|---|---|---|
| `correctness_rate` | Fraction of structured cases passing | {0, 1} (all-or-nothing gate) |
| `stress_pass_rate` | Fraction of 30 stress trials passing | [0, 1] |
| `sol_score` | `achieved_throughput / hw_peak` | [0, 1] physics-bounded |
| **`final_score`** | `correctness × stress × SOL` | [0, 1] |

`speedup_vs_ref` is logged as context but is **not** part of the score — it can be inflated by a slow reference.

---

## Repository Structure

```
KTBench/
├── configs/
│   ├── eval_defaults.toml   — timing trials, stress config, antihack thresholds
│   └── problems.toml        — all 18 problem definitions (metadata + test cases)
│
├── problems/                — one subdirectory per problem; code files only
│   └── {problem_id}/
│       ├── source.py        — A100 kernel to translate
│       ├── reference_tgt.py — H100 reference (performance baseline)
│       ├── generator.py     — input tensor factory
│       └── perf.py          — optional flops formula
│
├── src/ktbench/
│   ├── problem.py           — Problem dataclass + load_problem()
│   ├── dataset.py           — load_all_problems() with filtering
│   ├── prompt.py            — build_prompt()
│   ├── score.py             — compute_final_score()
│   ├── source_bundle.py     — collects .cu/.h companion files for prompt
│   ├── registry/
│   │   ├── hardware.py      — A100/H100 specs, compute_sol()
│   │   └── dsl.py           — CUDA model loading via exec()
│   └── eval/
│       ├── harness.py       — eval_translation() entry point
│       ├── correctness.py   — structured cases + stress suite
│       ├── performance.py   — SOL timing, DRAM BW measurement
│       ├── antihack.py      — static checker, utilisation gate
│       └── isolation.py     — subprocess wrapper + watchdog
│
├── tools/
│   └── tools.py             — 5 agent tools (static_check, compile_kernel,
│                              run_correctness, get_gpu_specs, submit_kernel)
│
├── integrations/
│   ├── __init__.py          — shared fmt_result(), extract_final_score()
│   └── inspect_ai.py        — Inspect AI task adapter (pip install .[inspect])
│
├── scripts/
│   ├── run_eval.py          — evaluate one candidate vs one problem
│   └── run_sweep.py         — multi-model sweep via Inspect AI
│
└── results/
    └── sweep.json           — leaderboard output from run_sweep.py
```

---

## Anti-Reward-Hacking

| Exploit | KTBench defence |
|---|---|
| Hardcoded output values | Values freshly sampled from `rng` each call; never reproducible |
| Shape-specific hacks | 30 stress trials with randomly sampled shapes from `shape_ranges` |
| Speedup ratio gaming | SOL denominator is hardware peak, not a runtime |
| `torch.empty()` stale memory | Output buffer zeroed before every comparison |
| Stack introspection | `inspect.stack`, `sys._getframe`, `ctypes` blocked statically |
| Grader monkey-patching | Candidate runs in isolated subprocess |
| Oracle file snooping | Static checker blocks direct path references to `oracle_tensors` |
