# KTBench — Kernel Translation Benchmark

## Vision

KTBench evaluates an LLM's ability to translate an arbitrary computational kernel from one DSL and hardware target to another, correctly and efficiently. The task is: given a working implementation in (src_dsl, src_hw), produce a semantically equivalent implementation in (tgt_dsl, tgt_hw) that saturates the target hardware's capabilities.

This is meaningfully different from KernelBench (optimize a PyTorch op). The model receives a real kernel with real optimization structure; it must understand the hardware-specific idioms, translate them, and re-optimize for the target. The benchmark is grounded in a physically-bounded performance metric (Speed-of-Light fraction), not a gameable speedup ratio.

---

## Core Design Principles

1. **Translation axis is the unit of difficulty** — (CUDA H100) → (HIP MI300) is one axis; (Triton A100) → (NKI Trainium) is another. Problems are tagged by their axis.
2. **Correctness is a diverse test suite, not a single forward pass** — each problem ships a structured set of kernel invocations covering shape variety, boundary conditions, and dtype coverage. The model must pass all of them.
3. **Shapes are never in the prompt** — the I/O contract specifies shape ranges, not fixed sizes. This eliminates the hardcoding exploit from KernelBench.
4. **Performance is SOL fraction, not speedup ratio** — SOL is bounded by physics, not by how slow you make the baseline.
5. **Hardware utilization is a hard gate** — a kernel with near-zero compute or DRAM utilization cannot receive a performance score regardless of its reported runtime.
6. **Eval runs in a subprocess it cannot observe** — eliminates grader monkey-patching, stack introspection, and pkill exploits.

---

## Repo Structure

```
KTBench/
├── plan.md                        # this file
├── questions.md                   # open design questions
├── pyproject.toml
├── README.md
│
├── problems/                      # problem library
│   └── {problem_id}/
│       ├── meta.toml              # translation axis, tags, difficulty, provenance
│       ├── source.py              # kernel to translate (ModelNew-style, src_dsl)
│       ├── oracle.py              # ground truth (PyTorch or canonical reference impl)
│       ├── reference_tgt.py       # handwritten reference in tgt_dsl — performance baseline
│       ├── test_suite.toml        # structured correctness test cases (see below)
│       └── notes.md               # human description, gotchas, expected difficulty
│
├── src/ktbench/
│   ├── registry/
│   │   ├── dsl.py                 # per-DSL: loader, compiler, executor, subprocess flag
│   │   └── hardware.py            # per-HW: vendor, peak FLOP/s, peak BW, valid DSLs
│   ├── eval/
│   │   ├── harness.py             # eval_translation() — top-level entry point
│   │   ├── correctness.py         # diverse test suite runner with memory zeroing
│   │   ├── performance.py         # SOL-grounded timing on target HW
│   │   ├── antihack.py            # static analysis + utilization gate + stress runner
│   │   └── isolation.py           # subprocess wrapper + watchdog (survives pkill)
│   ├── dataset.py                 # problem loading, filtering by dsl/hw/tag
│   ├── prompt.py                  # prompt builder: source.py + signature, no shapes
│   └── score.py                   # correctness × stress_robustness × SOL_tgt/SOL_src
│
├── scripts/
│   ├── run_eval.py                # evaluate a single problem
│   ├── run_sweep.py               # full benchmark sweep across problems
│   ├── add_problem.py             # scaffold a new problem directory
│   └── build_oracle_tensors.py    # pre-run oracle on reference HW, store tensors
│
└── configs/
    └── eval_defaults.toml
```

---

## Problem Format

### `meta.toml`

```toml
problem_id   = "flash_attn2_fwd_h100_to_mi300"
name         = "Flash Attention 2 Forward (H100 CUDA → MI300 HIP)"
src_dsl      = "cuda"
src_hw       = "nvidia_h100"
tgt_dsl      = "hip"
tgt_hw       = "amd_mi300"
tags         = ["attention", "fp16", "tiling", "shared-memory"]
difficulty   = 4                   # 1–5
provenance   = "vllm/flash_attn2"  # where the source kernel came from
```

### `source.py`

The kernel the model must translate. Follows the `ModelNew` interface (same as KernelBench) so the eval harness can load it generically. Includes the full device code and host wrapper, but **not** any input shape constants — those are only in `test_suite.toml`.

### `oracle.py`

A numerically correct reference implementation (usually PyTorch eager or a well-validated CUDA kernel). The oracle is executed at dataset-build time (`build_oracle_tensors.py`) on source HW, and outputs are stored as safetensors files (one per test case) alongside the problem. During live eval, the candidate's outputs are compared against these stored tensors — the oracle does not need to be present on the eval machine. Tensors are versioned by: source kernel commit, framework version, dtype policy, seed, and source HW ID.

### `reference_tgt.py`

A handwritten (or best-known) implementation in the **target DSL** on target HW. This is the performance baseline — the candidate is scored relative to how well this implementation uses the target hardware. May be auto-generated (e.g., Triton from torch.compile for CUDA→Triton problems) and then hand-tuned. Its timing is measured live at eval time on target HW.

### `test_suite.toml`

This is the key addition beyond KernelBench. Each problem ships a **structured set of kernel invocations** that the candidate must pass. Cases cover:

```toml
[meta]
num_correctness_cases = 10    # must pass all 10 to get correctness credit
num_stress_cases      = 30    # sampled from ranges; must pass 90% for perf credit
dtype_variants        = ["fp16", "bf16"]

# --- Structured correctness cases (curated, fixed) ---
[[cases]]
id    = "small_sq"
desc  = "small square head dim"
shape = {batch=2, heads=8, seq=128, head_dim=64}
dtype = "fp16"

[[cases]]
id    = "large_seq"
desc  = "long sequence stress"
shape = {batch=1, heads=32, seq=8192, head_dim=128}
dtype = "fp16"

[[cases]]
id    = "nonpow2_seq"
desc  = "non-power-of-2 sequence length"
shape = {batch=4, heads=16, seq=1000, head_dim=64}
dtype = "bf16"

[[cases]]
id    = "single_head"
desc  = "degenerate single head"
shape = {batch=8, heads=1, seq=512, head_dim=128}
dtype = "fp16"

# ... up to num_correctness_cases

# --- Stress ranges (sampled per-run, not in prompt) ---
[stress_ranges]
batch     = [1, 32]
heads     = [1, 64]
seq       = [64, 16384]
head_dim  = [32, 256]
```

The structured cases are **curated by the benchmark authors** to cover:
- Normal operating point (matches what a real model uses)
- Boundary conditions (seq=1, head_dim=32, batch=1)
- Non-power-of-2 dimensions (breaks naive tiling assumptions)
- Large inputs that stress register/shared memory pressure
- Mixed dtype variants

The stress cases are random draws from the ranges, sampled fresh each eval run so they cannot be memorized.

---

## Eval Pipeline

```
candidate_src (tgt_dsl)
        │
        ▼
[1] Static analysis (antihack.py)
        │  checks: no-ops, output aliasing, grader imports, sys/os abuse
        │  ✗ → flagged: static_exploit  (scored 0, logged)
        ▼
[2] Compilation check (isolation.py subprocess)
        │  ✗ → compile_fail
        ▼
[3] Correctness: structured test suite (correctness.py)
        │  - load stored oracle tensors from test_suite.toml
        │  - zero GPU output buffers before each candidate run
        │  - check pointer aliasing between output and oracle buffers
        │  - allclose with dtype-appropriate tolerance
        │  - must pass ALL num_correctness_cases
        │  ✗ → correctness_fail  (no perf score)
        ▼
[4] Hardware utilization gate (antihack.py)
        │  SOL_compute > 2% AND SOL_dram > 2% (at least one axis must be active)
        │  ✗ → flagged: suspected_noop  (correctness credited, perf blocked)
        ▼
[5] Stress testing (antihack.py)
        │  - 30 random draws from stress_ranges
        │  - shapes not seen during correctness phase
        │  - must pass ≥90% to unlock perf score
        │  ✗ → stress_fail  (correctness credited, perf blocked)
        ▼
[6] Performance timing (performance.py)
        │  - Time candidate AND handwritten reference_tgt.py on same target HW
        │  - CUDA/HIP event timing, N=20 warm trials each
        │  - SOL fraction = achieved_FLOP_or_BW / hw_peak (physics ceiling)
        │  - also records: occupancy, kernel launches, memory, energy (NVML)
        ▼
[7] Final score
        correctness_rate × stress_pass_rate × SOL_tgt
        (speedup_vs_ref displayed separately; not part of the gated score)
```

---

## Anti-Reward-Hacking: All Seven Fixes

| KernelBench exploit | Mechanism | KTBench fix |
|---|---|---|
| `torch.empty()` stale GPU memory | CUDA allocator reuses memory with prior ref output | Explicit `cudaMemset` zero of output buffer before each candidate run |
| Hardcoded input shapes | Fixed shapes disclosed in prompt → zero-work kernel | Shapes never in prompt; test_suite.toml not on candidate Python path |
| Output pointer aliasing (no-op) | Candidate output aliases reference tensor → passes allclose | Detached, freshly allocated output buffer; pointer equality check before comparison |
| Speedup ratio gaming | Slow the baseline, inflate ratio | Score is SOL fraction of HW peak — cannot manipulate the denominator |
| Grader monkey-patching / operator overload | Candidate patches timing or comparison functions | Eval runs in subprocess; candidate code has no access to eval module namespace |
| Stack introspection for pre-computed tensors | Walk call stack to find reference output tensor | Oracle tensors stored as files, not live Python objects; subprocess boundary prevents access |
| `pkill` the evaluator | Kill the Python eval process | Subprocess is owned by watchdog; SIGKILL is caught, trial scored as error |

---

## Metrics

Full metric set (ported from KernelBench extended metrics, mapped to the translation context). The `ref_` fields always refer to the **handwritten reference in tgt_dsl** timed on the same target HW as the candidate — there is no source-kernel comparison in the score.

| Metric group | Fields | Notes |
|---|---|---|
| Correctness | `compiled`, `correctness`, `correctness_trials` | Fraction of structured cases passing (all-or-nothing gate for perf) |
| Stress robustness | `stress_pass_rate` | Fraction of randomized stress trials passing (≥90% gate for perf) |
| Candidate timing | `runtime`, `runtime_stats` | Mean ± std over N=20 warm trials on target HW |
| Reference timing | `ref_runtime`, `ref_runtime_stats` | Handwritten `reference_tgt.py` timed on same target HW, same trial count |
| Speedup vs. reference | `speedup_vs_ref` | `ref_runtime / runtime` — >1 means candidate beats the handwritten reference |
| Numerical precision | `max_abs_error`, `mean_abs_error`, `max_rel_error`, `mean_rel_error` | Aggregated across all correctness trials |
| Memory efficiency | `peak_memory_bytes`, `ref_peak_memory_bytes`, `memory_ratio` | Candidate vs. handwritten reference; lower ratio = more memory efficient |
| Kernel launch / fusion | `num_kernels`, `ref_num_kernels`, `fusion_ratio`, `kernel_breakdown` | Fewer launches with same output = better fusion |
| SOL score | `sol_score`, `arithmetic_intensity`, `achieved_bandwidth_gbps`, `achieved_gflops`, `bottleneck` | `achieved / hw_peak`; identifies compute- vs. memory-bound |
| Energy efficiency | `energy_mj`, `ref_energy_mj`, `energy_ratio`, `avg_power_w` | NVML on NVIDIA, skipped otherwise |
| Roofline / occupancy | `roofline_efficiency`, `occupancy_pct`, `memory_throughput_pct`, `compute_throughput_pct` | From Nsight profiling with heuristic fallback |
| Anti-hack flags | `excessive_speedup`, `suspected_noop`, `static_exploit` | Any flag blocks the perf score; logged for audit |

---

## Scoring

Performance is measured entirely on the **target hardware**, relative to the handwritten `reference_tgt.py` in the same target DSL. There is no comparison to the source kernel or source hardware.

### Gates (must pass both to receive a performance score)
1. **Correctness gate** — candidate must pass all structured test cases (correctness_rate = 1.0)
2. **Stress gate** — candidate must pass ≥90% of randomized stress trials (stress_pass_rate ≥ 0.9)
3. **Utilization gate** — SOL_compute > 2% OR SOL_dram > 2%; blocks suspected no-ops

### Final score (for ranking)

```
final_score = correctness_rate × stress_pass_rate × SOL_tgt
```

- `correctness_rate` ∈ {0, 1} — all-or-nothing on structured cases
- `stress_pass_rate` ∈ [0, 1] — continuous; acts as a robustness weight
- `SOL_tgt` ∈ [0, 1] — fraction of target HW's theoretical throughput ceiling achieved; bounded by physics, not gameable

A no-op kernel that passes structured tests scores `1 × 0 × 0 = 0`. A correct, robust, hardware-saturating kernel scores close to 1.

### Leaderboard columns (all displayed separately)
| Column | Description |
|---|---|
| `correctness_rate` | Fraction of structured cases passing |
| `stress_pass_rate` | Fraction of stress trials passing |
| `sol_score` | SOL fraction on target HW (the primary perf signal) |
| `speedup_vs_ref` | `ref_runtime / runtime` vs. handwritten reference — context, not gated |
| `occupancy_pct` | GPU occupancy % |
| `memory_ratio` | Peak memory vs. handwritten reference |
| `fusion_ratio` | Kernel launch count vs. handwritten reference |
| `energy_ratio` | Energy vs. handwritten reference (NVIDIA only) |
| **`final_score`** | `correctness × stress × SOL` — primary ranking key |


---

## DSL Registry (initial set)

| DSL | HW targets | Load method | Notes |
|---|---|---|---|
| `cuda` | NVIDIA (all) | `load_inline` | C++/CUDA |
| `cute` | NVIDIA (Ampere+) | `load_inline` | CuTe abstractions |
| `triton` | NVIDIA, AMD | tempfile + JIT | Cross-vendor |
| `tilelang` | NVIDIA | tempfile + JIT | fp16/bf16 only |
| `helion` | NVIDIA | tempfile + JIT | |
| `hip` | AMD (MI200+) | `load_inline` analog | HIP C++ |
| `nki` | AWS Trainium/Inferentia | tempfile | Compile-check only on non-Neuron HW |
| `pallas` | TPU | tempfile | |
| `numba` | NVIDIA, AMD | tempfile | |
| `mojo` | NVIDIA | tempfile | |

---

## Hardware Registry

### Official Eval Fleet

These are the machines used for official leaderboard evaluation:

| HW key | Count | Vendor | FP16 TFLOP/s | DRAM BW GB/s | Notes |
|---|---|---|---|---|---|
| `nvidia_h200_sxm` | up to 8 | nvidia | 989 | 4800 | GH100 die, HBM3e — primary eval target |
| `nvidia_a100_sxm` | up to 2 | nvidia | 312 | 2000 | SXM4 — secondary / cross-gen problems |
| `aws_trainium2` | TBD | aws | ~832 | ~820 | NKI only; availability TBD |

The H200 is the primary eval target. Multi-GPU problems use up to 8× H200 in a single node (NVLink). The A100 is used for problems that specifically target Ampere-era optimizations (e.g., async copy, TF32). Trainium is used for NKI problems when hardware is available.

### Extended Registry (for problem authoring; not all available in eval fleet)

| HW key | Vendor | FP16 TFLOP/s | DRAM BW GB/s | Notes |
|---|---|---|---|---|
| `nvidia_h100_sxm` | nvidia | 989 | 3350 | SXM5; similar compute to H200, less bandwidth |
| `nvidia_a100_sxm` | nvidia | 312 | 2000 | In fleet |
| `nvidia_h200_sxm` | nvidia | 989 | 4800 | In fleet |
| `amd_mi300x` | amd | 1307 | 5300 | Not in fleet; problems supported, eval requires external HW |
| `amd_mi250x` | amd | 383 | 3277 | Not in fleet |
| `aws_trainium2` | aws | ~832 | ~820 | Fleet availability TBD |
| `google_tpuv4` | google | ~275 | ~300 | Not in fleet; Pallas problems deferred |

---

## Initial Problem Set Ideas

Seed from three sources:

1. **KernelBench level5 hardware_translation_stub** (10+ existing problems: flash_attn2, paged_attn, RoPE, RMSNorm, marlin, etc.) — already have CUDA source, need HIP/Triton/NKI targets added
2. **vLLM / FlashAttention / xFormers** production kernels with known-good CUDA implementations
3. **KernelBench levels 1–4** Triton implementations (from `_translation_sources/triton/`) as source → CUDA/HIP targets

Target: 50 problems at launch across 5+ translation axes.

---

## Open Questions

See `questions.md`.
