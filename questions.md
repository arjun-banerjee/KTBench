# KTBench — Open Design Questions

Resolved decisions are marked ✅. Still-open questions are marked ❓. New questions surfaced by the answers are marked 🆕.

---

## 0. Metrics

**Q0.0 ✅ — Port KernelBench's extended metric set**
KTBench should carry all metrics from KernelBench's `KernelExecResult`, mapped to the translation context:

| Metric group | Fields | Notes in translation context |
|---|---|---|
| Correctness | `compiled`, `correctness`, `correctness_trials` | Fraction of structured cases passing |
| Timing | `runtime`, `runtime_stats` | Candidate on target HW |
| Reference timing | `ref_runtime`, `ref_runtime_stats` | Handwritten reference in tgt_dsl on target HW |
| Speedup vs. reference | `speedup_vs_ref` | candidate runtime / handwritten reference runtime |
| Source timing | `source_runtime`, `source_runtime_stats`, `source_backend` | Source kernel on source HW (pre-stored in meta.toml) |
| Numerical precision | `max_abs_error`, `mean_abs_error`, `max_rel_error`, `mean_rel_error` | Per correctness trial, aggregated |
| Memory efficiency | `peak_memory_bytes`, `ref_peak_memory_bytes`, `memory_ratio` | Candidate vs. handwritten reference |
| Kernel launch / fusion | `num_kernels`, `ref_num_kernels`, `fusion_ratio`, `kernel_breakdown` | |
| SOL score | `sol_score`, `arithmetic_intensity`, `achieved_bandwidth_gbps`, `achieved_gflops`, `bottleneck` | Grounded in target HW ceiling |
| Energy efficiency | `energy_mj`, `ref_energy_mj`, `energy_ratio`, `avg_power_w` | |
| Roofline / occupancy | `roofline_efficiency`, `occupancy_pct`, `memory_throughput_pct`, `compute_throughput_pct` | |
| Anti-hack flags | `excessive_speedup`, `suspected_noop`, `static_exploit` | Gate on utilization floor |
| Stress robustness | `stress_pass_rate` | Fraction of randomized stress trials passing |

---

## 1. Problem Format & Test Suite

**Q1.1 ✅ — Who authors the structured correctness cases?**
Auto-generation with human review. Each problem includes a generator; humans curate/approve categories (small, large, non-power-of-two, degenerate, boundary-aligned, adversarial).

**Q1.2 ❓ — Sandbox to prevent reading test_suite.toml**
The container-per-eval (Q3.2) handles filesystem isolation if oracle tensors and test shapes are not mounted into the candidate container. But:

🆕 **Q1.2a — Where do oracle tensors live relative to the eval container?**
If the grader runs in the host process and the candidate runs in the container, the grader can pass oracle tensors as tensors over a socket/pipe — the candidate never sees the file. Is this the intended architecture, or should the grader itself run in the container with oracle tensors injected at launch?

🆕 **Q1.2b — Can the candidate recover shapes from output buffer allocations?**
The grader allocates output buffers of the correct shape and passes them to `ModelNew.forward()`. The candidate can call `.shape` on those buffers. Does this leak the test case shapes? Options:
- (a) Pass shapes explicitly as forward() arguments (they're already implicitly in the input tensors anyway) — accept this is visible
- (b) Use opaque buffers (not standard PyTorch tensors) — complex, breaks DSL interop
- (c) Accept that individual case shapes are visible during the run; the anti-hardcoding protection comes from having many diverse cases + stress cases that are never mounted

**Q1.3 ✅ — Oracle execution strategy**
Pre-run at dataset-build time on source HW. Store output tensors. Version by: source kernel commit, framework version, dtype policy, seed, hardware ID.

**Q1.4 ✅ — Oracle tensor storage**
safetensors, one file per case (or per problem if small). Metadata: shape, dtype, seed, tolerance, source HW, SW versions, checksum.

🆕 **Q1.5 — Who stores the handwritten reference kernel and when is it run?**
The new scoring model (Q4.1) compares the candidate against a handwritten reference implementation in the target DSL. Questions:
- Is the handwritten reference stored in the problem directory (`reference_tgt.py`)?
- Is it public (visible to models in the prompt)? If yes, models can copy it — is that acceptable if they get the right answer?
- Is its timing pre-stored in meta.toml, or re-run live at eval time on target HW?
- Who writes it — problem authors, or is it auto-generated (e.g., Triton from torch.compile for CUDA→Triton problems)?

---

## 2. Correctness Check Design

**Q2.1 ✅ — Entry point interface**
Single-entry `ModelNew.forward()`. Diversity comes from cases exercising different shapes, dtypes, masks, layouts, modes. Multi-entry kernels wrapped behind `forward()`.

**Q2.2 ✅ — Helper methods**
Allowed internally. Only `forward()` is called by the grader. Static checker inspects the whole file.

**Q2.3 ✅ — Tolerance policy**
Per-problem in `meta.toml`, with dtype defaults (fp32: rtol=atol=1e-4; fp16/bf16: rtol=atol=1e-2). Reductions/attention/low-precision can override.

**Q2.4 ✅ — Stochastic kernels**
Excluded from v1.

🆕 **Q2.5 — How are causal masks / attention biases handled in the test suite?**
Some kernels (attention, paged attention) require auxiliary inputs like causal masks, key padding masks, or block tables. These are part of `forward()` arguments. Should these be included in test case inputs in `test_suite.toml` as fixed tensors, or generated by the grader at eval time from the case metadata?

🆕 **Q2.6 — What is the "correctness" definition for kernels that are numerically sensitive by design?**
Some kernels (e.g., online softmax, two-pass reductions) have intermediate accumulation strategies that legitimately produce results slightly outside standard fp16 tolerance when implemented differently (e.g., using f32 accumulation vs bf16). Is tolerance in `meta.toml` the full answer, or do we need a higher-level "functionally equivalent" correctness check (e.g., perplexity equivalence for attention)?

---

## 3. Eval Infrastructure

**Q3.1 ✅ — Eval runtime**
Hardware-aware dispatcher abstraction. v1 supports Modal + local/SLURM. No hard-coded cloud provider.

🆕 **Q3.1a — How is the dispatcher configured per-problem?**
Each problem specifies `tgt_hw`. The dispatcher must map `tgt_hw` → available machine pool. Is this a config file (`configs/hw_map.toml`) that operators fill in for their cluster, or is it expected that the benchmark ships a default Modal mapping for common HW?

**Q3.2 ✅ — Subprocess isolation**
Container-per-eval for official leaderboard runs.

🆕 **Q3.2a — GPU passthrough in container for multi-vendor HW**
NVIDIA containers need `nvidia-docker` / `--gpus` flags; AMD containers need ROCm runtime. Does the eval harness manage container launching, or is this expected to be configured externally (e.g., by a SLURM job template or Modal image)?

**Q3.3 ✅ — Compilation cache**
Content-addressed, keyed by: submission hash, problem ID, target backend, compiler version (CUDA/HIP/Triton), hardware arch. No shared writable state across users.

🆕 **Q3.4 — How is the agentic multi-turn toolset imported from the KernelBench repo?**
Q6.3 answer says to take tooling from the KernelBench repo in `abaner`. Should KTBench vendor a copy, depend on it as a package, or import via a git submodule? And which specific pieces: the agent runner, sweep scripts, timing utilities, or everything under `src/kernelbench/`?

---

## 4. Scoring & Metrics

**Q4.1 ✅ — Performance baseline is handwritten reference in target language**
Candidate is compared against a handwritten (or best-known) reference implementation in the target DSL on the target HW — not against the source kernel. See Q1.5 for open questions about who provides it.

**Q4.2 ✅ — N/A** (decoupled by pre-storing)

**Q4.3 ✅ — Separate leaderboards**
Yes: correctness-only + all individual metrics displayed separately. Primary ranking TBD.

🆕 **Q4.3a — What is the primary ranking metric on the leaderboard?**
Candidates: (a) SOL fraction on target HW, (b) speedup vs. handwritten reference, (c) correctness × stress_pass_rate × SOL, (d) separate correctness and perf rankings. The composite metric from the plan was `correctness × stress_robustness × SOL_tgt/SOL_ref` — is this still right given the new baseline?

**Q4.4 ✅ — Cross-HW performance comparison**
Compare to handwritten reference in target language, not to source HW. Hardware capability differences are absorbed into the reference baseline.

🆕 **Q4.5 — How is energy measurement handled for AMD/NKI HW?**
KernelBench's `measure_energy` uses NVML (NVIDIA only). For AMD (ROCm) and AWS Trainium, different APIs are needed (rocm-smi, neuron-monitor). Should energy be HW-conditional (measured when available, skipped otherwise), or mandatory for all targets?

---

## 5. Problem Sourcing & Coverage

**Q5.1 ✅ — All axes in scope**
All translation axes supported from v1: CUDA↔HIP, CUDA↔Triton, Triton↔CUDA, CUDA→NKI, CUDA→Pallas, and dtype/layout conversions within the same HW.

🆕 **Q5.1a — How many problems per axis in v1?**
"50 problems across 5+ axes" was the plan target. Is the distribution meant to be uniform (10 per axis) or weighted by practical demand (e.g., more CUDA→HIP and CUDA→Triton)?

**Q5.2 ✅ — Source kernel format**
Allow PyTorch, CUDA, or any supported DSL as source. Problem directory format should be flexible on `src_dsl`.

🆕 **Q5.2a — How is a PyTorch source kernel normalized for prompting?**
A raw PyTorch `nn.Module` is already in the ModelNew-compatible format. A raw `.cu` file is not. Is there a normalization step (`add_problem.py`) that wraps raw CUDA files into a `ModelNew`-style Python module, or is the raw `.cu` passed directly to the model?

**Q5.3 ✅ — Source kernel validation**
Requires passing a validation suite against a trusted reference (PyTorch eager or known production impl). Custom kernels require human review + oracle-generation checks.

---

## 6. Prompt Design

**Q6.1 ✅ — Prompt contents**
Include: source code, target backend/HW, interface signature, dtype/layout assumptions, semantic description, allowed libraries.
Exclude: exact hidden test shapes, oracle outputs, stress cases, scoring internals.

🆕 **Q6.1a — Is the handwritten reference kernel (Q1.5) included in the prompt?**
If it exists and is public, it dramatically helps the model. Is this intentional (the task is adaptation/optimization, not blank-sheet translation), or should it be withheld to keep the task harder?

**Q6.2 ✅ — Hardware summary in prompt**
Short summary: memory hierarchy, wavefront/warp size, approximate bandwidth/FLOPs, backend constraints. No per-problem performance targets.

**Q6.3 ✅ — Multi-turn support**
Both single-turn (clean capability comparison) and multi-turn agentic (realistic engineering, stronger sandboxing, hidden tests). Separate leaderboards. Tooling ported from KernelBench repo.

🆕 **Q6.4 — In multi-turn mode, can the agent observe timing outputs?**
If the agent can run the kernel and observe runtime, it can binary-search over tile sizes / launch configurations. This is realistic engineering behavior but creates a new reward hacking surface (agent can probe the grader indirectly). Should timing feedback be available to the agent, and if so, should it be the real eval timing or a sandboxed proxy?
