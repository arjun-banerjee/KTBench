# KTBench — Open Design Questions

These are unresolved decisions that need answers before implementation begins.
Questions are grouped by component and roughly ordered by blocking dependency.

---

## 1. Problem Format & Test Suite

**Q1.1 — Who authors the structured correctness cases?**
The test_suite.toml includes curated cases (small/large/nonpow2/degenerate). These need to be written by a human or generated and validated for each problem. What is the authoring workflow — manual curation, auto-generation with human review, or something else?

**Q1.2 — Should the structured cases be public or held-out?**
If the test cases are visible to the model (e.g., it can read `test_suite.toml`), it can still hardcode for those specific shapes. Options:
- (a) Structured cases are fully public; only stress cases are hidden → simpler but still gameable on structured cases
- (b) Structured cases are held-out at eval time, only shape descriptions in prompt → stronger anti-hack
- (c) Structured cases are public but stress cases are generated server-side per-run → leaderboard-only protection

**Q1.3 — How is the oracle executed when source and target HW differ?**
For e.g. H100 → MI300, the oracle must run on one of them. Options:
- (a) Pre-run oracle at dataset-build time on source HW, store output tensors alongside test cases → eval machine only needs target HW
- (b) Run oracle live on source HW during eval → requires both HWs simultaneously, complex infra
- (c) PyTorch eager always used as oracle → simpler but may not match source kernel numerics for low-precision ops

Which is the right default? Pre-storing (a) seems right for most cases, but raises the question of how oracle tensors are versioned and validated.

**Q1.4 — Oracle tensor storage format?**
If pre-storing oracle tensors: safetensors? pickle? npy? Per-case or batched? How large can this get for a problem with 10 cases × large tensors (e.g., 8192-length sequences)?

---

## 2. Correctness Check Design

**Q2.1 — What does "diverse kernels in correctness check" mean precisely?**
The current plan has each structured case call `model.forward(*case_inputs)` once. But some kernels have multiple callable entry points or modes (e.g., paged attention has `paged_attention_v1` and `paged_attention_v2`). Should the test suite be able to call different entry points per case, or is the `ModelNew.forward()` interface always the single entry point?

**Q2.2 — Should the model be able to expose multiple kernels / functions?**
Related to Q2.1: some GPU libraries expose a family of kernels (e.g., different tile sizes, different precisions) under one wrapper that dispatches. Should `ModelNew` be allowed to have helper methods, or must everything go through `forward()`? This affects both the prompt design and what the static checker needs to inspect.

**Q2.3 — Tolerance policy for cross-hardware correctness?**
fp16 and bf16 arithmetic may differ between NVIDIA and AMD due to different rounding, fused ops, etc. A strict `allclose(atol=1e-2)` may reject legitimate correct translations. Options:
- (a) Per-dtype tolerances (same as KernelBench: fp32=1e-4, fp16/bf16=1e-2)
- (b) Per-problem tolerances declared in `meta.toml` (more flexible, more work)
- (c) Relative-only tolerance with a generous rtol (handles scale-dependent ops)
What level of cross-vendor numerical divergence is acceptable?

**Q2.4 — How to handle kernels with non-deterministic outputs?**
Some kernels (e.g., dropout, sampling) are stochastic. The oracle stores an expected distribution, not a fixed output. Options:
- (a) Exclude stochastic kernels from the benchmark (simplest)
- (b) Fix RNG seeds and require bit-exact match within tolerance (works if both HWs implement same RNG)
- (c) Statistical correctness (mean/variance comparison) — much more complex

---

## 3. Eval Infrastructure

**Q3.1 — Where does eval actually run?**
Options:
- (a) Modal (current KernelBench approach) — cloud GPU on-demand
- (b) Local cluster / SLURM
- (c) Both, with a hardware-aware dispatcher that routes to correct HW vendor
For cross-vendor problems (NVIDIA → AMD), jobs must be dispatched to different machines. Is there existing infra for this, or does it need to be built?

**Q3.2 — Subprocess isolation: how deep?**
The plan calls for running candidate eval in a subprocess to block grader access. How isolated should it be?
- (a) `subprocess.Popen` with restricted environment (easy, blocks most exploits)
- (b) Docker/container per eval (stronger, higher overhead)
- (c) seccomp/namespace isolation (strongest, complex)
Given the pkill and stack-introspection attacks seen in KernelBench, is (a) sufficient?

**Q3.3 — How is compilation cached across eval runs?**
KernelBench uses `torch_extensions` build directories. For a translation benchmark with many DSLs, compiled artifacts can be large. Is there a plan for a shared compilation cache, or does each eval run compile from scratch?

---

## 4. Scoring & Metrics

**Q4.1 — How is SOL_src_equiv computed?**
The translation efficiency score requires knowing how efficiently the source kernel runs on source HW. Options:
- (a) Pre-compute and store in `meta.toml` at dataset-build time → immutable, simple
- (b) Re-run source kernel on source HW at eval time → requires source HW present
- (c) Use a hardware-normalized estimate (FLOP count / HW peak) → avoids needing source HW but less accurate

**Q4.2 — What happens to the score when source HW is not available at eval time?**
If a submission is evaluated on a cluster that only has the target HW (e.g., AMD), can we still compute SOL_tgt / SOL_src_equiv? Pre-storing SOL_src in meta.toml (option a above) decouples this.

**Q4.3 — Is there a separate correctness-only leaderboard?**
For models that produce correct but slow translations, a correctness-only metric is still meaningful. Should the benchmark report correctness and SOL separately on the leaderboard, or only the combined score?

**Q4.4 — How to handle problems where the target HW can legitimately beat the source HW?**
e.g., H100 → H200: the translation may be trivially correct and faster because H200 has 4800 GB/s vs H100's 3350 GB/s. The SOL_tgt / SOL_src_equiv ratio may be > 1 not due to better translation but just better hardware. Is capping at 2.0 the right normalization, or should we normalize by the HW bandwidth ratio?

---

## 5. Problem Sourcing & Coverage

**Q5.1 — Which translation axes should be in v1?**
Candidate axes ranked by feasibility and interest:
1. CUDA (H100) → HIP (MI300) — highest practical demand
2. CUDA (H100) → Triton — many existing Triton sources to compare against
3. Triton (A100) → CUDA (H100) — reverse direction, tests understanding of Triton idioms
4. CUDA (H100) → NKI (Trainium2) — novel, harder to evaluate without Trainium HW
5. CUDA (H100) → Pallas (TPU) — interesting but TPU eval infra is complex
Which axes should be in v1 vs. deferred?

**Q5.2 — Should the source kernel be the full `.cu` file or a ModelNew wrapper?**
For production kernels (flash_attn2, vLLM paged_attn), the source is a large multi-file CUDA project. Should the problem normalize this into a single ModelNew-style file, or should the model be given the raw source files? The former is more tractable for eval; the latter is more realistic.

**Q5.3 — How do we validate that the source kernel is itself correct?**
If we seed problems from production kernels, we assume they are correct. But for custom-authored source kernels, how do we validate the source before including it? Run against PyTorch eager? Have a human review?

---

## 6. Prompt Design

**Q6.1 — What exactly is in the prompt?**
Proposed: source kernel code + target DSL/HW description + ModelNew interface signature. NOT in prompt: input shapes, test case details, oracle outputs.
Is this the right information budget? Too much context → model can exploit structure; too little → unfair for hard translation tasks.

**Q6.2 — Should the prompt include hardware spec sheets?**
e.g., "MI300X has 192 GB HBM3, 5.3 TB/s bandwidth, 304 CUs, 1307 TFLOP/s FP16." This is useful context for hardware-aware translation but also gives away performance targets. Include or exclude?

**Q6.3 — Multi-turn vs. single-turn?**
KernelBench supports agentic multi-turn (agent can run code, observe errors, iterate). Should KTBench also support this? Multi-turn is more realistic but makes eval more expensive and introduces new reward hacking surfaces (agent can observe intermediate timing outputs).
