# Fused RMSNorm + Residual Add: A100 CUDA → H100 Triton

## Description
Fuses two common LLM decoder operations into one kernel:
  1. Residual addition: z = x + r
  2. RMS normalisation: y = z / rms(z) * w   where rms(v) = sqrt(mean(v²) + ε)

Inputs: `x [B, L, D]`, `r [B, L, D]`, `w [D]` (fp16) → `y [B, L, D]` (fp16).
One block/program normalises one token row (B×L rows total).

Used in LLaMA, Mistral, Gemma, and virtually all recent decoder-only models
as the pre-norm layer before each attention or FFN sub-block.

## Translation Challenges

- **Two-pass reduction vs. single-pass tl.sum**: The A100 source uses an
  explicit two-pass warp reduction: first `__shfl_down_sync` within each warp,
  then a cross-warp reduction via `__shared__ float warp_sums[4]`.
  In Triton, `tl.sum(z * z, axis=0) / D` achieves the same in one line.
  Candidates often mistakenly unroll the loop or forget to divide by D.

- **Shared-memory broadcast of rms_inv**: The CUDA kernel stores the scalar
  `rms_inv` in `__shared__ float` and synchronises before the second pass.
  In Triton the scalar is just a Python-level variable; no explicit sync needed.

- **BLOCK_D must cover full D**: Unlike the CUDA kernel's strided loop
  `for (int i = tid; i < D; i += kNT)`, Triton requires BLOCK_D ≥ D with
  masking.  The candidate must set `BLOCK_D = next_power_of_2(D)` and use
  `mask = offs < D` on every load/store.

- **Dtype handling**: The CUDA kernel hard-codes `__half2float` / `__float2half`
  conversions.  In Triton, `.to(tl.float32)` on load and `.to(x_ptr.dtype.element_ty)`
  on store keeps the kernel dtype-agnostic.

## Known Gotchas

- Missing the `/ D` in `tl.sum(z * z, axis=0) / D` computes variance ∝ N
  instead of mean, producing a numerically different (and wrong) normalisation.

- Setting `eps = 0` causes NaN when a row is all-zeros; use `1e-6` (the value
  used in the LLaMA reference implementation).

- `BLOCK_D` must be a power of two for Triton's register allocator; using
  `min(next_power_of_2(D), 8192)` caps tile size for very large D.

## Provenance
Hand-written for KTBench, pattern from FlashAttention / Apex fused layer-norm
and the LLaMA RMSNorm reference implementation.
