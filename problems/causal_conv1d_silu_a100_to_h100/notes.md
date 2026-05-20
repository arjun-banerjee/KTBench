# Causal Depthwise Conv1D + SiLU: A100 CUDA → H100 Triton

## Description
Depthwise 1D convolution (one independent filter per channel, `groups=D`) with
filter width 4 and a causal (left-padded) boundary condition.  The SiLU
activation `x / (1 + exp(-x))` is fused into the output.

Tensor shapes: `x [B, D, L]`, `weight [D, 4]` → `y [B, D, L]` (fp16).

Used as the sequence mixer in Mamba/S4 state space models and many linear
RNN variants.  Provenance: adapted from Dao-AILab/causal-conv1d (Apache 2.0,
Tri Dao 2024).

## Translation Challenges

- **Causal history management**: The CUDA source uses a shared-memory ring
  buffer (`hist[]`) to pass the 3 prior elements between successive 512-element
  chunks.  Triton has no built-in inter-program state; instead the candidate
  must use masked loads (`other=0.0`) with negative offset indices to achieve
  the same causal zero-padding semantics.

- **Cooperative vs. independent loading**: The A100 kernel uses 128 threads to
  cooperatively fill a shared-memory staging buffer before each thread computes
  its 4 outputs.  The Triton idiom is for each program to load its slice
  independently with `tl.load` — the candidate must recognise this mismatch.

- **Static unroll of the tap loop**: In CUDA, `#pragma unroll` over `kW=4` is
  explicit.  In Triton, `tl.static_range(kW)` achieves the same unrolling.
  Using a dynamic Python `range` instead will produce a runtime loop and
  significantly hurt performance.

- **SiLU in Triton**: `tl.exp` is available; the candidate must apply
  `acc / (1.0 + tl.exp(-acc))` element-wise on the output vector — not loop
  over scalars.

## Known Gotchas

- The tap offset `out_offs - (kW - 1 - wi)` can be negative for positions near
  the start of the sequence.  The mask `(tap_offs >= 0)` is essential; missing
  it causes out-of-bounds reads that may return garbage values.

- BLOCK_L must be a `tl.constexpr`; choose `triton.next_power_of_2(L)` capped
  at a reasonable tile size (512 works well for H100).

- Storing results as `x_ptr.dtype.element_ty` rather than hard-coding
  `tl.float16` keeps the kernel dtype-agnostic.

## Provenance
Kernel adapted from Dao-AILab/causal-conv1d (Apache 2.0, Tri Dao 2024),
simplified to a single width=4 forward-pass kernel for KTBench.
