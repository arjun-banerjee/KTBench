# Online Softmax: CUDA H200 → Triton H200

## Description
Row-wise softmax over a 2D tensor [N, D] using the Milakov-Norrie online algorithm
for numerical stability. Each row is independent - no cross-row communication needed.

## Translation Challenges
- **Tiling strategy**: The CUDA version uses one warp per row with a warp-level
  reduction. Triton uses a 1D program grid with `BLOCK_D` tiles per row; the
  translator must choose an appropriate BLOCK_D (ideally a power-of-2 ≥ D).
- **Warp shuffle → Triton reduction**: CUDA uses `__shfl_xor_sync` for the
  warp-level max/sum reduction. In Triton this becomes `tl.max` / `tl.sum`.
- **fp32 accumulation**: Both implementations accumulate in fp32 for numerical
  stability even when the input/output dtype is fp16/bf16.
- **Non-power-of-2 D**: The candidate must handle arbitrary D with masking
  (`mask = cols < D`), not just power-of-2 sizes.

## Known Gotchas
- `tl.exp` operates in fp32 by default - no explicit cast needed.
- `BLOCK_D` must be a `tl.constexpr`; choose it as `triton.next_power_of_2(D)`.
- For very large D (>16384), a multi-pass approach may be needed.

## Provenance
Hand-written for KTBench as a difficulty-2 demonstration problem.
