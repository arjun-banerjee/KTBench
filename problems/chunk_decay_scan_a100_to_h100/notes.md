# Chunked Exponential-Decay Scan: A100 CUDA → H100 Triton

## Description
Computes a per-dimension first-order IIR (infinite impulse response) scan:

    y[t] = decay · y[t-1] + x[t],   y[-1] = 0
    where decay[h,d] = exp(log_decay[h,d]) ∈ (0, 1)

Equivalently, this is the exponential-decay cumulative sum:

    y[b,h,t,d] = ∑_{s=0}^{t} x[b,h,s,d] · exp((t-s)·log_decay[h,d])

Tensor shapes: `x [B,H,L,D]` fp16, `log_decay [H,D]` fp32 → `y [B,H,L,D]` fp16.

This is the core inner operation of Gated Linear Attention (Yang et al. 2023)
and RetNet (Sun et al. 2023): it accumulates a running key-value state that
exponentially forgets older context.

## Translation Challenges

- **Sequential scan → Triton inner loop**: The A100 source uses a plain
  C++ `for` loop over L with state living in registers.  In Triton, the
  equivalent is an inner `tl.static_range` loop over a BLOCK_L tile of L.
  A common mistake is to try using `tl.associative_scan` directly — this
  requires providing an associative combine function and careful handling of
  the tile-to-tile carry state.

- **State carry across tiles**: The running `state` variable must be
  explicitly carried from one BLOCK_L tile to the next.  The candidate must
  initialise `state = tl.zeros(...)` before the outer tile loop and update it
  at the end of each tile, not reset it.

- **Decay in fp32**: `log_decay` is stored as fp32 to avoid underflow.
  The candidate must load it as `tl.float32`, compute `tl.exp(decay)`, and
  accumulate the scan state in `tl.float32` regardless of the input dtype.

- **Strided 2D load**: Each tile is `[BLOCK_L, BLOCK_D]`.  The pointer
  arithmetic `base_ptr + l_offs[:, None] * stride_l` produces a 2D load
  which maps to efficient bulk memory transfers on H100.

## Known Gotchas

- `tl.static_range(BLOCK_L)` is required for the inner tile loop; using
  Python `range(BLOCK_L)` at kernel level produces a Triton error because
  `BLOCK_L` is a `tl.constexpr` that cannot be used as a Python int.

- The in-place `out_tile` update via `tl.where` is correct but may be slow
  for large BLOCK_L.  An alternative is to accumulate directly:
  `state = state * decay + x_tile[t, :]` and store `state` at each step.

- Ensure `log_decay` has shape `[H, D]` and is indexed as `h * D + d_offs`,
  not `h + d_offs * H` (row-major vs. column-major confusion).

## Provenance
Hand-written for KTBench; kernel pattern from the inner loop of Gated Linear
Attention (Yang et al., GLA, 2023) and RetNet (Sun et al., 2023).
