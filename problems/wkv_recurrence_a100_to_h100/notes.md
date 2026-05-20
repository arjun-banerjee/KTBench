# WKV Token-Mixing Recurrence: A100 CUDA → H100 Triton

## Description
Implements the WKV attention-free token-mixing used in RWKV-5:

    state[n] ← 0
    for t in range(T):
        kv[n]  = k[t,n] · v[t,n]
        y[t,n] = r[t,n] · (u[n] · kv[n] + state[n])
        state[n] = exp(w[n]) · state[n] + kv[n]

Inputs: `r, k, v [B,H,T,N]` fp16, `w [H,N]` fp32 (log-decay ≤ 0), `u [H,N]` fp16.
Output: `y [B,H,T,N]` fp16.

`w` (time-mixing weights) control how fast each feature dimension forgets older
context.  `u` (bonus) gives the current time step additional weight regardless
of the decay — a key difference from standard attention.

This kernel is adapted from BlinkDL/RWKV-LM WKV5 (Apache 2.0) and is the
inner loop of every RWKV-5/6 language model inference pass.

## Translation Challenges

- **Scalar-per-thread → vector-per-program**: The A100 source assigns one
  CUDA thread per feature dimension n and maintains a scalar `state` register.
  In Triton, a single program handles a `BLOCK_N`-wide slice of N with a
  vector state: `state = tl.zeros((BLOCK_N,), dtype=tl.float32)`.

- **Non-contiguous time stride**: The inner loop reads `r_ptr[base + t*stride_t + n]`.
  In Triton the offset `t_off = t * stride_t` must be a Python integer, and
  `tl.load(r_ptr + base + t_off + n_offs, ...)` performs a contiguous vector
  load of BLOCK_N elements.  A common mistake is computing the pointer offset
  inside tl.load with mixed Python/Triton arithmetic, which causes type errors.

- **Constant preloading**: The A100 source loads `dec` and `un` once before
  the loop.  In Triton, `decay` and `u` should similarly be loaded once before
  the T-loop, not inside it.

- **fp32 accumulation**: `w` is fp32; the scan state must stay in fp32 to avoid
  precision loss over long sequences.  Load `w` as `tl.float32` and cast `r, k, v`
  to fp32 inside the loop with `.to(tl.float32)`.

- **Dynamic loop length**: T is not known at compile time.  Use a Python-level
  `for t in range(T)` (not `tl.static_range`).  Triton compiles this correctly
  as a dynamic loop; no special annotation is required.

## Known Gotchas

- The bonus term `u` multiplies `kv` at time t — it is NOT the same as the
  state update.  Forgetting to include `u * kv` in `yt` is a common error that
  makes the first time step wrong.

- `state` is NOT reset between sequences in the same batch — it is initialised
  to zero once before the T-loop.  Resetting inside the loop produces zeros for
  all y values.

- `tl.exp(w)` computes exp in fp32; since `w ≤ 0`, this always produces
  values in (0, 1].  If w is mistakenly cast to fp16 before `tl.exp`, underflow
  can make the decay exactly 0 for large |w|.

## Provenance
Adapted from BlinkDL/RWKV-LM WKV5 forward CUDA kernel (Apache 2.0).
Kernel structure from `RWKV-v5/cuda/wkv5_cuda.cu` in the RWKV-LM repository.
