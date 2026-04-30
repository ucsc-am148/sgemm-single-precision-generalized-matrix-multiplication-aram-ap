# Homework: SGEMM optimization arc (K2–K5)

Implement four progressively faster matrix-multiply kernels in
`numba.cuda`.

You implement K2, K3, K4, and K5. K1 (naive) is given to you as a
worked example so you can see the `@cuda.jit` signature shape every
kernel must match.

## How submission works

This assignment is a GitHub Classroom repo. **You only edit
`kernels.py`** the launch wrappers, tile-size constants, and harness
code are already in place. Push your edits to the `main` branch of this
repo. Each push that touches `kernels.py` triggers the autograder,
which runs on a Modal A100 40GB and posts your grade as a comment on
the commit.

You have **10 graded submissions per assignment**. Use them don't push
half-baked attempts. Run the local sanity check first.

## Grading rubric

The autograder runs each of your kernels on a battery of shapes and
checks two things per kernel:

1. **Correctness**: output matches a reference within fp32 tolerance.
2. **Performance floor**: your kernel runs at least as fast as a
   per-kernel threshold calibrated on the autograder hardware.

Both checks must pass for a kernel to count. Grading is **cumulative**:

| Kernels passed         | Grade |
| ---------------------- | ----- |
| K2                     | C     |
| K2 + K3                | C+    |
| K2 + K3 + K4           | B−    |
| K2 + K3 + K4 + K5      | B     |

K3 only counts if K2 also passes, K4 only if K2+K3, etc. So implement
in order!

## Where to run the sanity check

`sanity_check.py` runs your kernels at 256² and 512² and compares
against a numpy reference. It is **correctness only**. Passing does
not guarantee a passing grade, since the autograder also enforces a
performance floor. K2–K5 will show up as `FAIL` until you implement
them; K1 (worked example) should `PASS` immediately.

You have three options for where to run it:

### Option A: Locally (best dev experience)

Requires a system-installed CUDA 12.x toolkit and a CUDA-capable GPU
(sm_70+). Set up a Python 3.12 virtual environment and install deps:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sanity_check.py
```

### Option B: Google Colab (no local GPU needed)

Colab provides a free T4 GPU.

1. Visit <https://colab.research.google.com> and create a new notebook.
2. Runtime -> Change runtime type -> Hardware accelerator: **T4 GPU**.
3. Run these three cells:

   ```python
   # Cell 1: upload kernels.py and sanity_check.py from your local clone.
   from google.colab import files
   files.upload()  # browse and select both files
   ```

   ```bash
   # Cell 2 install dependencies. Colab ships CUDA already.
   !pip install -q numpy==2.4.4 numba==0.65.0 numba-cuda==0.30.0
   ```

   ```bash
   # Cell 3 run the sanity check.
   !python sanity_check.py
   ```

After editing your kernels, either re-run cell 1 to upload again or use
a `%%writefile kernels.py` magic cell to edit in-notebook. Colab's T4
has a different perf profile than the autograder's A100 correctness
results will track, but timings will not.

### Option C: Modal A100 (matches the autograder)

Modal gives every account $30/month of free credit, which is enough
for thousands of sanity checks on the same A100 40GB the autograder
uses. One run costs about $0.01.

```
pip install modal
modal token new            # one-time browser auth, links your account
modal run modal_sanity.py
```

The script (`modal_sanity.py`, included in this repo) ships your local
`kernels.py` and `sanity_check.py` to a Modal container, runs them on
A100, and streams the output back. Use this when you want
representative timings before spending a graded submission.

## Files in this repo

- `kernels.py`: your edit surface; numba kernels for K1–K5
- `sanity_check.py`: local correctness + rough timing
- `modal_sanity.py`: `modal run` wrapper that runs sanity_check on A100 (Option C above)
- `kernel6_stretch.cu`: optional stretch goal (cupy.RawKernel; see below)
- `requirements.txt`: Python deps
- `README.md`: this file
- `.github/workflows/sgemm-grader.yml`: the autograder Action (don't edit)

## Stretch (K6)

Once K2–K5 pass, you can optionally implement K6 (vectorized `float4`
loads) in `kernel6_stretch.cu`. K6 cannot be expressed in `numba.cuda`
You have to write CUDA C++ launched via `cupy.RawKernel`. To run it
locally you'll need `cupy-cuda12x` (also in `requirements.txt`). The
reward for passing K6 on the autograder is not yet decided; for now,
treat it as a learning bonus.

## Kernel specifications

A short spec for each kernel; the docstring inside each `@cuda.jit`
function in `kernels.py` has more detailed hints.

### K1: Naive (worked example)

One thread per output element. Each thread loops over K and accumulates
`A[x, i] * B[i, y]` into a register, then writes one C element. No
tiling, no shared memory; every multiply pulls fresh from global
memory.

**Provided as a reference implementation; you do not edit K1.**

Use it as a template for the `@cuda.jit` signature and bounds-check
pattern every kernel must follow.

  Tile: implicit 32 × 32 (one thread per element).
  Launch: `block = (BLOCKSIZE, BLOCKSIZE)` (32×32, 2D);
  `grid = (ceil(M/BLOCKSIZE), ceil(N/BLOCKSIZE))`.
  Thread maps directly: `x = blockIdx.x * blockDim.x + threadIdx.x`,
  `y = blockIdx.y * blockDim.y + threadIdx.y`. Note that consecutive
  threads in a warp share `x` and step in `y` (fixed in K2)

### K2: GMEM coalescing

Rewrite K1 so that 32 threads in a warp write to 32 *consecutive
columns* of C (and read 32 consecutive elements of B). The arithmetic
stays identical to K1; only the threadIdx-to-output mapping changes.
The new mapping makes `B[i, y]` reads and `C[x, y]` stores collapse
into 128-byte transactions.

- Tile: 32 × 32 output per block.
- Launch: `block = (BLOCKSIZE * BLOCKSIZE,)` (1024 threads, 1D);
  `grid = (ceil(M/32), ceil(N/32))`.

**Suggestion**: with `threadIdx.x` running 0..1023, derive
`(row_in_tile, col_in_tile)` using integer division and modulo by
`BLOCKSIZE`. Be careful which derived value indexes the column
that's the whole point of the optimization.

### K3: Shared-memory cache-blocking

Compute each `BM3 × BN3` output tile by streaming the K dimension in
chunks of `BK3`. Each input element of A and B should be loaded from
global memory once per block (instead of once per thread, like in K2).

- Tile sizes: `BM3 = BN3 = BK3 = 32`.
- Launch: `block = (BM3 * BN3,)` (1024 threads, 1D);
  `grid = (ceil(M/BM3), ceil(N/BN3))`.

**Suggestion**: per K-chunk, do four steps:

1. Cooperatively load `As[BM3, BK3]` and `Bs[BK3, BN3]` into shared
   memory (one element per thread per slice). Use `0.0` when the
   global index is out of bounds (the K loop walks past the end of K).
2. `cuda.syncthreads()`.
3. Each thread accumulates `acc += As[local_row, dk] * Bs[dk, local_col]`
   for `dk in range(BK3)`.
4. `cuda.syncthreads()` before the next K-chunk.

Use `cuda.shared.array((BM3, BK3), float32)` for `As` and the matching
shape `(BK3, BN3)` for `Bs`.

### K4: 1D register tiling

Extend K3 so that each thread owns `TM4 = 8` rows in a single column of
the `BM4 × BN4` output tile and keeps `TM4` register accumulators. Per
inner-k step, broadcast one `Bs[dk, thread_col]` value and FMA it into
all `TM4` accumulators against `TM4` `As` values from one column of
`As`. Arithmetic intensity goes up by `TM4` over K3.

- Tile sizes: `BM4 = BN4 = 64`, `BK4 = 8`, `TM4 = 8`.
- Launch: `block = ((BM4 * BN4) // TM4,)` (512 threads);
  `grid = (ceil(N/BN4), ceil(M/BM4))`. From K4 onward `blockIdx.x`
  indexes *columns* of C (axis swap from K1–K3); `run_k4` does the
  swap on the launch side.

**Suggestion**: cooperative loads are tidy here. A's tile is
`BM4 × BK4 = 512` elements, B's is `BK4 × BN4 = 512`, and you have
512 threads, so exactly one element per thread per tile (no inner-load
loop). Use `cuda.local.array(TM4, float32)` for the per-thread
accumulator and initialize every entry to 0.0 before the K-loop.

### K5: 2D register tiling

Extend K4 to a `TM5 × TN5 = 8 × 8` register tile per thread. Inside
the inner-k loop, cache `TM5` `As` values and `TN5` `Bs` values into
register arrays and do the `TM5 × TN5` outer-product update: 64 FMAs
per dk step against `TM5 + TN5 = 16` register loads.

- Tile sizes: `BM5 = BN5 = 128`, `BK5 = 8`, `TM5 = TN5 = 8`.
- Launch: `block = ((BM5 * BN5) // (TM5 * TN5),)` (256 threads);
  `grid = (ceil(N/BN5), ceil(M/BM5))`.

**Suggestion**: cooperative loads now need a stride loop because the
tile has `BM5 * BK5 = 1024` elements but the block has only 256
threads, so each thread loads `1024 / 256 = 4` elements of A per
K-chunk and similarly for B. Pick the per-thread row stride so
consecutive threads touch consecutive memory addresses (= coalesced
GMEM loads). Use `cuda.local.array((TM5, TN5), float32)` for the
accumulators (numba supports tuple-shaped local arrays); two more
`cuda.local.array(TM5, float32)` / `cuda.local.array(TN5, float32)`
hold the cached `reg_a` / `reg_b` for the outer product.

## References

- siboehm's blog: <https://siboehm.com/articles/22/CUDA-MMM>
- Numba CUDA docs: <https://numba.readthedocs.io/en/stable/cuda/>
- Course lectures 10 and 11 cover the optimization arc this assignment
  walks through.
