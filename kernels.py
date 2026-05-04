"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing (TODO) ──────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE. 
    Be careful which one indexes the column.
    """

    x = cuda.blockIdx.x * BLOCKSIZE + (cuda.threadIdx.x // BLOCKSIZE)
    y = cuda.blockIdx.y * BLOCKSIZE + (cuda.threadIdx.x % BLOCKSIZE)

    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp



# ── K3: shared-memory cache-blocking (TODO) ─────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """

    As = cuda.shared.array((BM3, BK3), dtype=float32)
    Bs = cuda.shared.array((BK3, BN3), dtype=float32)

    thread_id_x = cuda.threadIdx.x
    row_in_tile = thread_id_x // BN3
    col_in_tile = thread_id_x % BN3

    row = cuda.blockIdx.x * BM3 + row_in_tile
    col = cuda.blockIdx.y * BN3 + col_in_tile

    tmp = float32(0.0)
    for i in range(0, K, BK3):
        As[row_in_tile, col_in_tile] = A[row, col_in_tile + i]
        Bs[row_in_tile, col_in_tile] = B[row_in_tile + i, col]
        cuda.syncthreads()

        for k in range(BK3):
            tmp += As[row_in_tile, k] * Bs[k, col_in_tile]

        cuda.syncthreads()

    C[row, col] = tmp

    return


# ── K4: 1D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """
    # TODO
    return


# ── K5: 2D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """
    # TODO
    return


# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]
