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

    # Derive the global row (x) and column (y) indices for this thread's output element.
    x = cuda.blockIdx.x * BLOCKSIZE + (cuda.threadIdx.x // BLOCKSIZE)
    y = cuda.blockIdx.y * BLOCKSIZE + (cuda.threadIdx.x % BLOCKSIZE)

    # Same arithmetic as K1, but now threads with consecutive threadIdx.x write to consecutive columns of C.
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

    # Shared memory tiles for A and B
    As = cuda.shared.array((BM3, BK3), dtype=float32)
    Bs = cuda.shared.array((BK3, BN3), dtype=float32)

    # Compute the global row and column indices for this thread's output element in C
    thread_id_x = cuda.threadIdx.x
    row_in_tile = thread_id_x // BN3
    col_in_tile = thread_id_x % BN3

    row = cuda.blockIdx.x * BM3 + row_in_tile
    col = cuda.blockIdx.y * BN3 + col_in_tile

    tmp = float32(0.0)

    # Loop over K dimension in chunks of BK3
    for i in range(0, K, BK3):
        # load A and B tiles into shared memory within bounds
        As[row_in_tile, col_in_tile] = A[row, col_in_tile + i] if (row < M and col_in_tile + i < K) else float32(0.0)
        Bs[row_in_tile, col_in_tile] = B[row_in_tile + i, col] if (row_in_tile + i < K and col < N) else float32(0.0)
        cuda.syncthreads()

        # dot the row of As into the column of Bs to update the per-thread accumulator
        for k in range(BK3):
            tmp += As[row_in_tile, k] * Bs[k, col_in_tile]

        cuda.syncthreads()

    if row < M and col < N:
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

    As = cuda.shared.array((BM4, BK4), dtype=float32)
    Bs = cuda.shared.array((BK4, BN4), dtype=float32)
    
    # Compute the global column index for this block
    tx = cuda.threadIdx.x

    thread_row = tx // BN4
    thread_col = tx % BN4

    a_row, a_col = tx // BK4, tx % BK4
    b_row, b_col = tx // BN4, tx % BN4

    c_col = cuda.blockIdx.x * BN4
    c_row = cuda.blockIdx.y * BM4

    accum = cuda.local.array(TM4, float32)
    for i in range(TM4):     # Initialize accum with 0.0f
        accum[i] = float32(0.0)

    for k in range(0, K, BK4):
        # Bounds-safe cooperative loads
        As[a_row, a_col] = A[c_row + a_row, a_col + k] if (c_row + a_row < M and k + a_col < K) else float32(0.0)
        Bs[b_row, b_col] = B[b_row + k, c_col + b_col] if (k + b_row < K and c_col + b_col < N) else float32(0.0)
        cuda.syncthreads()

        for j in range(BK4):
            b_val = Bs[j, thread_col]
            for m in range(TM4):
                # accumulate dot product
                accum[m] += As[thread_row*TM4 + m, j] * b_val
        cuda.syncthreads()

    for m in range(TM4):
        gr = c_row + thread_row * TM4 + m
        gc = c_col + thread_col
        if gr < M and gc < N:
            C[gr, gc] = accum[m]



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

    NUM_THREADS = (BM5*BN5) // (TM5*TN5) # 256 threads per block
    # BM5*BK5 = 1024 A elements / 256 threads = 4 loads on each--same for B.
    NUM_A_ELEM = (BM5 * BK5) // NUM_THREADS # 4
    NUM_B_ELEM = (BK5 * BN5) // NUM_THREADS # 4
    A_ROW_STRIDE = NUM_THREADS // BK5 # 32: 256 threads / 8 K-columns=32 rows per pass
    B_ROW_STRIDE = NUM_THREADS // BN5 # 2: 256 threads / 128 N-columns = 2 rows per pass

    As = cuda.shared.array((BM5, BK5), dtype=float32)
    Bs = cuda.shared.array((BK5, BN5), dtype=float32)

    tx = cuda.threadIdx.x
    thread_col = tx % (BN5 // TN5)
    thread_row = tx // (BN5 // TN5)

    c_col = cuda.blockIdx.x * BN5
    c_row = cuda.blockIdx.y * BM5

    # row/col index for loading A/B
    a_row, a_col = tx // BK5, tx % BK5 
    b_row, b_col = tx // BN5, tx % BN5

    accum = cuda.local.array((TM5, TN5), float32)
    for m in range(TM5):     # Initialize accum with 0.0f
        for n in range(TN5):
            accum[m, n] = float32(0.0)
    
    for k in range(0, K, BK5):
        # Load A tile
        for s in range(NUM_A_ELEM):
            lr = a_row + s * A_ROW_STRIDE
            As[lr, a_col] = A[c_row + lr, k + a_col] if (c_row + lr < M and k + a_col < K) else float32(0.0)
        # Load B tile
        for s in range(NUM_B_ELEM):
            lr = b_row + s * B_ROW_STRIDE
            Bs[lr, b_col] = B[k + lr, c_col + b_col] if (k + lr < K and c_col + b_col < N) else float32(0.0)

        cuda.syncthreads()

        for j in range(BK5):
            # Cache column of As and row of Bs into registers
            reg_a = cuda.local.array(TM5, dtype=float32)
            reg_b = cuda.local.array(TN5, dtype=float32)

            for m in range(TM5):
                reg_a[m] = As[thread_row * TM5 + m, j]
            for n in range(TN5):
                reg_b[n] = Bs[j, thread_col * TN5 + n]
            # Calculate outer product with each (m, n) pair
            for m in range(TM5):
                for n in range(TN5):
                    accum[m, n] += reg_a[m] * reg_b[n]

        cuda.syncthreads()
    
    # Write register tile back to global mem
    for m in range(TM5):
        for n in range(TN5):
            gr = c_row + thread_row * TM5 + m
            gc = c_col + thread_col * TN5 + n
            if gr < M and gc < N:
                C[gr, gc] = accum[m, n]


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
