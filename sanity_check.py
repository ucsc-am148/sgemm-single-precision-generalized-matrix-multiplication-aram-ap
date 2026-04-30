"""Local sanity check for the hw-sgemm kernels.

Runs each kernel from kernels.py at one square shape and two
asymmetric (M != N) shapes, comparing against numpy A @ B. Reports
max abs error per kernel + a rough timing. The asymmetric shapes
catch a common class of bug where you swap blockIdx.x / blockIdx.y
in your kernel; with square inputs gridDim ends up the same in both
directions and the bug hides.

This is NOT the autograder. It verifies correctness and prints a
ballpark GFLOPs number; the Modal autograder runs additional shapes
and enforces a per-kernel performance floor. Passing locally is
necessary but not sufficient for a passing grade.

Until you implement K2-K5, those kernels will leave C as zeros and
show up as FAIL. K1 (worked example) should PASS as soon as you run
this.

Usage: python sanity_check.py
"""
import os
import time

# Silence numba's "low occupancy" hints — at sanity-check sizes the grids are
# small on purpose. The autograder runs at sizes large enough to saturate the
# A100, where occupancy is what it is.
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")

import numpy as np
from numba import cuda

import kernels


SIZES = [
    (256, 256, 256),    # square baseline
    (768, 256, 512),    # M > N: catches blockIdx.x/y axis confusion bugs
    (256, 768, 512),    # N > M: mirror — same class of bug, opposite direction
]
TOL = 1e-2   # max abs error tolerance (lenient — fp32 roundoff at this size)


def bench_one(fn, dA, dB, dC, M, N, K, warmup=2, iters=5):
    for _ in range(warmup):
        fn(dA, dB, dC, M, N, K)
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(dA, dB, dC, M, N, K)
    cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run_one(name, fn, A, B, ref, M, N, K):
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.to_device(np.zeros((M, N), dtype=np.float32))
    try:
        fn(dA, dB, dC, M, N, K)
        cuda.synchronize()
    except Exception as e:
        print(f"{name:14s} {M:>5d} {N:>5d} {K:>5d}  CRASH: {type(e).__name__}: {e}")
        return
    out = dC.copy_to_host()
    err = float(np.max(np.abs(out - ref)))
    t = bench_one(fn, dA, dB, dC, M, N, K)
    g = 2.0 * M * N * K / t / 1e9
    verdict = "PASS" if err < TOL else "FAIL"
    print(f"{name:14s} {M:>5d} {N:>5d} {K:>5d} {err:>12.4f} {t*1000:>8.3f} {g:>9.1f}  [{verdict}]")


def main():
    np.random.seed(0)
    print(f"{'kernel':14s} {'M':>5} {'N':>5} {'K':>5} {'max abs err':>12} {'ms':>8} {'GFLOPs':>9}")
    print("-" * 72)
    for (M, N, K) in SIZES:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        ref = A @ B
        run_one("k1_naive*", kernels.run_k1, A, B, ref, M, N, K)   # worked example
        for name, fn in kernels.KERNELS:
            run_one(name, fn, A, B, ref, M, N, K)
    print()
    print("k1_naive is the worked example (always PASS). K2-K5 are graded.")
    print("Note: the Modal autograder also enforces a per-kernel performance floor.")
    print("A passing sanity check is necessary but not sufficient for a passing grade.")


if __name__ == "__main__":
    main()
