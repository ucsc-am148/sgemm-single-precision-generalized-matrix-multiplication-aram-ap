// K6 stretch goal: vectorized float4 GMEM/SMEM loads on top of K5.
//
// numba.cuda cannot express float4 / reinterpret_cast, so this kernel
// is written in CUDA C++ and launched via cupy.RawKernel. Implementing
// this is OPTIONAL. The reward is TBD -- for now, treat it as a learning
// bonus.
//
// To use: implement the body below, then load and launch the kernel
// from a Python script using:
//
//     import cupy as cp, pathlib
//     code = pathlib.Path("kernel6_stretch.cu").read_text()
//     k6 = cp.RawKernel(code, "sgemm_vectorize",
//                       options=("-std=c++17", "--use_fast_math"))
//     k6((grid_x, grid_y), (256,), (dA, dB, dC, M, N, K))
//
// The launch shape matches K5 (256 threads, BM=BN=128, BK=8, TM=TN=8).
// Vectorization requires M, N, K all multiples of 4.
//
// Hints (siboehm K6):
// - Load A as float4, then SCATTER it transposed into SMEM so As is
//   stored as [BK rows x BM cols] -- this makes the inner-k step's TM
//   register loads a contiguous 8-float read.
// - Load B as float4 and store linearly.
// - In the inner loop, two float4 reads cover the TM = 8 reg_a entries,
//   and two more cover the TN = 8 reg_b entries.
// - Output stores are also vectorized: two float4 stores per row of the
//   thread's 8x8 result tile.


/* 
 * -----------------------------------
 * A100 Specs (SXM4 40GB, Ampere GA100):
 *
 * Compute
 *      108 SMs | 64 FP32 cores/SM -> 6912 FP32 CUDA cores total
 *      Peak FP32: 19.5 TFLOPS
 *
 * Thread hierarchy
 *      Warp size: 32 threads
 *      Max warps/SM: 64 (2048 threads/SM)
 *      Total GPCs: 7
 *      Max SMs/GPC: 16
 *      Max threads/block: 1024
 *      Registers/SM: 65536 x 32-bit (256 KB)
 *
 * Memory
 *      L1 + SMEM/SM: 192 KB combined; SMEM configurable up to 164 KB
 *      L2 cache: 40 MB
 *      HBM2 bandwidth: 1555 GB/s
 *
 * Roofline ridge point: 19500 GFLOPS / 1555 GB/s ~= 12.5 FLOP/byte
 *      kernel must reuse each loaded float >12.5x to be compute-bound
 * 
 * Data obtained from: 
 *      https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pd
 *      https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
 * 
 * ----------------------------------
 * 
 * Notes on optimizing this kernel:
 *
 * Constraints: 
 *  - All values (BM, BN, BK, TM, TN) must be multiples of 4.
 *  - A tile must split evenly across threads -> (BM * BK) % (4 * THREADS) == 0
 *  - B tile must split evenly across threads -> (BK * BN) % (4 * THREADS) == 0
 *  - A_ROW_STRIDE must divide BM -> BM % A_ROW_STRIDE == 0
 *  - B_ROW_STRIDE must divide BK -> BK % B_ROW_STRIDE == 0
 * 
 * Loading does exactly 1 float4 load per thread per tile,
 *  - So, total # of loads = (BM * BN) / 4. 
 *  - Ideally, lower BM/BN values would reduce num loads (good?)
 * 
 * BENCHMARK RESULTS (Modal, A100 SXM4 40GB, 1 run each):
 *  - (BM, BN, BK, TM, TN) => GFLOPs
 *
 *  - (128, 128, 8, 8, 8)  => 343.1, 1135.9, 1139.0
 *  - (128, 128, 16, 8, 8) => 356.1, 1176.6, 1191.2
 *  - (128, 64, 16, 8, 4)  => 581.8, 1976.7, 1970.2
 *  - (64, 64, 16, 4, 4)   => 584.2, 3028.8, 3040.8
 * 
 * ----------------------------------
 * Observations:
 *  - I bruteforced these results, but I have an idea why (64, 64, 16, 4, 4) is better than (128, 128, 8, 8, 8):
 *      - It has the smallest BM and BN, which minimizes the number of loads (1024 total) vs 4096 for the (128, 128, *, *, *) configs.
 *      - BK=16 halves the sync rate vs BK=8, which helps a lot since we have to sync after every tile load.
 *      - TM=TN=4 is enough to keep the threads busy with compute while waiting 
 *      - The A100 has max AI of around 12.5 FLOP/byte
 *          - AI = (BM*BN)/(2*(BM+BN)) since we load BM*BN floats and do 2*BM*BN FLOPs per tile
 *          - 64/64 AI = (64*64)/(2*(64+64)) = 16 FLOP/byte
 *          - 128/128 AI = (128*128)/(2*(128+128)) = 32 FLOP/byte
 *          - both are above the ridge point, but the smaller tiles have greater occupancy
 */

// Tunables. launch must use THREADS = (BM*BN)/(TM*TN) threads/block.
// Constraints:
//   - BK, BN, TM, TN must be multiples of 4
//   - A tile (BM*BK) and B tile (BK*BN) must split evenly across THREADS
//   - A_ROW_STRIDE must divide BM, B_ROW_STRIDE must divide BK
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

// Derived
#define THREADS            ((BM * BN) / (TM * TN))
#define A_THREADS_PER_ROW  (BK / 4)                       // float4s per A row
#define B_THREADS_PER_ROW  (BN / 4)                       // float4s per B row
#define A_ROW_STRIDE       (THREADS / A_THREADS_PER_ROW)  // m-rows covered per pass
#define B_ROW_STRIDE       (THREADS / B_THREADS_PER_ROW)  // k-rows covered per pass
#define A_VECS_PER_THREAD  ((BM * BK) / (4 * THREADS))    // float4 loads of A per thread per chunk
#define B_VECS_PER_THREAD  ((BK * BN) / (4 * THREADS))    // float4 loads of B per thread per chunk

extern "C" __global__
void sgemm_vectorize(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {

    const int b_row = blockIdx.y; // BM-row tile index
    const int b_col = blockIdx.x; // BN-col tile index

    const int t_row = threadIdx.x / (BN / TN); // m-reg-tile row
    const int t_col = threadIdx.x % (BN / TN); // n-reg-tile col

    // SMEM for A and B tiles
    __shared__ float As[BK][BM]; // Transposed -> BK rows, BM cols
    __shared__ float Bs[BK][BN]; // BK rows, BN cols

    // Reg tiles for C
    float reg_c[TM][TN] = {0};
    float __align__(16) reg_a[TM];
    float __align__(16) reg_b[TN];

    // Per-thread starting position in each tile load
    const int a_M_base = threadIdx.x / A_THREADS_PER_ROW;
    const int a_K_grp  = threadIdx.x % A_THREADS_PER_ROW;
    const int b_K_base = threadIdx.x / B_THREADS_PER_ROW;
    const int b_N_grp  = threadIdx.x % B_THREADS_PER_ROW;

    for (int k_base = 0; k_base < K; k_base += BK) {
        // Load A transposed: As[k, m] = A[m, k]
        // A_VECS_PER_THREAD float4s/thread
        #pragma unroll
        for (int v = 0; v < A_VECS_PER_THREAD; ++v) {
            const int m = a_M_base + v * A_ROW_STRIDE;
            const float4 a_vec = *reinterpret_cast<const float4*>(
                A + (b_row * BM + m) * K + (k_base + a_K_grp * 4));
            As[a_K_grp * 4 + 0][m] = a_vec.x;
            As[a_K_grp * 4 + 1][m] = a_vec.y;
            As[a_K_grp * 4 + 2][m] = a_vec.z;
            As[a_K_grp * 4 + 3][m] = a_vec.w;
        }

        // Load B linearly: Bs[k][n] = B[k, n]
        // B_VECS_PER_THREAD float4s/thread
        #pragma unroll
        for (int v = 0; v < B_VECS_PER_THREAD; ++v) {
            const int k = b_K_base + v * B_ROW_STRIDE;
            const float4 b_vec = *reinterpret_cast<const float4*>(
                B + (k_base + k) * N + (b_col * BN + b_N_grp * 4));
            *reinterpret_cast<float4*>(&Bs[k][b_N_grp * 4]) = b_vec;
        }

        __syncthreads(); // Ensure all threads have loaded their tiles into SMEM

        // Inner k-loop: accumulate the TM x TN tile into reg_c
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; ++k_inner) {
            // Load TM elements of A (TM/4 float4 reads) for this k into reg_a.
            #pragma unroll
            for (int vi = 0; vi < TM / 4; ++vi)
                *reinterpret_cast<float4*>(&reg_a[vi * 4]) =
                    *reinterpret_cast<const float4*>(&As[k_inner][t_row * TM + vi * 4]);

            // Load TN elements of B (TN/4 float4 reads) for this k into reg_b.
            #pragma unroll
            for (int vi = 0; vi < TN / 4; ++vi)
                *reinterpret_cast<float4*>(&reg_b[vi * 4]) =
                    *reinterpret_cast<const float4*>(&Bs[k_inner][t_col * TN + vi * 4]);

            // Compute outer prod and accumulate into reg_c
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads(); // Ensure all threads have finished computing before we write back to GMEM
    }

    // --- Handle output store for the thread's TM x TN tile in reg_c ---
    const int c_row = b_row * BM + t_row * TM; // Starting row index in C for this thread
    const int c_col = b_col * BN + t_col * TN; // Starting col

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int vi = 0; vi < TN / 4; ++vi)
            *reinterpret_cast<float4*>(&C[(c_row + i) * N + c_col + vi * 4]) =
                *reinterpret_cast<const float4*>(&reg_c[i][vi * 4]);
    }

}