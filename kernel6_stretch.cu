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

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__
void sgemm_vectorize(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {

    const int b_row = blockIdx.y; // BM-row tile index
    const int b_col = blockIdx.x; // BN-col tile index

    const int t_row = threadIdx.x / (BN / TN); // [0..15] m-reg-tile row
    const int t_col = threadIdx.x % (BN / TN); // [0..15] n-reg-tile col

    // SMEM for A and B tiles
    __shared__ float As[BK][BM]; // Transposed -> BK rows, BM cols
    __shared__ float Bs[BK][BN]; // BK rows, BN cols

    // Reg tiles for C
    float reg_c[TM][TN] = {0}; // 8x8 tile of C in registers, zero-init
    float __align__(16) reg_a[TM]; // 8 elements of A in registers, 16-byte aligned for float4 loads
    float __align__(16) reg_b[TN]; // 8 elements of B in registers, 16-byte aligned for float4 loads

    // Compute the thread's starting indices
    const int a_vec_idx = threadIdx.x; // [0..255] Each thread loads one float4 (4 floats)
    const int a_smem_row = a_vec_idx / (BM / 4); // SMEM row index for A [0..7]
    const int a_smem_col = a_vec_idx % (BM / 4); // SMEM col index for A [0..31]

    const int b_vec_idx = threadIdx.x; // [0..255] Each thread loads one float4 (4 floats)
    const int b_smem_row = b_vec_idx / (BN / 4); // [0..7] SMEM row index for B
    const int b_smem_col = b_vec_idx % (BN / 4); // [0..31] SMEM col index for B

    // Load A into SMEM.  Each thread loads one float4 (4 floats).
    for (int k_base = 0; k_base < K; k_base += BK) {
        // Load A tile into SMEM A[m, k] w/ offset m*K + k
        // m = b_row * BM + a_smem_col * 4 + [0..3]
        const float* a_ptr = A + ((b_row * BM + a_smem_col * 4) * K) + (k_base + a_smem_row);

        // We want to load A transposed so the BK direction is the 'row' of SMEM and BM is the 'column'.
        // This will lead to contiguous GMEM reads by iterating the float4 load over the K dimension
        const int a_M_idx = a_vec_idx / (BK / 4); // [0..127] m-offset in tile
        const int a_K_grp = a_vec_idx % (BK / 4); // [0..1] k-offset group

        const float4 a_vec = *reinterpret_cast<const float4*>(
            A + (b_row * BM + a_M_idx) * K + (k_base + a_K_grp * 4)); // Load 4 floats as float4
        
        // Scatter the loaded float4 into SMEM transposed: As[k, m] = A[m, k]
        As[a_K_grp * 4 + 0][a_M_idx] = a_vec.x;
        As[a_K_grp * 4 + 1][a_M_idx] = a_vec.y;
        As[a_K_grp * 4 + 2][a_M_idx] = a_vec.z;
        As[a_K_grp * 4 + 3][a_M_idx] = a_vec.w;

        // Load B tile into SMEM B[k, n] w/ offset k*N + n
        const int b_K_idx = b_vec_idx / (BN / 4); // [0..7] k-offset in tile
        const int b_N_grp = b_vec_idx % (BN / 4); // [0..31] n-group of 4

        const float4 b_vec = *reinterpret_cast<const float4*>(
            B + (k_base + b_K_idx) * N + (b_col * BN + b_N_grp * 4)); // Load 4 floats as float4

        // Bs[k][n] is alr contig in SMEM, so we can store the loaded float4 directly without scattering
        *reinterpret_cast<float4*>(&Bs[b_K_idx][b_N_grp * 4]) = b_vec; // Store B tile linearly in SMEM


        __syncthreads(); // Ensure all threads have loaded their tiles into SMEM

        // Inner k-loop: accumulate the 8x8 tile into reg_c
        #pragma unroll // Unroll the inner loop for better performance, since BK=8 is small and known at compile time
        for (int k_inner = 0; k_inner < BK; ++k_inner) {
            // Load TM=8 elements of A (two float4 reads) for this k into reg_a. 
            *reinterpret_cast<float4*>(&reg_a[0]) = *reinterpret_cast<const float4*>(&As[k_inner][t_row * TM]); // Load 8 floats as two float4s
            *reinterpret_cast<float4*>(&reg_a[4]) = *reinterpret_cast<const float4*>(&As[k_inner][t_row * TM + 4]);

            // Load TN=8 elements of B (two float4 reads) for this k into reg_b.
            *reinterpret_cast<float4*>(&reg_b[0]) = *reinterpret_cast<const float4*>(&Bs[k_inner][t_col * TN]); // Load 8 floats as two float4s
            *reinterpret_cast<float4*>(&reg_b[4]) = *reinterpret_cast<const float4*>(&Bs[k_inner][t_col * TN + 4]);

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

    // --- Handle output store for the thread's 8x8 tile in reg_c ---
    const int c_row = b_row * BM + t_row * TM; // Starting row index in C for this thread
    const int c_col = b_col * BN + t_col * TN; // Starting col

    #pragma unroll
    for (int i = 0; i < TM; ++i) {

        float4 lo = {reg_c[i][0], reg_c[i][1], reg_c[i][2], reg_c[i][3]}; // First 4 elements of the row
        float4 hi = {reg_c[i][4], reg_c[i][5], reg_c[i][6], reg_c[i][7]}; // Last 4 elements of the row

        *reinterpret_cast<float4*>(&C[(c_row + i) * N + c_col]) = lo; // Store first 4 elements of the row
        *reinterpret_cast<float4*>(&C[(c_row + i) * N + c_col + 4]) = hi; // Store last 4 elements of the row
    }

}