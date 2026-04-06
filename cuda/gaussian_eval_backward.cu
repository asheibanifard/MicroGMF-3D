#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// OPTIMIZED Backward kernel: per-point with shared memory tiling
// and warp-level reduction for K-parameter gradients.
// ============================================================
//
// Key improvements:
//  1. One thread per point n, loops over K → eliminates atomicAdd on grad_x
//  2. Gaussian params loaded into shared memory tiles → reduces global reads
//  3. Warp-level __shfl_down_sync reduction → 32× fewer atomicAdds on K-params
//  4. Early skip for negligible Gaussian contributions
//
#define BACKWARD_TILE_K 32

__launch_bounds__(256, 4)
__global__ void gaussian_eval_backward_kernel(
    const float* __restrict__ grad_output,  // (N, K)
    const float* __restrict__ x,            // (N, 3)
    const float* __restrict__ means,        // (K, 3)
    const float* __restrict__ L_chol,       // (K, 3, 3)
    const float* __restrict__ amplitudes,   // (K,)
    const float* __restrict__ vals,         // (N, K)
    float* __restrict__ grad_x,             // (N, 3)
    float* __restrict__ grad_means,         // (K, 3)
    float* __restrict__ grad_amplitudes,    // (K,)
    float* __restrict__ grad_L,             // (K, 9)
    const int N,
    const int K
) {
    // Shared memory for Gaussian parameter tiles
    __shared__ float s_means[BACKWARD_TILE_K * 3];   // (TILE, 3)
    __shared__ float s_L[BACKWARD_TILE_K * 6];       // (TILE, 6) lower-tri: L00,L10,L11,L20,L21,L22
    __shared__ float s_amp[BACKWARD_TILE_K];          // (TILE,)

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const bool valid = (n < N);

    // Load point coords into registers
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (valid) {
        px = x[n * 3 + 0];
        py = x[n * 3 + 1];
        pz = x[n * 3 + 2];
    }

    // Register accumulators for grad_x — NO atomicAdd needed!
    float gx0 = 0.0f, gx1 = 0.0f, gx2 = 0.0f;

    // Process all K Gaussians in tiles
    for (int tile = 0; tile < K; tile += BACKWARD_TILE_K) {
        const int tile_size = min(BACKWARD_TILE_K, K - tile);

        // Cooperative load: threads < tile_size each load one Gaussian
        if (threadIdx.x < tile_size) {
            const int k = tile + threadIdx.x;
            s_means[threadIdx.x * 3 + 0] = means[k * 3 + 0];
            s_means[threadIdx.x * 3 + 1] = means[k * 3 + 1];
            s_means[threadIdx.x * 3 + 2] = means[k * 3 + 2];
            const float* Lk = &L_chol[k * 9];
            s_L[threadIdx.x * 6 + 0] = Lk[0];  // L00
            s_L[threadIdx.x * 6 + 1] = Lk[3];  // L10
            s_L[threadIdx.x * 6 + 2] = Lk[4];  // L11
            s_L[threadIdx.x * 6 + 3] = Lk[6];  // L20
            s_L[threadIdx.x * 6 + 4] = Lk[7];  // L21
            s_L[threadIdx.x * 6 + 5] = Lk[8];  // L22
            s_amp[threadIdx.x] = amplitudes[k];
        }
        __syncthreads();

        for (int tk = 0; tk < tile_size; tk++) {
            const int k = tile + tk;

            // Read grad_output and val for this (n,k)
            float go = 0.0f, val = 0.0f;
            if (valid) {
                go  = grad_output[n * K + k];
                val = vals[n * K + k];
            }

            // --- Compute per-thread gradient contributions ---
            float lgm0 = 0.0f, lgm1 = 0.0f, lgm2 = 0.0f;
            float lga = 0.0f;
            float lgL00 = 0.0f, lgL10 = 0.0f, lgL11 = 0.0f;
            float lgL20 = 0.0f, lgL21 = 0.0f, lgL22 = 0.0f;
            float lgx0 = 0.0f, lgx1 = 0.0f, lgx2 = 0.0f;

            // Early skip: if contribution is negligible, leave at 0
            if (valid && fabsf(go * val) > 1e-12f) {
                const float amp = s_amp[tk];

                // diff = x_n - mu_k (from shared memory)
                const float d0 = px - s_means[tk * 3 + 0];
                const float d1 = py - s_means[tk * 3 + 1];
                const float d2 = pz - s_means[tk * 3 + 2];

                const float L00 = s_L[tk * 6 + 0];
                const float L10 = s_L[tk * 6 + 1];
                const float L11 = s_L[tk * 6 + 2];
                const float L20 = s_L[tk * 6 + 3];
                const float L21 = s_L[tk * 6 + 4];
                const float L22 = s_L[tk * 6 + 5];

                // Forward sub: y = L^{-1} d
                const float y0 = d0 / L00;
                const float y1 = (d1 - L10 * y0) / L11;
                const float y2 = (d2 - L20 * y0 - L21 * y1) / L22;

                // grad through Mahalanobis
                const float gm = go * val * (-0.5f);
                const float gy0 = gm * 2.0f * y0;
                const float gy1 = gm * 2.0f * y1;
                const float gy2 = gm * 2.0f * y2;

                // Backward sub: gd = L^{-T} gy
                const float gd2 = gy2 / L22;
                const float gd1 = (gy1 - L21 * gd2) / L11;
                const float gd0 = (gy0 - L10 * gd1 - L20 * gd2) / L00;

                // grad_x contribution (accumulated in registers)
                lgx0 = gd0;  lgx1 = gd1;  lgx2 = gd2;

                // grad_means = -gd
                lgm0 = -gd0;  lgm1 = -gd1;  lgm2 = -gd2;

                // grad_amplitudes
                lga = (amp > 1e-12f) ? (go * val / amp) : 0.0f;

                // grad_L_chol: grad_L_{ij} = -gd_i * y_j  (j<=i)
                lgL00 = -gd0 * y0;
                lgL10 = -gd1 * y0;  lgL11 = -gd1 * y1;
                lgL20 = -gd2 * y0;  lgL21 = -gd2 * y1;  lgL22 = -gd2 * y2;
            }

            // Accumulate grad_x directly (no atomic!)
            gx0 += lgx0;  gx1 += lgx1;  gx2 += lgx2;

            // --- Warp-level reduction for K-parameter gradients ---
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lgm0  += __shfl_down_sync(0xffffffff, lgm0,  off);
                lgm1  += __shfl_down_sync(0xffffffff, lgm1,  off);
                lgm2  += __shfl_down_sync(0xffffffff, lgm2,  off);
                lga   += __shfl_down_sync(0xffffffff, lga,   off);
                lgL00 += __shfl_down_sync(0xffffffff, lgL00, off);
                lgL10 += __shfl_down_sync(0xffffffff, lgL10, off);
                lgL11 += __shfl_down_sync(0xffffffff, lgL11, off);
                lgL20 += __shfl_down_sync(0xffffffff, lgL20, off);
                lgL21 += __shfl_down_sync(0xffffffff, lgL21, off);
                lgL22 += __shfl_down_sync(0xffffffff, lgL22, off);
            }

            // Lane 0 of each warp atomicAdds the warp sum (32× fewer atomics)
            if (lane == 0) {
                atomicAdd(&grad_means[k * 3 + 0], lgm0);
                atomicAdd(&grad_means[k * 3 + 1], lgm1);
                atomicAdd(&grad_means[k * 3 + 2], lgm2);
                atomicAdd(&grad_amplitudes[k], lga);
                float* gLk = &grad_L[k * 9];
                atomicAdd(&gLk[0], lgL00);
                atomicAdd(&gLk[3], lgL10);
                atomicAdd(&gLk[4], lgL11);
                atomicAdd(&gLk[6], lgL20);
                atomicAdd(&gLk[7], lgL21);
                atomicAdd(&gLk[8], lgL22);
            }
        }
        __syncthreads();
    }

    // Write grad_x directly (no atomicAdd!)
    if (valid) {
        grad_x[n * 3 + 0] = gx0;
        grad_x[n * 3 + 1] = gx1;
        grad_x[n * 3 + 2] = gx2;
    }
}


// ============================================================
// C++ backward interface
// ============================================================
std::vector<torch::Tensor> gaussian_eval_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes,
    torch::Tensor vals
) {
    const int N = x.size(0);
    const int K = means.size(0);

    auto grad_x = torch::zeros_like(x);
    auto grad_means = torch::zeros_like(means);
    auto grad_amplitudes = torch::zeros_like(amplitudes);
    auto grad_L = torch::zeros({K, 9}, x.options());  // (K, 3, 3) flattened

    // Per-point launch: N threads (not N*K), each loops over K
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int smem = BACKWARD_TILE_K * (3 + 6 + 1) * sizeof(float);

    gaussian_eval_backward_kernel<<<blocks, threads, smem>>>(
        grad_output.data_ptr<float>(),
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        vals.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_means.data_ptr<float>(),
        grad_amplitudes.data_ptr<float>(),
        grad_L.data_ptr<float>(),
        N, K
    );

    return {grad_x, grad_means, grad_L.reshape({K, 3, 3}), grad_amplitudes};
}


// ============================================================
// Module binding
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_cuda", &gaussian_eval_backward_cuda, "Backward pass (CUDA)");
}
