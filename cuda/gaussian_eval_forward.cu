#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// Forward kernel: evaluate all Gaussians at all points
// ============================================================
// Each thread handles one (n, k) pair.
// Fused kernel avoids materialising large intermediate tensors.
__global__ void gaussian_eval_forward_kernel(
    const float* __restrict__ x,           // (N, 3)
    const float* __restrict__ means,       // (K, 3)
    const float* __restrict__ L_chol,      // (K, 3, 3) Cholesky factors (lower triangular)
    const float* __restrict__ amplitudes,  // (K,)
    float* __restrict__ output,            // (N, K)
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int n = idx / K;
    const int k = idx % K;

    // diff = x_n - mu_k
    const float d0 = x[n * 3 + 0] - means[k * 3 + 0];
    const float d1 = x[n * 3 + 1] - means[k * 3 + 1];
    const float d2 = x[n * 3 + 2] - means[k * 3 + 2];

    // L = L_chol[k] (3x3 lower triangular, row-major)
    const float* L = &L_chol[k * 9];

    // Forward substitution: solve L y = d
    const float y0 = d0 / L[0];
    const float y1 = (d1 - L[3] * y0) / L[4];
    const float y2 = (d2 - L[6] * y0 - L[7] * y1) / L[8];

    // Mahalanobis distance
    const float mahal = y0 * y0 + y1 * y1 + y2 * y2;

    output[idx] = amplitudes[k] * expf(-0.5f * mahal);
}


// ============================================================
// C++ forward interface
// ============================================================
torch::Tensor gaussian_eval_forward_cuda(
    torch::Tensor x,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes
) {
    const int N = x.size(0);
    const int K = means.size(0);

    auto output = torch::zeros({N, K}, x.options());

    const int threads = 256;
    const int blocks = (N * K + threads - 1) / threads;

    gaussian_eval_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        output.data_ptr<float>(),
        N, K
    );

    return output;
}


// ============================================================
// Module binding
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &gaussian_eval_forward_cuda, "Forward pass (CUDA)");
}
