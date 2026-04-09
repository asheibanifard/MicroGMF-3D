#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// CUDA MIP splatting for Gaussian field
// -----------------------------------------------------------------------------
// Output per pixel:
//   mip(y, x) = max_z sum_k a_k * exp(-0.5 * ||L_k^{-1}(p - mu_k)||^2)
// where p = [x, y, z].
//
// Thread mapping: one thread per pixel.
// This is a forward-only renderer for realtime inference/visualization.
// -----------------------------------------------------------------------------

__global__ void mip_splat_kernel(
    const float* __restrict__ x_vals,      // (W,)
    const float* __restrict__ y_vals,      // (H,)
    const float* __restrict__ z_vals,      // (Z,)
    const float* __restrict__ means,       // (K, 3)
    const float* __restrict__ L_chol,      // (K, 3, 3) lower-triangular
    const float* __restrict__ amplitudes,  // (K,)
    float* __restrict__ out_mip,           // (H, W)
    const int H,
    const int W,
    const int Z,
    const int K
) {
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    const int P = H * W;
    if (pix >= P) return;

    const int yi = pix / W;
    const int xi = pix % W;

    const float x = x_vals[xi];
    const float y = y_vals[yi];

    float best = -INFINITY;

    for (int zi = 0; zi < Z; ++zi) {
        const float z = z_vals[zi];
        float density = 0.0f;

        // Sum all Gaussians at this sample point.
        for (int k = 0; k < K; ++k) {
            const float* mu = &means[k * 3];
            const float* L = &L_chol[k * 9];

            const float d0 = x - mu[0];
            const float d1 = y - mu[1];
            const float d2 = z - mu[2];

            // Solve L yv = d using forward substitution.
            const float y0 = d0 / L[0];
            const float y1 = (d1 - L[3] * y0) / L[4];
            const float y2 = (d2 - L[6] * y0 - L[7] * y1) / L[8];

            const float mahal = y0 * y0 + y1 * y1 + y2 * y2;
            density += amplitudes[k] * __expf(-0.5f * mahal);
        }

        best = fmaxf(best, density);
    }

    out_mip[pix] = best;
}


torch::Tensor mip_splat_forward_cuda(
    torch::Tensor x_vals,
    torch::Tensor y_vals,
    torch::Tensor z_vals,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes
) {
    const int W = static_cast<int>(x_vals.size(0));
    const int H = static_cast<int>(y_vals.size(0));
    const int Z = static_cast<int>(z_vals.size(0));
    const int K = static_cast<int>(means.size(0));

    auto out = torch::empty({H, W}, x_vals.options());

    const int threads = 256;
    const int blocks = (H * W + threads - 1) / threads;

    mip_splat_kernel<<<blocks, threads>>>(
        x_vals.data_ptr<float>(),
        y_vals.data_ptr<float>(),
        z_vals.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        out.data_ptr<float>(),
        H,
        W,
        Z,
        K
    );

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &mip_splat_forward_cuda, "MIP splatting forward (CUDA)");
}
