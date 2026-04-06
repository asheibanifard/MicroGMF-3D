from __future__ import annotations

# ---------------------------------------------------------------------------
# Custom autograd function wrapping the separate CUDA kernels
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F

try:
    import gaussian_eval_cuda_forward
    import gaussian_eval_cuda_backward
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


def _build_L_chol(log_scales: torch.Tensor, quaternions: torch.Tensor, eps: float = 1e-5):
    """
    Compute Cholesky factor from learnable (log_scales, quaternions).
    This function is differentiable through PyTorch autograd.

    Returns L such that  L Lᵀ = R diag(s²) Rᵀ + εI.
    """
    K = log_scales.shape[0]
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(K, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    S2 = torch.diag_embed(scales ** 2)
    Sigma = R @ S2 @ R.transpose(-2, -1)
    Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device).unsqueeze(0)
    return torch.linalg.cholesky(Sigma_reg.float())


class _GaussianEvalCUDA(torch.autograd.Function):
    """
    Wraps the separate CUDA forward and backward kernels.

    The kernels operate on (x, means, L_chol, amplitudes) and produce (N,K).
    The backward kernel returns grad_x, grad_means, grad_amplitudes —
    but NOT grad_L_chol.

    To propagate gradients to log_scales and quaternions we:
      1) Compute grad_L_chol analytically from the Mahalanobis distance
      2) Recompute L_chol from (log_scales, quaternions) inside backward
         with autograd enabled, then call torch.autograd.grad to chain
         grad_L_chol → grad_log_scales, grad_quaternions.
    """

    @staticmethod
    def forward(ctx, x, means, log_scales, quaternions, log_amplitudes, L_chol_detached):
        """
        Args:
            x:                (N, 3) query points
            means:            (K, 3) Gaussian centres
            log_scales:       (K, 3) learnable log-scales
            quaternions:      (K, 4) learnable quaternions
            log_amplitudes:   (K,) learnable log-amplitudes
            L_chol_detached:  (K, 3, 3) precomputed Cholesky factor (detached)
        """
        amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

        # CUDA forward: returns (N, K)
        vals_nk = gaussian_eval_cuda_forward.forward_cuda(
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol_detached.contiguous().float(),
            amplitudes.detach().contiguous().float(),
        )

        # Sum over K → (N,)
        output = vals_nk.sum(dim=1)

        ctx.save_for_backward(
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol_detached, amplitudes, vals_nk,
        )
        return output.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output_n):
        (
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol, amplitudes, vals_nk,
        ) = ctx.saved_tensors
        N, K = vals_nk.shape

        # Expand (N,) → (N, K) since output = vals_nk.sum(dim=1)
        grad_nk = grad_output_n[:, None].expand(N, K).contiguous()

        # CUDA backward: returns [grad_x, grad_means, grad_L_chol, grad_amplitudes]
        # The kernel already computes grad_L_chol via analytic differentiation
        # of the forward substitution, accumulated over all N points per Gaussian.
        cuda_grads = gaussian_eval_cuda_backward.backward_cuda(
            grad_nk.float(),
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol.contiguous().float(),
            amplitudes.detach().contiguous().float(),
            vals_nk.contiguous().float(),
        )
        grad_x = cuda_grads[0]               # (N, 3)
        grad_means = cuda_grads[1]            # (K, 3)
        grad_L_chol = cuda_grads[2]           # (K, 3, 3) — from CUDA kernel
        grad_amplitudes_raw = cuda_grads[3]   # (K,)

        # Ensure only lower-triangular entries are used
        grad_L_chol = torch.tril(grad_L_chol)

        # --- Chain grad_L_chol → grad_log_scales, grad_quaternions ---
        # Recompute L_chol from (log_scales, quaternions) WITH autograd,
        # then use torch.autograd.grad to propagate.
        with torch.enable_grad():
            ls = log_scales.detach().requires_grad_(True)
            qt = quaternions.detach().requires_grad_(True)
            L_recomp = _build_L_chol(ls, qt)
            grads = torch.autograd.grad(
                L_recomp, [ls, qt],
                grad_outputs=grad_L_chol,
                allow_unused=True,
            )
            grad_log_scales = grads[0]
            grad_quaternions = grads[1]

        # grad_log_amplitudes: chain rule through amplitudes = exp(log_amp_clamped)
        grad_log_amplitudes = grad_amplitudes_raw * amplitudes

        # Return grads for: x, means, log_scales, quaternions, log_amplitudes, L_chol
        return (
            grad_x.to(x.dtype),
            grad_means.to(means.dtype),
            grad_log_scales,
            grad_quaternions,
            grad_log_amplitudes.to(log_amplitudes.dtype),
            None,  # L_chol_detached — no grad needed
        )


_gaussian_eval_cuda_fn = _GaussianEvalCUDA.apply
