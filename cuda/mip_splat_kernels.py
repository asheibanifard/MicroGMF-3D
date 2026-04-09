from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import mip_splat_cuda  # type: ignore[import-not-found]
    HAS_MIP_CUDA = True
except ImportError:
    HAS_MIP_CUDA = False


def _build_L_chol_from_params(log_scales: torch.Tensor, quaternions: torch.Tensor, eps: float = 1e-5):
    """Build Cholesky factors for anisotropic Gaussian covariance."""
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
    Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
    return torch.linalg.cholesky(Sigma_reg.float())


def mip_splat_forward(
    x_vals: torch.Tensor,
    y_vals: torch.Tensor,
    z_vals: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quaternions: torch.Tensor,
    log_amplitudes: torch.Tensor,
) -> torch.Tensor:
    """Run CUDA MIP splatting kernel. Inputs must be CUDA tensors."""
    if not HAS_MIP_CUDA:
        raise RuntimeError("mip_splat_cuda extension is not available")

    amps = torch.exp(log_amplitudes.clamp(-10.0, 6.0)).contiguous().float()
    L_chol = _build_L_chol_from_params(log_scales, quaternions).contiguous().float()

    return mip_splat_cuda.forward_cuda(
        x_vals.contiguous().float(),
        y_vals.contiguous().float(),
        z_vals.contiguous().float(),
        means.contiguous().float(),
        L_chol,
        amps,
    )
