from __future__ import annotations

import torch
import torch.nn.functional as F
from model import GaussianMixtureField
from regularisers import tubular_regulariser, cross_section_symmetry_reg

def loss_volume(
    field: GaussianMixtureField,
    x: torch.Tensor,
    v: torch.Tensor,
    # neighbour data (optional — pass None to skip gradient loss)
    x_dx: torch.Tensor | None = None,
    v_dx: torch.Tensor | None = None,
    x_dy: torch.Tensor | None = None,
    v_dy: torch.Tensor | None = None,
    x_dz: torch.Tensor | None = None,
    v_dz: torch.Tensor | None = None,
    *,
    w_grad: float = 0.3,
    w_tube: float = 1e-4,
    w_cross: float = 1e-4,
    w_scale: float = 5e-4,
    scale_target: float | None = 0.03,
) -> tuple[torch.Tensor, dict]:

    pred_raw = field(x)
    gt_target = v
    pred_cmp = pred_raw
    l_rec = F.mse_loss(pred_cmp, gt_target)

    # --- gradient supervision (finite differences) ---
    l_grad = torch.zeros((), device=x.device)
    if x_dx is not None and w_grad > 0:
        # Match signed gradients (preserves edge direction).
        p_dx = field(x_dx)
        p_dy = field(x_dy)
        p_dz = field(x_dz)

        c = pred_cmp
        c_dx = p_dx
        c_dy = p_dy
        c_dz = p_dz

        gv = v
        gv_dx = v_dx
        gv_dy = v_dy
        gv_dz = v_dz
        l_grad = (
            F.l1_loss(c_dx - c, gv_dx - gv)  # signed gradient in x
            + F.l1_loss(c_dy - c, gv_dy - gv)  # signed gradient in y
            + F.l1_loss(c_dz - c, gv_dz - gv)  # signed gradient in z
        )

    # --- covariance regularisers ---
    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regulariser(Sigma)
    l_csym = cross_section_symmetry_reg(Sigma)

    # --- scale regulariser (prevent blobs) ---
    scales = torch.exp(field.log_scales).clamp(1e-6, 1e2)
    if scale_target is not None:
        l_scale = F.relu(scales - scale_target).mean()
    else:
        l_scale = scales.mean()

    total = (
        l_rec
        + w_grad * l_grad
        + w_tube * l_tube
        + w_cross * l_csym
        + w_scale * l_scale
    )
    parts = {
        "rec": l_rec,
        "grad": l_grad,
        "tube": l_tube,
        "csym": l_csym,
        "scale": l_scale,
    }
    return total, parts

def render_soft_mip_z(
    field: GaussianMixtureField,
    xy: torch.Tensor,
    n_z: int,
    tau: float,
    pt_chunk: int = 4096,
) -> torch.Tensor:
    """Soft z-MIP via LogSumExp along z-rays."""
    device = xy.device
    P = xy.shape[0]
    z_vals = torch.linspace(-1, 1, n_z, device=device, dtype=xy.dtype)

    pts = torch.cat(
        [
            xy[:, None, :].expand(P, n_z, 2),
            z_vals[None, :, None].expand(P, n_z, 1),
        ],
        dim=-1,
    ).reshape(-1, 3)  # (P*n_z, 3)

    vals = []
    for i in range(0, pts.shape[0], pt_chunk):
        vals.append(field(pts[i : i + pt_chunk]))
    v = torch.cat(vals).reshape(P, n_z)

    tau_safe = max(tau, 1e-6)
    return tau_safe * torch.logsumexp(v / tau_safe, dim=1)


def loss_mip(
    field: GaussianMixtureField,
    xy: torch.Tensor,
    mip_gt: torch.Tensor,
    n_z: int,
    tau: float,
    *,
    w_tube: float = 1e-4,
    w_cross: float = 1e-4,
    mip_batch: int = 512,
) -> tuple[torch.Tensor, dict]:
    P = xy.shape[0]
    if P <= mip_batch:
        pred = render_soft_mip_z(field, xy, n_z, tau)
    else:
        chunks = []
        for i in range(0, P, mip_batch):
            chunks.append(render_soft_mip_z(field, xy[i : i + mip_batch], n_z, tau))
        pred = torch.cat(chunks)

    gt_mip = mip_gt
    pred_mip = pred
    l_img = F.l1_loss(pred_mip, gt_mip)

    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regulariser(Sigma)
    l_csym = cross_section_symmetry_reg(Sigma)

    total = l_img + w_tube * l_tube + w_cross * l_csym
    return total, {"mip": l_img, "tube": l_tube, "csym": l_csym}
