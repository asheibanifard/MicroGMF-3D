"""
Adapter: run gsplat's CUDA rasterizer on our GaussianScene + Camera model,
using rasterize_mode="antialiased" which implements MIP splatting (adds an
isotropic 2D filter to the projected covariance, eliminating aliasing).
"""

from __future__ import annotations

import torch
from torch import Tensor

from gsplat.rendering import rasterization

from renderer import Camera, GaussianScene


def _build_viewmat(cam: Camera) -> Tensor:
    """Build a 4x4 world-to-camera matrix from our Camera dataclass."""
    E = torch.eye(4, device=cam.device, dtype=cam.R_wc.dtype)
    E[:3, :3] = cam.R_wc
    E[:3, 3] = cam.t_wc
    return E  # (4, 4)


def _build_K(cam: Camera, dtype=torch.float32) -> Tensor:
    """Build a 3x3 intrinsics matrix from our Camera dataclass."""
    K = torch.zeros(3, 3, device=cam.device, dtype=dtype)
    K[0, 0] = cam.fx
    K[1, 1] = cam.fy
    K[0, 2] = cam.cx
    K[1, 2] = cam.cy
    K[2, 2] = 1.0
    return K  # (3, 3)


def render_gsplat(
    scene: GaussianScene,
    cam: Camera,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,      # minimum 2D radius (pixels) — gsplat's MIP filter
    vol_shape: tuple = (100, 647, 813),  # (Z, Y, X) voxel counts of the source volume
) -> Tensor:
    """
    Render GaussianScene from Camera using gsplat's CUDA rasterizer with MIP splatting.

    vol_shape corrects the aspect ratio: the model normalises all axes to [-1,1]
    regardless of voxel count, so a (100,647,813) volume has a heavily stretched Z.
    We rescale means and scales by (vox_count / max_vox_count) per axis to undo this.

    Returns:
        image: (H, W) float tensor in [0, 1]
    """
    device = scene.device

    # --- Aspect-ratio correction (undo per-axis [-1,1] normalisation) ---
    Z, Y, X = vol_shape
    aspect = torch.tensor(
        [X / X, Y / X, Z / X],   # (x, y, z) scale factors — X is reference axis
        device=device, dtype=torch.float32,
    )

    # --- Parameter mapping ---
    means   = scene.means.float() * aspect[None, :]            # (N, 3) corrected
    quats   = torch.nn.functional.normalize(                   # (N, 4) wxyz
                  scene.quaternions.float(), dim=-1)
    scales  = torch.exp(scene.scales.float()) * aspect[None, :]  # (N, 3) log→linear, corrected

    # Amplitude → opacity.
    # Our intensity is the peak Gaussian response. We expose it as opacity so
    # gsplat alpha-composites correctly; color is kept white (= 1 channel at 1.0).
    amplitudes = torch.exp(scene.intensity.float().squeeze(-1))  # (N,)
    opacities  = amplitudes.clamp(0.0, 1.0)                      # (N,)
    colors     = torch.ones(scene.N, 1, device=device)           # (N, 1) grayscale=1

    # --- Camera ---
    viewmat = _build_viewmat(cam).unsqueeze(0)   # (1, 4, 4)
    K       = _build_K(cam).unsqueeze(0)         # (1, 3, 3)

    # --- Rasterize (MIP splatting = antialiased mode) ---
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=cam.width,
        height=cam.height,
        near_plane=near_plane,
        far_plane=far_plane,
        eps2d=eps2d,
        sh_degree=None,
        packed=True,
        rasterize_mode="antialiased",   # <-- MIP splatting
        render_mode="RGB",
    )

    # render_colors: (1, H, W, 1)  — single intensity channel
    image = render_colors[0, ..., 0]   # (H, W)
    return image
