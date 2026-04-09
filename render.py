#!/usr/bin/env python3
"""
Render a reconstructed GaussianMixtureField volume using orthographic MIP splatting.

This version fixes several issues in the older script:
1. Rotation is handled correctly in the rendering space rather than by rendering first
   and reshaping later.
2. Output image size is derived from the rotated volume extents.
3. CUDA and PyTorch renderers share the same camera-space rendering plan.
4. The code path is cleaner and easier to debug.

Coordinate convention:
- Model/world axes: X, Y, Z
- Volume shape is given as (Z, Y, X)
- Rendering is orthographic.
- We render in "camera space" after rotating the object into that space.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile

from model import GaussianMixtureField
from utils import load_config

try:
    from cuda.mip_splat_kernels import HAS_MIP_CUDA, mip_splat_forward
except ImportError:
    HAS_MIP_CUDA = False
    mip_splat_forward = None


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def _compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    return float(ssim(gt, pred, data_range=1.0))


def _compute_lpips(gt: np.ndarray, pred: np.ndarray, device: str) -> float | None:
    try:
        import lpips  # type: ignore
    except ImportError:
        return None

    gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    gt_t = gt_t * 2.0 - 1.0
    pred_t = pred_t * 2.0 - 1.0

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    with torch.no_grad():
        score = loss_fn(gt_t.to(device), pred_t.to(device))
    return float(score.item())


def _compute_metrics(gt_mip: np.ndarray, pred_mip: np.ndarray, device: str) -> dict[str, float | None]:
    gt_vis = _normalise_for_display(gt_mip)
    pred_vis = _normalise_for_display(pred_mip)
    return {
        "psnr": _compute_psnr(gt_vis, pred_vis, data_range=1.0),
        "ssim": _compute_ssim(gt_vis, pred_vis),
        "lpips": _compute_lpips(gt_vis, pred_vis, device=device),
    }


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _normalise_for_display(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    lo = float(img.min())
    hi = float(np.percentile(img, percentile))
    if hi <= lo + 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _load_gt_mip_if_available(cfg: dict[str, Any]) -> np.ndarray | None:
    tif_path = cfg.get("data", {}).get("tif_path")
    if not tif_path or not os.path.exists(tif_path):
        return None

    gt = tifffile.imread(tif_path).astype(np.float32)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    return gt.max(axis=0)


def _save_visualization(
    mip_pred: np.ndarray,
    output_path: str,
    gt_mip: np.ndarray | None,
    metrics: dict[str, float | None] | None = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pred_vis = _normalise_for_display(mip_pred)

    if gt_mip is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=120)
        ax.imshow(pred_vis, cmap="gray", origin="lower")
        ax.set_title("Reconstructed MIP")
        ax.axis("off")
    else:
        gt_vis = _normalise_for_display(gt_mip)
        diff = np.abs(gt_vis - pred_vis)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

        axes[0].imshow(gt_vis, cmap="gray", origin="lower")
        axes[0].set_title("Ground Truth MIP")
        axes[0].axis("off")

        axes[1].imshow(pred_vis, cmap="gray", origin="lower")
        axes[1].set_title("Reconstructed MIP")
        axes[1].axis("off")

        if metrics is not None:
            p = metrics.get("psnr")
            s = metrics.get("ssim")
            l = metrics.get("lpips")
            text_lines = [
                f"PSNR: {p:.3f} dB" if p is not None and np.isfinite(p) else "PSNR: inf",
                f"SSIM: {s:.4f}" if s is not None else "SSIM: unavailable",
                f"LPIPS: {l:.4f}" if l is not None else "LPIPS: unavailable",
            ]
            axes[1].text(
                0.02,
                0.02,
                "\n".join(text_lines),
                transform=axes[1].transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                color="white",
                bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.6},
            )

        im = axes[2].imshow(diff, cmap="inferno", origin="lower")
        axes[2].set_title("Absolute Difference")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Model/config loading
# -----------------------------------------------------------------------------

def _load_model(ckpt_path: str, config_path: str, device: str) -> tuple[GaussianMixtureField, dict[str, Any]]:
    state = torch.load(ckpt_path, weights_only=True, map_location=device)
    num_gaussians = int(state["means"].shape[0])

    cfg = load_config(config_path)
    bounds = cfg.get("model", {}).get("bounds", [[-1, 1], [-1, 1], [-1, 1]])

    model = GaussianMixtureField(
        num_gaussians=num_gaussians,
        bounds=bounds,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _infer_volume_shape(cfg: dict[str, Any], explicit_shape: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if explicit_shape is not None:
        return explicit_shape

    tif_path = cfg.get("data", {}).get("tif_path")
    if tif_path and os.path.exists(tif_path):
        vol = tifffile.imread(tif_path)
        return tuple(int(v) for v in vol.shape)

    return (100, 647, 813)


def _normalize_bounds_to_volume_aspect_ratio(
    bounds: list[list[float]],
    shape_zyx: tuple[int, int, int],
) -> list[list[float]]:
    z_size, y_size, x_size = shape_zyx
    max_size = max(x_size, y_size, z_size)

    x_scale = x_size / max_size
    y_scale = y_size / max_size
    z_scale = z_size / max_size

    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    z_range = bounds[2][1] - bounds[2][0]

    x_center = (bounds[0][0] + bounds[0][1]) / 2.0
    y_center = (bounds[1][0] + bounds[1][1]) / 2.0
    z_center = (bounds[2][0] + bounds[2][1]) / 2.0

    return [
        [x_center - x_range * x_scale / 2.0, x_center + x_range * x_scale / 2.0],
        [y_center - y_range * y_scale / 2.0, y_center + y_range * y_scale / 2.0],
        [z_center - z_range * z_scale / 2.0, z_center + z_range * z_scale / 2.0],
    ]


# -----------------------------------------------------------------------------
# Rotation / quaternion helpers
# -----------------------------------------------------------------------------

def euler_to_rotation_matrix(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # yaw=z, pitch=y, roll=x
    Rz = torch.tensor(
        [[cy, -sy, 0.0],
         [sy,  cy, 0.0],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )

    Ry = torch.tensor(
        [[ cp, 0.0, sp],
         [0.0, 1.0, 0.0],
         [-sp, 0.0, cp]],
        device=device,
        dtype=dtype,
    )

    Rx = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0,  cr, -sr],
         [0.0,  sr,  cr]],
        device=device,
        dtype=dtype,
    )

    return Rz @ Ry @ Rx


def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    R = R.to(torch.float32)
    trace = float((R[0, 0] + R[1, 1] + R[2, 2]).item())

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = float((R[2, 1] - R[1, 2]).item()) / s
        y = float((R[0, 2] - R[2, 0]).item()) / s
        z = float((R[1, 0] - R[0, 1]).item()) / s
    elif float(R[0, 0].item()) > float(R[1, 1].item()) and float(R[0, 0].item()) > float(R[2, 2].item()):
        s = math.sqrt(1.0 + float(R[0, 0].item()) - float(R[1, 1].item()) - float(R[2, 2].item())) * 2.0
        w = float((R[2, 1] - R[1, 2]).item()) / s
        x = 0.25 * s
        y = float((R[0, 1] + R[1, 0]).item()) / s
        z = float((R[0, 2] + R[2, 0]).item()) / s
    elif float(R[1, 1].item()) > float(R[2, 2].item()):
        s = math.sqrt(1.0 + float(R[1, 1].item()) - float(R[0, 0].item()) - float(R[2, 2].item())) * 2.0
        w = float((R[0, 2] - R[2, 0]).item()) / s
        x = float((R[0, 1] + R[1, 0]).item()) / s
        y = 0.25 * s
        z = float((R[1, 2] + R[2, 1]).item()) / s
    else:
        s = math.sqrt(1.0 + float(R[2, 2].item()) - float(R[0, 0].item()) - float(R[1, 1].item())) * 2.0
        w = float((R[1, 0] - R[0, 1]).item()) / s
        x = float((R[0, 2] + R[2, 0]).item()) / s
        y = float((R[1, 2] + R[2, 1]).item()) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], device=R.device, dtype=R.dtype)
    q = q / (torch.norm(q) + 1e-12)
    return q


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply quaternions in [w, x, y, z] format.
    Supports broadcasting.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    out = torch.stack([w, x, y, z], dim=-1)
    out = out / (torch.linalg.norm(out, dim=-1, keepdim=True) + 1e-12)
    return out


# -----------------------------------------------------------------------------
# Render planning
# -----------------------------------------------------------------------------

@dataclass
class RenderPlan:
    rotation_matrix: torch.Tensor
    world_center: torch.Tensor
    x_vals: torch.Tensor
    y_vals: torch.Tensor
    z_vals: torch.Tensor
    out_h: int
    out_w: int
    depth_samples: int


def _compute_axis_step(axis_dir: torch.Tensor, world_lengths_xyz: torch.Tensor, shape_zyx: tuple[int, int, int]) -> float:
    """
    Estimate an effective sampling step along an arbitrary camera-space axis.

    world_lengths_xyz = [Lx, Ly, Lz]
    shape_zyx = (Z, Y, X)
    """
    z_size, y_size, x_size = shape_zyx
    voxel_steps_xyz = torch.tensor(
        [
            float(world_lengths_xyz[0]) / max(x_size - 1, 1),
            float(world_lengths_xyz[1]) / max(y_size - 1, 1),
            float(world_lengths_xyz[2]) / max(z_size - 1, 1),
        ],
        dtype=axis_dir.dtype,
        device=axis_dir.device,
    )

    # Weighted average projected step
    step = torch.sum(torch.abs(axis_dir) * voxel_steps_xyz)
    return float(step.item())


def _make_render_plan(
    bounds: list[list[float]],
    shape_zyx: tuple[int, int, int],
    rotation_matrix: torch.Tensor,
    screen_size: int = 0,
    device: str = "cpu",
) -> RenderPlan:
    """
    Build a camera-space orthographic rendering plan.

    We rotate the object into camera space. Then:
    - x_vals, y_vals define the screen plane extents
    - z_vals defines the MIP depth traversal
    """
    x_lo, x_hi = float(bounds[0][0]), float(bounds[0][1])
    y_lo, y_hi = float(bounds[1][0]), float(bounds[1][1])
    z_lo, z_hi = float(bounds[2][0]), float(bounds[2][1])

    center = torch.tensor(
        [(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0, (z_lo + z_hi) / 2.0],
        device=device,
        dtype=torch.float32,
    )

    half_extents = torch.tensor(
        [(x_hi - x_lo) / 2.0, (y_hi - y_lo) / 2.0, (z_hi - z_lo) / 2.0],
        device=device,
        dtype=torch.float32,
    )
    world_lengths = 2.0 * half_extents

    # In camera space, the object is rotated by R.
    # Camera-space extents of the rotated AABB are:
    # new_half = |R| @ old_half
    new_half = torch.abs(rotation_matrix) @ half_extents

    x_extent = float((2.0 * new_half[0]).item())
    y_extent = float((2.0 * new_half[1]).item())
    z_extent = float((2.0 * new_half[2]).item())

    cam_x = rotation_matrix[0, :]
    cam_y = rotation_matrix[1, :]
    cam_z = rotation_matrix[2, :]

    step_x = _compute_axis_step(cam_x, world_lengths, shape_zyx)
    step_y = _compute_axis_step(cam_y, world_lengths, shape_zyx)
    step_z = _compute_axis_step(cam_z, world_lengths, shape_zyx)

    out_w_native = max(1, int(round(x_extent / max(step_x, 1e-8))))
    out_h_native = max(1, int(round(y_extent / max(step_y, 1e-8))))
    depth_native = max(1, int(round(z_extent / max(step_z, 1e-8))))

    if screen_size > 0:
        scale = screen_size / max(out_h_native, out_w_native)
        out_h = max(1, int(round(out_h_native * scale)))
        out_w = max(1, int(round(out_w_native * scale)))
    else:
        out_h = out_h_native
        out_w = out_w_native

    depth_samples = depth_native

    x_vals = torch.linspace(-x_extent / 2.0, x_extent / 2.0, out_w, device=device)
    y_vals = torch.linspace(-y_extent / 2.0, y_extent / 2.0, out_h, device=device)
    z_vals = torch.linspace(-z_extent / 2.0, z_extent / 2.0, depth_samples, device=device)

    return RenderPlan(
        rotation_matrix=rotation_matrix,
        world_center=center,
        x_vals=x_vals,
        y_vals=y_vals,
        z_vals=z_vals,
        out_h=out_h,
        out_w=out_w,
        depth_samples=depth_samples,
    )


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def _compute_splat_mip_torch(
    model: GaussianMixtureField,
    plan: RenderPlan,
    xy_batch_size: int,
    k_chunk: int,
    device: str,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(plan.y_vals, plan.x_vals, indexing="ij")
    yy_flat = yy.reshape(-1)
    xx_flat = xx.reshape(-1)

    mip = torch.full((plan.out_h * plan.out_w,), -torch.inf, device=device)

    # To evaluate the model in world coordinates, map camera-space points back:
    # p_cam = R (p_world - c)
    # => p_world = R^T p_cam + c
    R_inv = plan.rotation_matrix.T
    center = plan.world_center

    with torch.no_grad():
        for zi, z_val in enumerate(plan.z_vals, start=1):
            for i in range(0, yy_flat.numel(), xy_batch_size):
                j = min(i + xy_batch_size, yy_flat.numel())

                bx = xx_flat[i:j]
                by = yy_flat[i:j]
                bz = torch.full_like(bx, z_val)

                coords_cam = torch.stack([bx, by, bz], dim=-1)
                coords_world = coords_cam @ R_inv.T + center

                vals = model(coords_world, k_chunk=k_chunk)
                mip[i:j] = torch.maximum(mip[i:j], vals)

            if zi % max(1, plan.depth_samples // 10) == 0 or zi == plan.depth_samples:
                print(f"  depth slice {zi:>4}/{plan.depth_samples}")

    return mip.reshape(plan.out_h, plan.out_w)


def _compute_splat_mip_cuda(
    model: GaussianMixtureField,
    plan: RenderPlan,
    device: str,
) -> torch.Tensor:
    if not HAS_MIP_CUDA or mip_splat_forward is None:
        raise RuntimeError("CUDA MIP splat extension is unavailable")

    means = model.means.clone()
    log_scales = model.log_scales
    quaternions = model.quaternions.clone()
    log_amplitudes = model.log_amplitudes

    # Rotate Gaussians into camera space:
    # p_cam = R (p_world - c) + 0
    center = plan.world_center
    R = plan.rotation_matrix

    means = (means - center) @ R.T

    rot_quat = _matrix_to_quaternion(R.to(dtype=quaternions.dtype))
    quaternions = _quaternion_multiply(rot_quat, quaternions)

    with torch.no_grad():
        return mip_splat_forward(
            x_vals=plan.x_vals,
            y_vals=plan.y_vals,
            z_vals=plan.z_vals,
            means=means,
            log_scales=log_scales,
            quaternions=quaternions,
            log_amplitudes=log_amplitudes,
        )


def _apply_letterbox(img: np.ndarray, target_size: int) -> np.ndarray:
    h, w = img.shape
    if h == target_size and w == target_size:
        return img

    canvas = np.zeros((target_size, target_size), dtype=img.dtype)
    y0 = (target_size - h) // 2
    x0 = (target_size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Render reconstructed volume via MIP Gaussian splatting")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gmf_refined_best.pt", help="Checkpoint path")
    parser.add_argument("--config", type=str, default="config.yml", help="YAML config path")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--xy-batch-size", type=int, default=8192, help="XY batch size for field evaluation")
    parser.add_argument("--k-chunk", type=int, default=512, help="Gaussian chunk size in model forward")
    parser.add_argument(
        "--renderer",
        type=str,
        default="auto",
        choices=["auto", "cuda", "torch"],
        help="Renderer backend",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="Optional volume shape override (Z Y X)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test/reconstructed_mip_splat.pdf",
        help="Output visualization path",
    )
    parser.add_argument(
        "--save-npy",
        type=str,
        default="test/reconstructed_mip_splat.npy",
        help="Path to save raw MIP array",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Disable GT loading/comparison panel even if tif_path exists",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="test/reconstructed_mip_metrics.txt",
        help="Path to save metrics when GT is available",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw angle in degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch angle in degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll angle in degrees")
    parser.add_argument(
        "--screen-size",
        type=int,
        default=0,
        help="Set largest output dimension to this size while preserving aspect ratio",
    )
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="Pad to exact square when using --screen-size",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model, cfg = _load_model(args.ckpt, args.config, device)
    bounds = cfg.get("model", {}).get("bounds", [[-1, 1], [-1, 1], [-1, 1]])

    shape_zyx = _infer_volume_shape(cfg, tuple(args.shape) if args.shape is not None else None)
    bounds = _normalize_bounds_to_volume_aspect_ratio(bounds, shape_zyx)

    rotation_matrix = euler_to_rotation_matrix(
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        roll_deg=args.roll,
        device=device,
        dtype=torch.float32,
    )

    plan = _make_render_plan(
        bounds=bounds,
        shape_zyx=shape_zyx,
        rotation_matrix=rotation_matrix,
        screen_size=args.screen_size,
        device=device,
    )

    print(f"Checkpoint: {args.ckpt}")
    print(f"Gaussians: {model.num_gaussians}")
    print(f"Volume shape (Z,Y,X): {shape_zyx}")
    print(f"Bounds: {bounds}")
    print(f"Rotation: yaw={args.yaw}°, pitch={args.pitch}°, roll={args.roll}°")
    print(f"Output image size: {plan.out_h} x {plan.out_w}")
    print(f"Depth samples: {plan.depth_samples}")

    use_cuda_renderer = (
        device == "cuda"
        and HAS_MIP_CUDA
        and (args.renderer in ("auto", "cuda"))
    )
    if args.renderer == "cuda" and not use_cuda_renderer:
        raise RuntimeError(
            "renderer=cuda requested but CUDA MIP kernel is unavailable or device is not cuda. "
            "Build extension first."
        )

    if use_cuda_renderer:
        print("Renderer: CUDA MIP splatting kernel")
        mip = _compute_splat_mip_cuda(
            model=model,
            plan=plan,
            device=device,
        )
    else:
        print("Renderer: PyTorch fallback")
        mip = _compute_splat_mip_torch(
            model=model,
            plan=plan,
            xy_batch_size=args.xy_batch_size,
            k_chunk=args.k_chunk,
            device=device,
        )

    mip_np = mip.detach().cpu().numpy().astype(np.float32)

    if args.letterbox and args.screen_size > 0:
        mip_np = _apply_letterbox(mip_np, args.screen_size)
        print(f"Letterboxed to {mip_np.shape}")

    Path(args.save_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.save_npy, mip_np)

    # Only compare against GT for the default, unrotated, native-orientation rendering.
    can_compare_gt = (
        not args.no_gt
        and args.yaw == 0.0
        and args.pitch == 0.0
        and args.roll == 0.0
        and args.screen_size == 0
    )

    gt_mip = _load_gt_mip_if_available(cfg) if can_compare_gt else None
    if not can_compare_gt:
        if args.no_gt:
            print("Skipping GT comparison (--no-gt)")
        elif (args.yaw != 0.0 or args.pitch != 0.0 or args.roll != 0.0):
            print("Skipping GT comparison for rotated view")
        elif args.screen_size > 0:
            print("Skipping GT comparison for custom screen size")

    metrics = None
    if gt_mip is not None:
        metrics = _compute_metrics(gt_mip, mip_np, device=device)

        print("Metrics (normalized MIP):")
        if metrics["psnr"] is not None:
            if np.isfinite(metrics["psnr"]):
                print(f"  PSNR : {metrics['psnr']:.4f} dB")
            else:
                print("  PSNR : inf")
        if metrics["ssim"] is not None:
            print(f"  SSIM : {metrics['ssim']:.6f}")
        else:
            print("  SSIM : unavailable (install scikit-image)")
        if metrics["lpips"] is not None:
            print(f"  LPIPS: {metrics['lpips']:.6f}")
        else:
            print("  LPIPS: unavailable (install lpips)")

        Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_path, "w", encoding="utf-8") as f:
            f.write("MIP reconstruction metrics\n")
            f.write("==========================\n")
            f.write(f"PSNR: {metrics['psnr']}\n")
            f.write(f"SSIM: {metrics['ssim']}\n")
            f.write(f"LPIPS: {metrics['lpips']}\n")
        print(f"Saved metrics: {args.metrics_path}")

    _save_visualization(mip_np, args.output, gt_mip, metrics=metrics)

    print(f"Saved raw MIP: {args.save_npy}")
    print(f"Saved visualization: {args.output}")


if __name__ == "__main__":
    main()