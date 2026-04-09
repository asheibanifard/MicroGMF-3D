from __future__ import annotations

import math

import torch


def camera_internal(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    skew: float = 0.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a 3x3 camera intrinsics matrix K.

    K = [[fx, skew, cx],
         [ 0,   fy, cy],
         [ 0,    0,  1]]
    """
    return torch.tensor(
        [[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def intrinsics_from_fov(
    width: int,
    height: int,
    fov_y_deg: float,
    principal_point: tuple[float, float] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create K from image size and vertical field of view in degrees."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if fov_y_deg <= 0.0 or fov_y_deg >= 180.0:
        raise ValueError("fov_y_deg must be in (0, 180)")

    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * float(height) / math.tan(0.5 * fov_y)
    fx = fy

    if principal_point is None:
        cx = (float(width) - 1.0) * 0.5
        cy = (float(height) - 1.0) * 0.5
    else:
        cx, cy = principal_point

    return camera_internal(fx=fx, fy=fy, cx=cx, cy=cy, device=device, dtype=dtype)


def camera_external(
    R: torch.Tensor,
    t: torch.Tensor,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a 4x4 world-to-camera extrinsics matrix [R|t].

    Args:
        R: Rotation matrix of shape (3, 3).
        t: Translation vector of shape (3,) or (3, 1).
    """
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {tuple(R.shape)}")

    t = t.reshape(-1)
    if t.shape[0] != 3:
        raise ValueError(f"t must have 3 elements, got {tuple(t.shape)}")

    dev = device if device is not None else R.device
    dt = dtype if dtype is not None else R.dtype

    Rt = torch.eye(4, device=dev, dtype=dt)
    Rt[:3, :3] = R.to(device=dev, dtype=dt)
    Rt[:3, 3] = t.to(device=dev, dtype=dt)
    return Rt


def invert_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 rigid transform matrix.

    Input and output are camera transforms of shape (4, 4).
    For world-to-camera input, the output is camera-to-world.
    """
    if extrinsics.shape != (4, 4):
        raise ValueError(f"extrinsics must have shape (4, 4), got {tuple(extrinsics.shape)}")

    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    R_inv = R.transpose(0, 1)
    t_inv = -R_inv @ t

    out = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype)
    out[:3, :3] = R_inv
    out[:3, 3] = t_inv
    return out


def look_at_extrinsics(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Construct world-to-camera extrinsics using a look-at convention.

    The camera looks from `eye` toward `target`.
    """
    eye = eye.reshape(3)
    target = target.reshape(3)

    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype)
    else:
        up = up.reshape(3).to(device=eye.device, dtype=eye.dtype)

    forward = target - eye
    forward = forward / (torch.linalg.norm(forward) + eps)

    right = torch.cross(forward, up, dim=0)
    right = right / (torch.linalg.norm(right) + eps)

    true_up = torch.cross(right, forward, dim=0)
    true_up = true_up / (torch.linalg.norm(true_up) + eps)

    # Camera basis in world coordinates. We use -forward so camera +Z points backward.
    R_c2w = torch.stack([right, true_up, -forward], dim=1)
    R_w2c = R_c2w.transpose(0, 1)
    t_w2c = -R_w2c @ eye

    return camera_external(R_w2c, t_w2c)


def split_extrinsics(extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 4x4 extrinsics matrix into (R, t)."""
    if extrinsics.shape != (4, 4):
        raise ValueError(f"extrinsics must have shape (4, 4), got {tuple(extrinsics.shape)}")
    return extrinsics[:3, :3], extrinsics[:3, 3]


__all__ = [
    "camera_internal",
    "intrinsics_from_fov",
    "camera_external",
    "invert_extrinsics",
    "look_at_extrinsics",
    "split_extrinsics",
]
