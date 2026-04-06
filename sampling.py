from __future__ import annotations

# ===================================================================
#  GPU volume sampling
# ===================================================================
import torch


_sampling_cdf_cache: dict[str, torch.Tensor] = {}


def _flat_indices_gpu(
    vol_gpu: torch.Tensor,
    num_samples: int,
    intensity_weighted: bool,
    cache_key: str | None,
) -> torch.Tensor:
    Z, Y, X = vol_gpu.shape
    Nvox = Z * Y * X
    device = vol_gpu.device

    if num_samples > Nvox:
        raise ValueError(f"num_samples={num_samples} > total voxels={Nvox}")

    if intensity_weighted:
        if cache_key and cache_key in _sampling_cdf_cache:
            cdf = _sampling_cdf_cache[cache_key]
        else:
            flat = vol_gpu.reshape(-1)
            probs = flat / (flat.sum() + 1e-12)
            cdf = torch.cumsum(probs, dim=0)
            if cache_key:
                _sampling_cdf_cache[cache_key] = cdf
        u = torch.rand(num_samples, device=device)
        idx = torch.searchsorted(cdf, u).clamp(0, Nvox - 1)
    else:
        idx = torch.randperm(Nvox, device=device)[:num_samples]
    return idx


def _idx_to_coords(idx: torch.Tensor, Z: int, Y: int, X: int):
    """Flat index → integer (z, y, x) + normalised [-1, 1] coords."""
    z = idx // (Y * X)
    rem = idx % (Y * X)
    y = rem // X
    x = rem % X

    xn = (x.float() / max(X - 1, 1)) * 2 - 1
    yn = (y.float() / max(Y - 1, 1)) * 2 - 1
    zn = (z.float() / max(Z - 1, 1)) * 2 - 1
    return z, y, x, xn, yn, zn


def sample_points_from_volume(
    vol_gpu: torch.Tensor,
    num_samples: int,
    intensity_weighted: bool = True,
    cache_key: str | None = None,
):
    """Sample random voxels and return (pts (N,3), vals (N,))."""
    Z, Y, X = vol_gpu.shape
    idx = _flat_indices_gpu(vol_gpu, num_samples, intensity_weighted, cache_key)
    z, y, x, xn, yn, zn = _idx_to_coords(idx, Z, Y, X)
    pts = torch.stack([xn, yn, zn], dim=1)
    vals = vol_gpu[z, y, x]
    return pts, vals


def sample_points_with_neighbors(
    vol_gpu: torch.Tensor,
    num_samples: int,
    delta_vox: int = 1,
    intensity_weighted: bool = True,
    cache_key: str | None = None,
):
    """
    Sample centre voxels *and* their +δ neighbours along each axis for
    finite-difference gradient supervision.

    Returns
    -------
    pts, vals,              — centre points
    pts_dx, vals_dx,        — neighbour shifted in x
    pts_dy, vals_dy,        — neighbour shifted in y
    pts_dz, vals_dz         — neighbour shifted in z
    """
    Z, Y, X = vol_gpu.shape
    device = vol_gpu.device

    idx = _flat_indices_gpu(vol_gpu, num_samples, intensity_weighted, cache_key)
    z, y, x, xn, yn, zn = _idx_to_coords(idx, Z, Y, X)

    pts = torch.stack([xn, yn, zn], dim=1)
    vals = vol_gpu[z, y, x]

    # Forward neighbours (clamped at boundary)
    x1 = (x + delta_vox).clamp(0, X - 1)
    y1 = (y + delta_vox).clamp(0, Y - 1)
    z1 = (z + delta_vox).clamp(0, Z - 1)

    vals_dx = vol_gpu[z, y, x1]
    vals_dy = vol_gpu[z, y1, x]
    vals_dz = vol_gpu[z1, y, x]

    # Neighbour normalised coords
    x1n = (x1.float() / max(X - 1, 1)) * 2 - 1
    y1n = (y1.float() / max(Y - 1, 1)) * 2 - 1
    z1n = (z1.float() / max(Z - 1, 1)) * 2 - 1

    pts_dx = torch.stack([x1n, yn, zn], dim=1)
    pts_dy = torch.stack([xn, y1n, zn], dim=1)
    pts_dz = torch.stack([xn, yn, z1n], dim=1)

    return pts, vals, pts_dx, vals_dx, pts_dy, vals_dy, pts_dz, vals_dz
