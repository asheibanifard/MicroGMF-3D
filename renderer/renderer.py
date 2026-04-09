import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# ============================================================
# 1. Gaussian storage
# ============================================================
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Batched quaternion (wxyz) → rotation matrix.
    Args:
        q: (N, 4) float tensor, wxyz convention.
    Returns:
        R: (N, 3, 3) float tensor.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros(N, 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class GaussianScene:
    means: torch.Tensor       # (N, 3)
    scales: torch.Tensor      # (N, 3) positive
    intensity: torch.Tensor   # (N, 1)
    quaternions: torch.Tensor # (N, 4)

    def __post_init__(self):
        if self.means.ndim != 2 or self.means.shape[1] != 3:
            raise ValueError("means must have shape (N, 3)")
        if self.scales.ndim != 2 or self.scales.shape[1] != 3:
            raise ValueError("scales must have shape (N, 3)")
        if self.intensity.ndim == 1:
            self.intensity = self.intensity[:, None]
        if self.intensity.ndim != 2 or self.intensity.shape[1] != 1:
            raise ValueError("intensity must have shape (N, 1) or (N,)")

        if not (self.means.shape[0] == self.scales.shape[0] == self.intensity.shape[0]):
            raise ValueError("means/scales/intensity must have same N")
        if not (self.means.shape[0] == self.quaternions.shape[0]):
            raise ValueError("means/quaternions must have same N")
        if self.quaternions.ndim != 2 or self.quaternions.shape[1] != 4:
            raise ValueError("quaternions must have shape (N, 4)")
        
    @property
    def device(self):
        return self.means.device

    @property
    def N(self):
        return self.means.shape[0]

    def covariance_world(self) -> torch.Tensor:
        # Axis-aligned Gaussian covariance
        # shape: (N, 3, 3)
        q = F.normalize(self.quaternions, dim=-1)
        R = quat_to_rotmat(q)  # (N, 3, 3)
        S2 = torch.diag_embed(self.scales ** 2)
        return R @ S2 @ R.transpose(-1, -2)


# ============================================================
# 2. Camera model
# ============================================================

@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_wc: torch.Tensor   # (3, 3): world -> camera rotation
    t_wc: torch.Tensor   # (3,):   world -> camera translation

    @property
    def device(self):
        return self.R_wc.device


def look_at(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns world->camera rotation R_wc and translation t_wc.
    Camera convention:
      x right, y down-ish via chosen up, z forward
    """
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype)

    forward = target - eye
    forward = forward / (torch.norm(forward) + 1e-8)

    right = torch.cross(forward, up, dim=0)
    right = right / (torch.norm(right) + 1e-8)

    true_up = torch.cross(right, forward, dim=0)
    true_up = true_up / (torch.norm(true_up) + 1e-8)

    # Camera basis in world coordinates
    # We want world->camera rotation
    # Camera z axis = forward
    R_cw = torch.stack([right, true_up, forward], dim=1)  # columns are basis
    R_wc = R_cw.T
    t_wc = -R_wc @ eye
    return R_wc, t_wc


# ============================================================
# 3. Scene creation
# ============================================================

def create_demo_scene(
    n_gaussians: int = 300,
    device: str = "cpu",
    seed: int = 0,
) -> GaussianScene:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # A rough "tube / cloud" style scene
    t = torch.linspace(-1.5, 1.5, n_gaussians, device=device)
    x = 0.7 * torch.sin(2.0 * t) + 0.08 * torch.randn(n_gaussians, device=device, generator=g)
    y = 0.5 * torch.cos(1.5 * t) + 0.08 * torch.randn(n_gaussians, device=device, generator=g)
    z = t + 0.08 * torch.randn(n_gaussians, device=device, generator=g)
    means = torch.stack([x, y, z], dim=1)

    scales = 0.04 + 0.03 * torch.rand(n_gaussians, 3, device=device, generator=g)
    # Make a bit anisotropic
    scales[:, 2] *= 1.8

    intensity = 0.5 + 0.5 * torch.rand(n_gaussians, 1, device=device, generator=g)

    quaternions = torch.zeros(n_gaussians, 4, device=device)
    quaternions[:, 0] = 1.0  # Identity rotation for all Gaussians (axis-aligned)
    return GaussianScene(means=means, scales=scales, intensity=intensity, quaternions=quaternions)


# ============================================================
# 4. Transform to camera
# ============================================================

def transform_to_camera(scene: GaussianScene, cam: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      means_cam: (N, 3)
      covs_cam:  (N, 3, 3)
    """
    means_cam = (cam.R_wc @ scene.means.T).T + cam.t_wc[None, :]
    covs_world = scene.covariance_world()
    covs_cam = cam.R_wc[None, :, :] @ covs_world @ cam.R_wc.T[None, :, :]
    return means_cam, covs_cam


# ============================================================
# 5. Cull
# ============================================================

def cull_gaussians(
    means_cam: torch.Tensor,
    covs_cam: torch.Tensor,
    intensity: torch.Tensor,
    cam: Camera,
    near: float = 0.1,
    far: float = 100.0,
    margin: float = 64.0,
) -> torch.Tensor:
    """
    Conservative culling by center projection only.
    Returns boolean mask of shape (N,)
    """
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]

    valid_z = (z > near) & (z < far)

    # Avoid divide by zero for invalid z
    z_safe = torch.clamp(z, min=near)
    u = cam.fx * (x / z_safe) + cam.cx
    v = cam.fy * (y / z_safe) + cam.cy

    in_screen = (
        (u >= -margin) &
        (u <= cam.width + margin) &
        (v >= -margin) &
        (v <= cam.height + margin)
    )

    strong_enough = (intensity[:, 0] > 1e-6)
    return valid_z & in_screen & strong_enough


# ============================================================
# 6. Project 3D Gaussian to 2D ellipse
# ============================================================

def project_gaussians_to_2d(
    means_cam: torch.Tensor,
    covs_cam: torch.Tensor,
    cam: Camera,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mu_2d:  (N, 2)
      cov_2d: (N, 2, 2)

    Uses Jacobian linearization:
      Sigma_2D = J Sigma_3D J^T

    Perspective projection:
      u = fx * x/z + cx
      v = fy * y/z + cy
    """
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z = means_cam[:, 2].clamp(min=1e-6)

    u = cam.fx * (x / z) + cam.cx
    v = cam.fy * (y / z) + cam.cy
    mu_2d = torch.stack([u, v], dim=1)

    N = means_cam.shape[0]
    J = torch.zeros(N, 2, 3, device=means_cam.device, dtype=means_cam.dtype)

    J[:, 0, 0] = cam.fx / z
    J[:, 0, 2] = -cam.fx * x / (z ** 2)

    J[:, 1, 1] = cam.fy / z
    J[:, 1, 2] = -cam.fy * y / (z ** 2)

    cov_2d = J @ covs_cam @ J.transpose(-1, -2)
    return mu_2d, cov_2d


# ============================================================
# 7. Apply MIP filtering
# ============================================================

def apply_mip_filter(
    cov_2d: torch.Tensor,
    pixel_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Adds isotropic pixel/filter support.
    """
    eye = torch.eye(2, device=cov_2d.device, dtype=cov_2d.dtype)[None, :, :]
    return cov_2d + (pixel_sigma ** 2) * eye


# ============================================================
# 8. Compute screen-space bounds
# ============================================================

def ellipse_bboxes(
    mu_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    width: int,
    height: int,
    nsigmas: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns integer bounding boxes:
      xmin, xmax, ymin, ymax each shape (N,)
    """
    # Use eigenvalues to estimate support radius
    evals = torch.linalg.eigvalsh(cov_2d)  # (N, 2)
    evals = torch.clamp(evals, min=1e-10)
    radii = nsigmas * torch.sqrt(evals.max(dim=1).values)

    xmin = torch.floor(mu_2d[:, 0] - radii).long().clamp(0, width - 1)
    xmax = torch.ceil(mu_2d[:, 0] + radii).long().clamp(0, width - 1)
    ymin = torch.floor(mu_2d[:, 1] - radii).long().clamp(0, height - 1)
    ymax = torch.ceil(mu_2d[:, 1] + radii).long().clamp(0, height - 1)
    return xmin, xmax, ymin, ymax


# ============================================================
# 9. Tile / bin
# ============================================================

def bin_to_tiles(
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    ymin: torch.Tensor,
    ymax: torch.Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
) -> Tuple[Dict[Tuple[int, int], List[int]], int, int]:
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    tile_bins: Dict[Tuple[int, int], List[int]] = {}

    for i in range(xmin.shape[0]):
        tx0 = int(xmin[i].item() // tile_size)
        tx1 = int(xmax[i].item() // tile_size)
        ty0 = int(ymin[i].item() // tile_size)
        ty1 = int(ymax[i].item() // tile_size)

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                key = (ty, tx)
                if key not in tile_bins:
                    tile_bins[key] = []
                tile_bins[key].append(i)

    return tile_bins, tiles_y, tiles_x


# ============================================================
# 10. Rasterize
# ============================================================

def rasterize_tiles(
    mu_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    intensity: torch.Tensor,
    tile_bins: Dict[Tuple[int, int], List[int]],
    width: int,
    height: int,
    tile_size: int = 16,
    mode: str = "max",  # "max" or "sum"
) -> torch.Tensor:
    """
    CPU/PyTorch reference rasterizer.
    """
    device = mu_2d.device
    image = torch.zeros(height, width, device=device, dtype=mu_2d.dtype)

    if mode not in {"max", "sum"}:
        raise ValueError("mode must be 'max' or 'sum'")

    # Precompute inverse covariance
    cov_inv = torch.linalg.inv(cov_2d)

    for (ty, tx), ids in tile_bins.items():
        y0 = ty * tile_size
        y1 = min((ty + 1) * tile_size, height)
        x0 = tx * tile_size
        x1 = min((tx + 1) * tile_size, width)

        if y1 <= y0 or x1 <= x0:
            continue

        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device, dtype=mu_2d.dtype),
            torch.arange(x0, x1, device=device, dtype=mu_2d.dtype),
            indexing="ij",
        )
        pix = torch.stack([xx, yy], dim=-1)  # (Htile, Wtile, 2)

        tile_img = torch.zeros(y1 - y0, x1 - x0, device=device, dtype=mu_2d.dtype)

        for i in ids:
            delta = pix - mu_2d[i]  # (Htile, Wtile, 2)
            mahal = torch.einsum("...i,ij,...j->...", delta, cov_inv[i], delta)
            contrib = intensity[i, 0] * torch.exp(-0.5 * mahal)

            if mode == "max":
                tile_img = torch.maximum(tile_img, contrib)
            else:
                tile_img = tile_img + contrib

        image[y0:y1, x0:x1] = tile_img if mode == "sum" else torch.maximum(image[y0:y1, x0:x1], tile_img)

    return image


# ============================================================
# 11. Full pipeline per frame
# ============================================================

def render_frame(
    scene: GaussianScene,
    cam: Camera,
    tile_size: int = 16,
    pixel_sigma: float = 1.0,
    nsigmas: float = 3.0,
    mode: str = "max",
) -> torch.Tensor:
    means_cam, covs_cam = transform_to_camera(scene, cam)

    mask = cull_gaussians(means_cam, covs_cam, scene.intensity, cam)
    if mask.sum() == 0:
        return torch.zeros(cam.height, cam.width, device=scene.device)

    means_cam = means_cam[mask]
    covs_cam = covs_cam[mask]
    intensity = scene.intensity[mask]

    mu_2d, cov_2d = project_gaussians_to_2d(means_cam, covs_cam, cam)
    cov_2d = apply_mip_filter(cov_2d, pixel_sigma=pixel_sigma)

    xmin, xmax, ymin, ymax = ellipse_bboxes(
        mu_2d, cov_2d, cam.width, cam.height, nsigmas=nsigmas
    )

    tile_bins, _, _ = bin_to_tiles(
        xmin, xmax, ymin, ymax,
        width=cam.width,
        height=cam.height,
        tile_size=tile_size,
    )

    image = rasterize_tiles(
        mu_2d=mu_2d,
        cov_2d=cov_2d,
        intensity=intensity,
        tile_bins=tile_bins,
        width=cam.width,
        height=cam.height,
        tile_size=tile_size,
        mode=mode,
    )

    # Normalize for display
    image = image / (image.max() + 1e-8)
    return image


# ============================================================
# 12. Interactive / repeated rendering
# ============================================================

def orbit_camera(
    width: int,
    height: int,
    radius: float,
    angle_deg: float,
    elevation_deg: float = 15.0,
    fx: float = 400.0,
    fy: float = 400.0,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: str = "cpu",
) -> Camera:
    angle = math.radians(angle_deg)
    elev = math.radians(elevation_deg)

    eye = torch.tensor([
        radius * math.cos(elev) * math.cos(angle),
        radius * math.sin(elev),
        radius * math.cos(elev) * math.sin(angle),
    ], device=device, dtype=torch.float32)

    target_t = torch.tensor(target, device=device, dtype=torch.float32)
    R_wc, t_wc = look_at(eye, target_t)

    return Camera(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
        R_wc=R_wc,
        t_wc=t_wc,
    )


def run_animation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create scene
    scene = create_demo_scene(n_gaussians=300, device=device, seed=42)

    width, height = 512, 512
    tile_size = 16
    pixel_sigma = 1.0
    mode = "max"   # try "sum" too

    fig, ax = plt.subplots(figsize=(6, 6))
    img_artist = ax.imshow(
        torch.zeros(height, width).cpu().numpy(),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        animated=True,
    )
    ax.set_title("3D Gaussian MIP Splatting")
    ax.axis("off")

    def update(frame_idx):
        angle = frame_idx * 3.0
        cam = orbit_camera(
            width=width,
            height=height,
            radius=4.0,
            angle_deg=angle,
            elevation_deg=20.0,
            fx=420.0,
            fy=420.0,
            device=device,
        )

        image = render_frame(
            scene=scene,
            cam=cam,
            tile_size=tile_size,
            pixel_sigma=pixel_sigma,
            nsigmas=3.0,
            mode=mode,
        )

        img_artist.set_array(image.detach().cpu().numpy())
        return [img_artist]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=120,
        interval=40,
        blit=True,
        repeat=True,
    )

    plt.show()


if __name__ == "__main__":
    run_animation()