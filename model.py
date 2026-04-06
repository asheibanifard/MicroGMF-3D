from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cuda.cuda_kernels import _build_L_chol, _gaussian_eval_cuda_fn
try:
    import gaussian_eval_cuda_forward
    import gaussian_eval_cuda_backward
    HAS_CUDA_EXTENSION = True
    print("✓ Custom CUDA extensions loaded (forward + backward)")
except ImportError:
    HAS_CUDA_EXTENSION = False
    print("✗ Custom CUDA extensions not found — using PyTorch fallback")

from regularisers import _gradient_magnitudes


class GaussianMixtureField(nn.Module):
    """
    Anisotropic 3-D Gaussian Mixture Field.

    Parameters are (means, log_scales, quaternions, log_amplitudes) and the
    covariance is  Σ = R diag(s²) Rᵀ  with R from unit quaternion.
    """

    def __init__(
        self,
        num_gaussians: int,
        init_scale: float = 0.05,
        init_amplitude: float = 0.1,
        bounds: list | None = None,
        aabb: list | None = None,
        swc_coords: np.ndarray | None = None,
        swc_radii: np.ndarray | None = None,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians

        # Axis-aligned bounding box
        if aabb is not None:
            self.aabb = torch.tensor(aabb, dtype=torch.float32)
        elif bounds is not None:
            self.aabb = torch.tensor(bounds, dtype=torch.float32)
        else:
            self.aabb = torch.tensor([[-1, 1], [-1, 1], [-1, 1]], dtype=torch.float32)

        # --- initialise means ---
        if swc_coords is not None:
            # Initialise from SWC neuron morphology
            n_swc = swc_coords.shape[0]
            if n_swc >= num_gaussians:
                # Subsample: uniformly pick num_gaussians points along the skeleton
                idx = np.linspace(0, n_swc - 1, num_gaussians, dtype=int)
                means = torch.from_numpy(swc_coords[idx]).float()
            else:
                # Fewer SWC nodes than Gaussians: use all nodes + fill rest
                # by interpolating random pairs along skeleton edges
                means_swc = torch.from_numpy(swc_coords).float()
                n_extra = num_gaussians - n_swc
                # Random pairs of consecutive nodes for interpolation
                if n_swc > 1:
                    pair_idx = torch.randint(0, n_swc - 1, (n_extra,))
                    t = torch.rand(n_extra, 1)
                    extra = means_swc[pair_idx] * (1 - t) + means_swc[pair_idx + 1] * t
                else:
                    extra = means_swc.expand(n_extra, -1).clone()
                # Add small jitter to avoid exact duplicates
                extra += torch.randn_like(extra) * 0.001
                means = torch.cat([means_swc, extra], dim=0)
            print(f"SWC init: {n_swc} nodes → {num_gaussians} Gaussians")
        elif bounds is not None:
            means = torch.zeros(num_gaussians, 3)
            for i in range(3):
                lo, hi = bounds[i][0], bounds[i][1]
                means[:, i] = torch.rand(num_gaussians) * (hi - lo) + lo
        else:
            means = torch.randn(num_gaussians, 3) * 0.1

        self.means = nn.Parameter(means)

        # Scale init: use SWC radii if provided, otherwise use init_scale
        if swc_radii is not None:
            # Per-Gaussian scale from SWC radius (isotropic initial scale)
            if swc_coords.shape[0] >= num_gaussians:
                idx = np.linspace(0, swc_coords.shape[0] - 1, num_gaussians, dtype=int)
                radii_sel = swc_radii[idx]
            else:
                # Pad extra Gaussians with median radius
                med_r = float(np.median(swc_radii))
                radii_sel = np.concatenate([
                    swc_radii,
                    np.full(num_gaussians - swc_coords.shape[0], med_r, dtype=np.float32)
                ])
            radii_t = torch.from_numpy(radii_sel).float().clamp(min=1e-4)
            self.log_scales = nn.Parameter(
                torch.log(radii_t).unsqueeze(-1).expand(-1, 3).contiguous()
            )
            print(f"SWC scale init: radius range [{radii_t.min():.4f}, {radii_t.max():.4f}]")
        else:
            # Fallback: auto-compute if init_scale too small
            if init_scale < 1e-3:
                side = 2.0  # [-1, 1]
                init_scale = side / (num_gaussians ** (1.0 / 3.0)) * 1.5
                print(f"⚠ init_scale too small, auto-set to {init_scale:.4f}")
            self.log_scales = nn.Parameter(
                torch.ones(num_gaussians, 3) * math.log(init_scale)
            )

        q = torch.zeros(num_gaussians, 4)
        q[:, 0] = 1.0  # identity rotation
        self.quaternions = nn.Parameter(q)

        # Amplitude init: start moderate, not at 1.0
        # With many overlapping Gaussians, amp=1.0 creates huge peaks;
        # amp=0.01–0.1 lets them sum to reasonable values.
        self.log_amplitudes = nn.Parameter(
            torch.ones(num_gaussians) * math.log(max(init_amplitude, 1e-6))
        )

    # ---- geometry helpers ------------------------------------------------
    @staticmethod
    def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, p=2, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
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

    def get_covariance_matrices(self) -> torch.Tensor:
        s = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        R = self.quat_to_rotmat(self.quaternions)
        S2 = torch.diag_embed(s ** 2)
        return R @ S2 @ R.transpose(-2, -1)

    # ---- constraints (call after optimizer.step) -------------------------
    def apply_aabb_clamp(self, margin: float = 0.0):
        aabb = self.aabb.to(self.means.device)
        with torch.no_grad():
            for i in range(3):
                self.means.data[:, i].clamp_(aabb[i, 0] + margin, aabb[i, 1] - margin)

    def clamp_log_scales_(self, lo: float, hi: float):
        with torch.no_grad():
            self.log_scales.data.clamp_(lo, hi)

    def clamp_log_amplitudes_(self, lo: float, hi: float):
        with torch.no_grad():
            self.log_amplitudes.data.clamp_(lo, hi)

    # ---- forward (dispatches CUDA kernel or K-chunked PyTorch) -----------
    def forward(self, x: torch.Tensor, k_chunk: int = 1024) -> torch.Tensor:
        """
        Evaluate field at query points  x  (N, 3) → (N,).

        If the custom CUDA extension is available and the full (N, K) tensor
        fits in GPU memory, uses the fused CUDA kernel for speedup on the
        forward + backward of (x, means, amplitudes), while gradients for
        log_scales and quaternions are chained through PyTorch autograd
        via the Cholesky factorisation.

        Otherwise falls back to K-chunked PyTorch solve_triangular.
        """
        N = x.shape[0]
        K = self.num_gaussians
        log_amps = self.log_amplitudes.clamp(-10.0, 6.0)

        # --- CUDA kernel path ---
        if HAS_CUDA_EXTENSION and x.is_cuda:
            # Memory estimate: (N*K) * ~48 bytes (vals + recompute in backward)
            mem_estimate = N * K * 48
            mem_free = torch.cuda.mem_get_info(x.device)[0]
            use_cuda = mem_estimate < mem_free * 0.8

            if use_cuda:
                if not hasattr(self, '_cuda_logged'):
                    print(f"✓ Using CUDA kernel: N={N}, K={K}, mem_estimate={mem_estimate/1e9:.2f}GB")
                    self._cuda_logged = True
                # Precompute L_chol (detached) — the autograd Function
                # recomputes it inside backward with grad tracking.
                with torch.no_grad():
                    L_chol = _build_L_chol(self.log_scales, self.quaternions)
                return _gaussian_eval_cuda_fn(
                    x, self.means, self.log_scales, self.quaternions,
                    log_amps, L_chol,
                )
            else:
                if not hasattr(self, '_fallback_logged'):
                    print(f"✗ CUDA kernel skipped (memory): N={N}, K={K}, need={mem_estimate/1e9:.2f}GB, free={mem_free/1e9:.2f}GB")
                    self._fallback_logged = True

        # --- Chunked PyTorch fallback ---
        dtype_in = x.dtype
        amps = torch.exp(log_amps)

        Sigma = self.get_covariance_matrices()
        eps = 1e-5
        Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device).unsqueeze(0)
        try:
            L = torch.linalg.cholesky(Sigma_reg.float())
        except torch._C._LinAlgError:
            Sigma_reg = Sigma + 1e-3 * torch.eye(3, device=Sigma.device).unsqueeze(0)
            L = torch.linalg.cholesky(Sigma_reg.float())

        out = torch.zeros(N, device=x.device, dtype=dtype_in)

        for ks in range(0, K, k_chunk):
            ke = min(ks + k_chunk, K)
            G = ke - ks

            mu = self.means[ks:ke]
            a = amps[ks:ke]
            Lc = L[ks:ke]

            diff = (x[:, None, :] - mu[None, :, :])
            diff_flat = diff.reshape(N * G, 3, 1).float()
            L_exp = Lc.unsqueeze(0).expand(N, G, 3, 3).reshape(N * G, 3, 3)

            y = torch.linalg.solve_triangular(L_exp, diff_flat, upper=False)
            mahal = (y.squeeze(-1) ** 2).sum(-1).reshape(N, G)

            vals = a[None, :] * torch.exp(-0.5 * mahal.to(dtype_in))
            out = out + vals.sum(dim=1)

        return out

    # ---- densify / prune -------------------------------------------------
    def densify_and_prune(
        self,
        grad_threshold: float = 1.5e-4,
        min_amplitude: float = 5e-4,
        max_scale: float = 0.8,
        split_scale_threshold: float = 0.05,
        enforce_aabb: bool = True,
        max_gaussians: int = 0,
        max_clones: int = 0,
    ) -> dict:
        with torch.no_grad():
            device = self.means.device

            grad_mag = _gradient_magnitudes(self.means)
            scales = torch.exp(self.log_scales).clamp(1e-5, 1e2)
            max_s = scales.max(dim=-1)[0]
            amps = torch.exp(self.log_amplitudes)

            high_grad = grad_mag > grad_threshold
            small = max_s < split_scale_threshold
            clone_mask = high_grad & small
            split_mask = high_grad & (~small)

            new_m, new_ls, new_q, new_la = [], [], [], []

            # Clone (with optional cap)
            if clone_mask.any():
                if max_clones > 0 and int(clone_mask.sum()) > max_clones:
                    # Keep only the top-gradient clones
                    clone_idx = clone_mask.nonzero(as_tuple=True)[0]
                    clone_grads = grad_mag[clone_idx]
                    _, topk = clone_grads.topk(max_clones, largest=True)
                    clone_idx = clone_idx[topk]
                    clone_mask = torch.zeros_like(clone_mask)
                    clone_mask[clone_idx] = True
                new_m.append(self.means[clone_mask])
                new_ls.append(self.log_scales[clone_mask])
                new_q.append(self.quaternions[clone_mask])
                new_la.append(self.log_amplitudes[clone_mask])

            # Split
            if split_mask.any():
                m = self.means[split_mask]
                ls = self.log_scales[split_mask]
                q = self.quaternions[split_mask]
                la = self.log_amplitudes[split_mask]
                R = self.quat_to_rotmat(q)
                s = torch.exp(ls)
                imax = s.argmax(dim=-1)
                bi = torch.arange(m.shape[0], device=device)
                principal = R[bi, :, imax]
                offset = s[bi, imax].unsqueeze(-1) * 0.5

                c1 = m + principal * offset
                c2 = m - principal * offset
                child_ls = ls - math.log(1.6)
                child_la = la - math.log(2.0)

                new_m.extend([c1, c2])
                new_ls.extend([child_ls, child_ls])
                new_q.extend([q, q])
                new_la.extend([child_la, child_la])

            # Prune (also remove split parents — they are replaced by their children)
            keep = (amps > min_amplitude) & (max_s < max_scale) & ~split_mask
            if enforce_aabb:
                aabb = self.aabb.to(device)
                within = torch.ones(self.means.shape[0], dtype=torch.bool, device=device)
                for i in range(3):
                    within &= (self.means[:, i] >= aabb[i, 0]) & (
                        self.means[:, i] <= aabb[i, 1]
                    )
                keep &= within

            old_K = self.num_gaussians
            pruned = int((~keep).sum().item())

            parts = [
                self.means[keep],
                self.log_scales[keep],
                self.quaternions[keep],
                self.log_amplitudes[keep],
            ]
            if new_m:
                parts_new = [
                    torch.cat(new_m),
                    torch.cat(new_ls),
                    torch.cat(new_q),
                    torch.cat(new_la),
                ]
                combined = [torch.cat([p, pn]) for p, pn in zip(parts, parts_new)]
            else:
                combined = parts

            # Enforce max_gaussians cap: keep highest-weight Gaussians
            cap_pruned = 0
            if max_gaussians > 0 and combined[0].shape[0] > max_gaussians:
                cap_pruned = combined[0].shape[0] - max_gaussians
                cap_weights = torch.exp(combined[3])
                _, topk_idx = cap_weights.topk(max_gaussians, largest=True)
                topk_idx = topk_idx.sort()[0]
                combined = [c[topk_idx] for c in combined]

            self.means = nn.Parameter(combined[0])
            self.log_scales = nn.Parameter(combined[1])
            self.quaternions = nn.Parameter(combined[2])
            self.log_amplitudes = nn.Parameter(combined[3])
            self.num_gaussians = int(combined[0].shape[0])

            return {
                "old": old_K,
                "new": self.num_gaussians,
                "pruned": pruned,
                "cloned": int(clone_mask.sum().item()),
                "split": int(split_mask.sum().item()) * 2,
                "Δ": self.num_gaussians - old_K,
                "cap_pruned": cap_pruned,
            }
