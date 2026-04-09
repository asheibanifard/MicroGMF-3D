"""
export_ply.py  —  Export a pretrained GMF model (.pt) to binary PLY for the JS renderer.

Usage:
    python export_ply.py model.pt output.ply

PLY output properties (float32, binary little-endian):
    x, y, z, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3, intensity
    rot_* = quaternion (w, x, y, z)
"""

import sys
import os
import struct
import numpy as np
import torch

# ── Key name candidates (edit if your model uses different names) ─────────────
MEAN_KEYS        = ['means', '_means', 'mu', '_mu',
                    'gaussians.means', 'model.means', 'params.means']
SCALE_KEYS       = ['log_scales', '_log_scales', 'log_scale',
                    'scales', '_scales', 'scale',
                    'gaussians.scales', 'model.scales', 'params.scales']
ROTATION_KEYS    = ['rotations', '_rotations', 'quats', '_quats',
                    'quaternions', 'rotation', 'quaternions',
                    'gaussians.rotations', 'model.rotations', 'params.rotations']
INTENSITY_KEYS   = ['log_intensities', '_log_intensities', 'log_intensity',
                    'intensities', '_intensities', 'intensity', 'log_amplitudes',
                    'features_dc', 'colors', 'color',
                    'gaussians.intensities', 'model.intensities', 'params.intensities']


def _get(state, candidates, name):
    for k in candidates:
        if k in state:
            return k, state[k].float().detach().cpu()
    print(f"  [!] Could not find '{name}'. Tried: {candidates}")
    print(f"      Available keys: {list(state.keys())}")
    return None, None


def export(model_path: str, output_path: str):
    print(f"\n── Loading  {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    # Unwrap checkpoint wrappers
    for wrapper_key in ('state_dict', 'model_state_dict', 'model', 'gaussians'):
        if isinstance(ckpt, dict) and wrapper_key in ckpt and isinstance(ckpt[wrapper_key], dict):
            ckpt = ckpt[wrapper_key]
            print(f"  Unwrapped '{wrapper_key}'")
            break
    if not isinstance(ckpt, dict):
        # nn.Module — grab its state dict
        ckpt = ckpt.state_dict()

    print("\n── State dict keys:")
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor):
            print(f"   {k:50s}  {str(tuple(v.shape)):20s}  {v.dtype}")

    # ── Means ────────────────────────────────────────────────────────────────
    key, means = _get(ckpt, MEAN_KEYS, 'means')
    if means is None:
        sys.exit(1)
    if means.dim() == 1:
        means = means.reshape(-1, 3)
    N = means.shape[0]
    print(f"\n── N = {N:,} Gaussians  (from '{key}')")

    # ── Scales ───────────────────────────────────────────────────────────────
    key, raw_scales = _get(ckpt, SCALE_KEYS, 'scales')
    if raw_scales is None:
        print("  Using default scale 0.005")
        scales = torch.full((N, 3), 0.005)
    else:
        if raw_scales.dim() == 1:
            raw_scales = raw_scales.reshape(-1, 3)
        if 'log' in (key or ''):
            scales = torch.exp(raw_scales)
        else:
            scales = raw_scales.abs()
        scales = scales.clamp(1e-7, 1e6)
        print(f"  Scales from '{key}'  range [{scales.min():.4e}, {scales.max():.4e}]")

    # ── Rotations (quaternion w,x,y,z) ───────────────────────────────────────
    key, raw_rot = _get(ckpt, ROTATION_KEYS, 'rotations')
    if raw_rot is None:
        print("  Using identity quaternions")
        quats = torch.zeros(N, 4); quats[:, 0] = 1.0
    else:
        if raw_rot.dim() == 1:
            raw_rot = raw_rot.reshape(-1, 4)
        quats = torch.nn.functional.normalize(raw_rot, dim=-1)
        print(f"  Rotations from '{key}'")

    # ── Intensities ──────────────────────────────────────────────────────────
    key, raw_int = _get(ckpt, INTENSITY_KEYS, 'log_amplitudes')
    if raw_int is None:
        print("  Using constant intensity 1.0")
        intensities = torch.ones(N)
    else:
        if 'log' in (key or ''):
            intensities = torch.exp(raw_int)
        else:
            intensities = raw_int
        # Flatten multi-channel (e.g. SH dc term → mean across channels)
        if intensities.dim() > 1:
            intensities = intensities.reshape(N, -1).mean(dim=-1)
        intensities = intensities.squeeze()
        # Normalise to [0, 1]
        lo, hi = intensities.min(), intensities.max()
        if hi > lo:
            intensities = (intensities - lo) / (hi - lo)
        print(f"  Intensities from '{key}'  range [{lo:.4f}, {hi:.4f}] → normalised [0,1]")

    # ── Pack & write binary PLY ───────────────────────────────────────────────
    means_np  = means.numpy().astype(np.float32)      # (N,3)
    scales_np = scales.numpy().astype(np.float32)     # (N,3)
    quats_np  = quats.numpy().astype(np.float32)      # (N,4)
    ints_np   = intensities.numpy().astype(np.float32).reshape(N,1)  # (N,1)

    data = np.hstack([means_np, scales_np, quats_np, ints_np])  # (N,11)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "property float intensity\n"
        "end_header\n"
    ).encode('ascii')

    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())

    mb = os.path.getsize(output_path) / 1024**2
    print(f"\n── Saved  {output_path}  ({mb:.1f} MB)")
    print(f"   Bounding box:")
    print(f"     x = [{means_np[:,0].min():.4f}, {means_np[:,0].max():.4f}]")
    print(f"     y = [{means_np[:,1].min():.4f}, {means_np[:,1].max():.4f}]")
    print(f"     z = [{means_np[:,2].min():.4f}, {means_np[:,2].max():.4f}]")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python export_ply.py model.pt output.ply")
        sys.exit(1)
    export(sys.argv[1], sys.argv[2])
