"""
Export GMF checkpoint → .splat binary for gsplat.js

.splat format (32 bytes per Gaussian):
  bytes  0-11 : position xyz         float32 × 3
  bytes 12-23 : scale    xyz         float32 × 3
  bytes 24-27 : colour   rgba        uint8   × 4
  bytes 28-31 : rotation wxyz        uint8   × 4   (mapped from [-1,1] → [0,255])

Usage:
  python export_splat.py [--ckpt PATH] [--out PATH] [--color R G B]
"""

import argparse
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F


def export_splat(ckpt_path: str, out_path: str, rgb=(0.2, 1.0, 0.5)):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    means   = ckpt["means"].float()        # (K, 3)  x y z in [-1,1]
    scales  = torch.exp(ckpt["log_scales"]).float()   # (K, 3)
    quats   = F.normalize(ckpt["quaternions"].float(), p=2, dim=-1)  # (K, 4) wxyz
    amps    = torch.exp(ckpt["log_amplitudes"]).float()               # (K,)  in (0,1)

    K = means.shape[0]
    print(f"Gaussians : {K}")
    print(f"Scale     : {scales.min():.5f} – {scales.max():.5f}")
    print(f"Amplitude : {amps.min():.5f} – {amps.max():.5f}")

    # Colour: fixed fluorescence tint modulated by amplitude
    r = int(rgb[0] * 255)
    g = int(rgb[1] * 255)
    b = int(rgb[2] * 255)

    # Quaternion: gsplat.js viewer decodes as u8 / 127.5 - 1
    # so encode as u8 = (q + 1) * 127.5, clamped to [0, 255].
    # Our model stores [w, x, y, z], export directly in that order
    q_np = quats.numpy()                            # (K, 4)  w x y z
    q_u8 = np.clip((q_np + 1.0) * 127.5, 0, 255).astype(np.uint8)  # keep as w x y z

    # Opacity: apply gamma < 1 to lift dim Gaussians (median amp ~0.02 → gamma 0.25 → ~180/255)
    # Raw linear mapping (amp*255) leaves median at ~5 which is invisible in the splatting renderer.
    # Lower gamma = brighter output (0.25 brightens more than 0.35)
    gamma = 0.25
    a_u8 = np.clip(amps.numpy() ** gamma * 255, 0, 255).astype(np.uint8)

    # Assemble binary blob
    pos  = means.numpy().astype(np.float32)         # (K, 3)
    sc   = scales.numpy().astype(np.float32)        # (K, 3)

    buf = bytearray(K * 32)
    for i in range(K):
        base = i * 32
        struct.pack_into("fff", buf, base,      pos[i, 0], pos[i, 1], pos[i, 2])
        struct.pack_into("fff", buf, base + 12, sc[i, 0],  sc[i, 1],  sc[i, 2])
        struct.pack_into("BBBB", buf, base + 24, r, g, b, a_u8[i])
        struct.pack_into("BBBB", buf, base + 28,                      # wxyz
                         q_u8[i, 0], q_u8[i, 1], q_u8[i, 2], q_u8[i, 3])

    with open(out_path, "wb") as f:
        f.write(buf)

    size_kb = len(buf) / 1024
    print(f"Written   : {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  default="checkpoints/gmf_refined_best.pt")
    parser.add_argument("--out",   default="viewer/scene.splat")
    parser.add_argument("--color", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        metavar=("R", "G", "B"),
                        help="Fluorescence tint in linear [0,1] (default: white/grayscale)")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export_splat(args.ckpt, args.out, tuple(args.color))
