# ===================================================================
#  Main
# ===================================================================
import argparse
import os

import numpy as np
import torch

from model import GaussianMixtureField
from train import train
from utils import (
    load_config,
    load_swc,
    load_tif_data,
    save_gt_volume_beside_tif,
    swc_to_normalised_coords,
)


def main():
    parser = argparse.ArgumentParser(description="Train Gaussian Mixture Field")
    parser.add_argument("--config", default="config.yml", help="YAML config path")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 0))
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seed: {seed}")

    dev = cfg["training"].get("device", "auto")
    device = "cuda" if (dev == "auto" and torch.cuda.is_available()) else dev
    if device != "cuda":
        raise RuntimeError("CUDA required.  Set device: cuda in config.")

    src_tif_path = cfg["data"]["tif_path"]
    vol = load_tif_data(src_tif_path)
    gt_tif_path = save_gt_volume_beside_tif(src_tif_path, vol, suffix="_gt")
    print(f"Saved GT volume: {gt_tif_path}")

    mc = cfg["model"]

    # Load SWC morphology for Gaussian initialization (if provided)
    swc_coords, swc_radii = None, None
    swc_path = cfg["data"].get("swc_path")
    if swc_path and os.path.exists(swc_path):
        swc_data = load_swc(swc_path)
        swc_coords, swc_radii = swc_to_normalised_coords(
            swc_data, vol.shape, bounds=mc.get("bounds")
        )
        print(f"Loaded SWC: {swc_data.shape[0]} nodes from {swc_path}")
    elif swc_path:
        print(f"WARNING: swc_path={swc_path} not found, using random init")

    field = GaussianMixtureField(
        num_gaussians=int(mc["num_gaussians"]),
        init_scale=float(mc.get("init_scale", 0.05)),
        init_amplitude=float(mc.get("init_amplitude", 0.1)),
        bounds=mc.get("bounds"),
        aabb=mc.get("aabb"),
        swc_coords=swc_coords,
        swc_radii=swc_radii,
    )

    # Resume from checkpoint if specified
    resume_path = args.resume or cfg["training"].get("resume_from")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        # Handle checkpoint with different num_gaussians
        K_ckpt = ckpt["means"].shape[0]
        if K_ckpt != field.num_gaussians:
            print(f"Checkpoint has K={K_ckpt} (config has {field.num_gaussians}), adjusting...")
            field = GaussianMixtureField(
                num_gaussians=K_ckpt,
                init_scale=float(mc.get("init_scale", 0.05)),
                init_amplitude=float(mc.get("init_amplitude", 0.1)),
                bounds=mc.get("bounds"),
                aabb=mc.get("aabb"),
            )
        missing, unexpected = field.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"Checkpoint missing keys (using init values): {missing}")
        if unexpected:
            print(f"Checkpoint has unexpected keys: {unexpected}")
        print(f"Resumed from {resume_path} (K={field.num_gaussians})")
    elif resume_path:
        print(f"WARNING: resume_from={resume_path} not found, training from scratch")

    log_dir = cfg["training"].get("log_dir", "logs")
    field = train(field, vol, cfg, device=device, log_dir=log_dir)

    out = cfg["training"].get("save_path")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(field.state_dict(), out)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()