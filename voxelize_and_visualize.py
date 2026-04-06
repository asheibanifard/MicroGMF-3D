#!/usr/bin/env python3
"""
Voxelize volume from trained Gaussian Mixture Field and visualize slices.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Setup paths
os.chdir('/mnt/intelpa-1/armin/containers/storage/main/FLUOR_GMM/hisnegs/main')
sys.path.insert(0, str(Path.cwd()))

from model import GaussianMixtureField
from utils import load_config

def main():
    # Configuration
    checkpoint_path = "checkpoints/gmf_refined_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}\n")
    
    # Load checkpoint
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    num_gaussians = state['means'].shape[0]
    
    print(f"✓ Loaded: {num_gaussians} Gaussians")
    
    # Load config and create model
    cfg = load_config("config.yml")
    model = GaussianMixtureField(
        num_gaussians=num_gaussians,
        bounds=cfg["model"]["bounds"],
    ).to(device)
    
    model.load_state_dict(state)
    model.eval()
    print("✓ Model initialized")
    
    # Define volume dimensions
    Z_size, Y_size, X_size = 100, 647, 813
    print(f"\nVoxelizing volume: {Z_size} × {Y_size} × {X_size} = {Z_size * Y_size * X_size:,} voxels")
    
    # Create voxel coordinate grid
    with torch.no_grad():
        # Map to [-1, 1]
        z_coords = torch.linspace(0, Z_size - 1, Z_size, device=device) / (Z_size - 1) * 2.0 - 1.0
        y_coords = torch.linspace(0, Y_size - 1, Y_size, device=device) / (Y_size - 1) * 2.0 - 1.0
        x_coords = torch.linspace(0, X_size - 1, X_size, device=device) / (X_size - 1) * 2.0 - 1.0
        
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        voxel_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # Reconstruct in batches
        batch_size = 8192
        reconstructed_vol = torch.zeros(Z_size * Y_size * X_size, device=device)
        
        print("\nReconstruction progress:")
        for batch_idx in range(0, len(voxel_coords), batch_size):
            batch_coords = voxel_coords[batch_idx:batch_idx + batch_size]
            batch_density = model(batch_coords, k_chunk=512)
            reconstructed_vol[batch_idx:batch_idx + len(batch_coords)] = batch_density.squeeze()
            
            progress = min(batch_idx + batch_size, len(voxel_coords))
            percent = 100.0 * progress / len(voxel_coords)
            print(f"  {percent:6.1f}% ({progress:,}/{len(voxel_coords):,})", end='\r')
        
        print(f"  100.0% ({len(voxel_coords):,}/{len(voxel_coords):,})  ")
        
        # Reshape to volume
        reconstructed_vol = reconstructed_vol.reshape(Z_size, Y_size, X_size)
        reconstructed_vol_np = reconstructed_vol.cpu().numpy()
    
    print("\n✓ Volume reconstruction complete")
    
    # Visualize slices
    print("\nVisualizing slices...")
    
    # Select evenly spaced slices
    num_slices = 8
    slice_indices = np.linspace(0, Z_size - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=100)
    axes = axes.flatten()
    
    for idx, z_idx in enumerate(slice_indices):
        slice_data = reconstructed_vol_np[z_idx]
        
        # Normalize for display
        slice_normalized = np.clip(slice_data / np.max(reconstructed_vol_np), 0, 1)
        
        im = axes[idx].imshow(slice_normalized, cmap='gray')
        axes[idx].set_title(f"Slice Z={z_idx}", fontsize=10)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig("test/voxelized_slices.png", dpi=100, bbox_inches='tight')
    print("✓ Saved: test/voxelized_slices.png")
    plt.show()
    
    # Print statistics
    print(f"\nVolume Statistics:")
    print(f"  Min:  {reconstructed_vol_np.min():.6f}")
    print(f"  Max:  {reconstructed_vol_np.max():.6f}")
    print(f"  Mean: {reconstructed_vol_np.mean():.6f}")
    print(f"  Std:  {reconstructed_vol_np.std():.6f}")

if __name__ == "__main__":
    main()
