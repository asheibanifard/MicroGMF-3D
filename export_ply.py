"""
Export GMF checkpoint → Stanford PLY format with Anisotropic Scaling
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def export_ply(ckpt_path: str, out_path: str, dims=(813, 647, 100)):
    """Export Gaussian model to PLY format with coordinate restoration."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    means = ckpt["means"].float()  # (K, 3)  x y z in [-1, 1]
    scales = torch.exp(ckpt["log_scales"]).float()  # (K, 3)
    quats = F.normalize(ckpt["quaternions"].float(), p=2, dim=-1)  # (K, 4) wxyz
    amps = torch.exp(ckpt["log_amplitudes"]).float()  # (K,) in (0, 1)

    K = means.shape[0]
    print(f"Gaussians : {K}")
    print(f"Dimensions: {dims}")
    
    # Scale factors (from [-1, 1] to physical voxel counts)
    # The range [-1, 1] has length 2. The range [0, dim] has length dim.
    # Scale factor = dim / 2.
    kx, ky, kz = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0
    
    # NEW: Handle anisotropy properly for both positions AND individual splat shapes
    # sigma' = K * sigma * K^T
    # We construct K diagonal matrix
    K_mat = torch.diag(torch.tensor([kx, ky, kz], dtype=torch.float32))

    # Build PLY header
    properties = [
        "x", "y", "z",  # position
        "nx", "ny", "nz",  # normals (placeholder, all 0)
        "opacity",  # opacity from amplitude
        "scale_0", "scale_1", "scale_2",  # scale (log-scaled)
        "rot_0", "rot_1", "rot_2", "rot_3",  # quaternion wxyz
    ]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {K}\n"
    for prop in properties:
        header += f"property float {prop}\n"
    header += "end_header\n"

    header_bytes = header.encode("utf-8")
    num_properties = len(properties)
    data_size = K * num_properties * 4
    output = bytearray(len(header_bytes) + data_size)
    output[:len(header_bytes)] = header_bytes
    offset = len(header_bytes)

    print("Scaling and packing...")

    # For math consistency, calculate the rotation matrices
    def quat_to_rotmat(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        R = torch.zeros(3, 3)
        R[0, 0] = 1 - 2 * (y * y + z * z)
        R[0, 1] = 2 * (x * y - w * z)
        R[0, 2] = 2 * (x * z + w * y)
        R[1, 0] = 2 * (x * y + w * z)
        R[1, 1] = 1 - 2 * (x * x + z * z)
        R[1, 2] = 2 * (y * z - w * x)
        R[2, 0] = 2 * (x * z - w * y)
        R[2, 1] = 2 * (y * z + w * x)
        R[2, 2] = 1 - 2 * (x * x + y * y)
        return R

    for i in range(K):
        # 1. Scale Position: [-1, 1] -> [0, dim]
        # x_phys = (x_norm + 1)/2 * dim
        # Or centered: x_phys = x_norm * (dim/2)
        pos = means[i] * torch.tensor([kx, ky, kz])
        
        # 2. Scale Covariance to maintain aspect ratio
        # We need to extract new scales and rotations from K * R * S
        R = quat_to_rotmat(quats[i])
        S = torch.diag(scales[i])
        # M = K * R * S
        M = K_mat @ R @ S
        
        # SVD: M = U * Sigma * V^T
        # But for 3DGS, we want Sigma' = M * M^T = R' * S'^2 * R'^T
        # So Sigma' = (K R S) (K R S)^T = K R S^2 R^T K^T
        Sigma_prime = M @ M.T
        
        # Eigen decomposition of Sigma_prime
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_prime)
        # eigenvalues = s'^2
        new_scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-8))
        # Ensure descending order or consistent mapping if needed, but 3DGS doesn't care about order of local axes
        
        # Reconstruction of R': eigenvectors form the columns of R'
        # Check determinant to ensure a rotation (not reflection)
        R_prime = eigenvectors
        if torch.linalg.det(R_prime) < 0:
            R_prime[:, 0] *= -1
            
        # Convert R' back to Quaternion wxyz
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        tr = torch.trace(R_prime)
        if tr > 0:
            s_q = torch.sqrt(tr + 1.0) * 2
            qw = 0.25 * s_q
            qx = (R_prime[2, 1] - R_prime[1, 2]) / s_q
            qy = (R_prime[0, 2] - R_prime[2, 0]) / s_q
            qz = (R_prime[1, 0] - R_prime[0, 1]) / s_q
        else:
            if (R_prime[0, 0] > R_prime[1, 1]) and (R_prime[0, 0] > R_prime[2, 2]):
                s_q = torch.sqrt(1.0 + R_prime[0, 0] - R_prime[1, 1] - R_prime[2, 2]) * 2
                qw = (R_prime[2, 1] - R_prime[1, 2]) / s_q
                qx = 0.25 * s_q
                qy = (R_prime[0, 1] + R_prime[1, 0]) / s_q
                qz = (R_prime[0, 2] + R_prime[2, 0]) / s_q
            elif R_prime[1, 1] > R_prime[2, 2]:
                s_q = torch.sqrt(1.0 + R_prime[1, 1] - R_prime[0, 0] - R_prime[2, 2]) * 2
                qw = (R_prime[0, 2] - R_prime[2, 0]) / s_q
                qx = (R_prime[0, 1] + R_prime[1, 0]) / s_q
                qy = 0.25 * s_q
                qz = (R_prime[1, 2] + R_prime[2, 1]) / s_q
            else:
                s_q = torch.sqrt(1.0 + R_prime[2, 2] - R_prime[0, 0] - R_prime[1, 1]) * 2
                qw = (R_prime[1, 0] - R_prime[0, 1]) / s_q
                qx = (R_prime[0, 2] + R_prime[2, 0]) / s_q
                qy = (R_prime[1, 2] + R_prime[2, 1]) / s_q
                qz = 0.25 * s_q
        
        q_final = torch.tensor([qw, qx, qy, qz])

        # 3. Opacity (un-normalize sigmoid)
        opacity = np.log(amps[i].item() / (1.0 - amps[i].item() + 1e-8))

        vertex_data = [
            pos[0].item(), pos[1].item(), pos[2].item(),
            0.0, 0.0, 0.0,
            opacity,
            np.log(new_scales[0].item()), np.log(new_scales[1].item()), np.log(new_scales[2].item()),
            q_final[0].item(), q_final[1].item(), q_final[2].item(), q_final[3].item()
        ]

        for j, val in enumerate(vertex_data):
            struct.pack_into("f", output, offset + (i * num_properties + j) * 4, float(val))

    with open(out_path, "wb") as f:
        f.write(output)

    size_mb = len(output) / (1024 * 1024)
    print(f"Written   : {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/gmf_refined_best.pt")
    parser.add_argument("--out", default="viewer/scene.ply")
    parser.add_argument("--dims", default="813,647,100", help="X,Y,Z dimensions")
    args = parser.parse_args()

    dims = tuple(int(d) for d in args.dims.split(","))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_ply(args.ckpt, args.out, dims=dims)
