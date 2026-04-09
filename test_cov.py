import torch
import numpy as np

def quat_to_rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.zeros((3, 3))
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

# A random quaternion
q = np.array([0.5, 0.4, 0.3, 0.71414284])
s = np.array([0.1, 0.2, 0.3])
R = quat_to_rotmat(q)
S2 = np.diag(s**2)
Sigma = R @ S2 @ R.T

print("Python Sigma:")
print(Sigma)
