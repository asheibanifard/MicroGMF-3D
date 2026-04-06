# ===================================================================
#  Regularisers
# ===================================================================
import torch
import torch.functional as F
import torch.nn as nn

def tubular_regulariser(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encourage tubular shapes: penalise (λ1+λ2)/λ3 being large."""
    eig = torch.linalg.eigvalsh(Sigma.float())        # (K, 3) ascending
    eig = torch.sort(eig, dim=-1)[0]
    return ((eig[:, 0] + eig[:, 1]) / (eig[:, 2] + eps)).mean()


def cross_section_symmetry_reg(Sigma: torch.Tensor) -> torch.Tensor:
    """Encourage circular cross-sections: penalise |λ1 − λ2|."""
    eig = torch.linalg.eigvalsh(Sigma.float())
    eig = torch.sort(eig, dim=-1)[0]
    return (eig[:, 0] - eig[:, 1]).abs().mean()


def _gradient_magnitudes(param: nn.Parameter) -> torch.Tensor:
    if param.grad is None:
        return torch.zeros(param.shape[0], device=param.device)
    if param.grad.ndim > 1:
        return torch.norm(param.grad, dim=-1)
    return param.grad.abs()
