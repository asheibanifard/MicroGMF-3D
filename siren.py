from __future__ import annotations

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Sinusoidal Representation Network for 3-D scalar field reconstruction.

    Architecture (default): 3 → [202] × 3 → 1  ≈ 0.316 MB (float32)
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 202,
        hidden_layers: int = 3,
        out_features: int = 1,
        omega_0: float = 30.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0)
        ]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / omega_0
            self.final.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 ** 2


class ReLUMLP(nn.Module):
    """
    ReLU MLP for 3-D scalar field reconstruction.

    Architecture (default): 3 → [202] × 3 → 1  ≈ 0.316 MB (float32)
    Identical structure to SIREN — isolates the effect of sine vs ReLU activation.
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 202,
        hidden_layers: int = 3,
        out_features: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 ** 2


class FourierFeatureINR(nn.Module):
    """
    Fourier-feature INR with a ReLU MLP head.

    Default architecture targets a similar parameter budget to MicroGMF:
    input(3) -> Fourier features(63) -> [179] x 3 -> 1  (~0.290 MB float32)
    """

    def __init__(
        self,
        in_features: int = 3,
        num_fourier_features: int = 30,
        sigma: float = 10.0,
        hidden_features: int = 179,
        hidden_layers: int = 3,
        out_features: int = 1,
        include_input: bool = True,
    ):
        super().__init__()
        self.include_input = include_input
        self.register_buffer("B", torch.randn(in_features, num_fourier_features) * sigma)

        ff_dim = 2 * num_fourier_features + (in_features if include_input else 0)
        layers: list[nn.Module] = [nn.Linear(ff_dim, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_features, out_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(self.encode(x))).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 ** 2
