"""PointNet-based autoencoder denoiser (proposal Section 5, Step 3, option b-1).

Architecture follows PointNet [Qi et al., CVPR 2017]: shared per-point MLPs
followed by a symmetric max-pool produce a permutation-invariant latent code;
a fully connected decoder reconstructs the patch as an unordered point set.
Trained with Chamfer loss on (noisy patch -> clean patch) pairs, the
bottleneck discards the high-frequency component of the input — which is
exactly the injected noise — and reconstructs the underlying surface.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Shared MLP (3 -> 64 -> 128 -> latent) + max-pool, as in PointNet."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, latent_dim, 1), nn.BatchNorm1d(latent_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) -> latent (B, latent_dim)
        feat = self.mlp(x.transpose(1, 2))        # (B, C, N)
        return feat.max(dim=2).values             # symmetric aggregation


class PointNetAutoencoder(nn.Module):
    """Encoder + FC decoder reconstructing a fixed-size patch."""

    def __init__(self, num_points: int = 256, latent_dim: int = 256, decoder_hidden: int = 512):
        super().__init__()
        self.num_points = num_points
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden), nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden), nn.ReLU(),
            nn.Linear(decoder_hidden, num_points * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) normalised patch -> reconstructed (B, N, 3)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.reshape(-1, self.num_points, 3)


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batched squared-Chamfer loss between two (B, N, 3) point sets.

    O(N^2) pairwise distances are fine at patch scale (N = 256).
    """
    d = torch.cdist(pred, target)                       # (B, Np, Nt)
    return d.min(dim=2).values.pow(2).mean() + d.min(dim=1).values.pow(2).mean()
