"""Lightweight DGCNN denoiser (proposal Section 5, Step 3, option b-2).

EdgeConv layers [Wang et al., DGCNN, TOG 2019] build a k-NN graph in feature
space and aggregate relative neighbour features, letting the network tell
isolated noise clusters (weakly connected subgraphs) from surface points.

Two output heads, mirroring how noise actually corrupts LiDAR data:
* displacement head — a per-point offset that undoes Gaussian jitter
  (residual formulation, as in score-based denoising);
* outlier head — a per-point logit classifying injected fog/outlier points,
  which are *removed* at inference (displacement alone cannot delete points).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def knn_graph_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """EdgeConv feature construction: concat(x_i, x_j - x_i) over k-NN.

    x: (B, C, N) -> (B, 2C, N, k)
    """
    b, c, n = x.shape
    # pairwise distances in feature space; indices are discrete so no grad is
    # needed here (also avoids a cdist-backward bug on the MPS backend)
    with torch.no_grad():
        d = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))  # (B, N, N)
        idx = d.topk(k + 1, largest=False).indices[:, :, 1:]   # (B, N, k), skip self
    idx_flat = idx.reshape(b, -1)                             # (B, N*k)
    neighbours = torch.gather(
        x.transpose(1, 2), 1, idx_flat.unsqueeze(-1).expand(-1, -1, c)
    ).reshape(b, n, k, c)                                        # (B, N, k, C)
    central = x.transpose(1, 2).unsqueeze(2).expand(-1, -1, k, -1)
    feat = torch.cat([central, neighbours - central], dim=3)  # (B, N, k, 2C)
    # contiguous() is required: BatchNorm2d's MPS backward cannot handle the
    # permuted (strided) layout and fails with a view/stride RuntimeError
    return feat.permute(0, 3, 1, 2).contiguous()              # (B, 2C, N, k)


class EdgeConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = knn_graph_features(x, self.k)
        return self.conv(feat).max(dim=3).values              # (B, out_dim, N)


class DGCNNDenoiser(nn.Module):
    def __init__(self, k: int = 16, hidden_dims: list[int] | None = None, emb_dim: int = 256):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64, 128]
        self.convs = nn.ModuleList()
        in_dim = 3
        for h in hidden_dims:
            self.convs.append(EdgeConv(in_dim, h, k))
            in_dim = h
        self.fuse = nn.Sequential(
            nn.Conv1d(sum(hidden_dims), emb_dim, 1), nn.BatchNorm1d(emb_dim), nn.LeakyReLU(0.2),
        )
        head_in = emb_dim + sum(hidden_dims)
        self.displacement_head = nn.Sequential(
            nn.Conv1d(head_in, 128, 1), nn.LeakyReLU(0.2), nn.Conv1d(128, 3, 1),
        )
        self.outlier_head = nn.Sequential(
            nn.Conv1d(head_in, 128, 1), nn.LeakyReLU(0.2), nn.Conv1d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, 3) -> (denoised points (B, N, 3), outlier logits (B, N))."""
        feat = x.transpose(1, 2)                              # (B, 3, N)
        layer_outs = []
        for conv in self.convs:
            feat = conv(feat)
            layer_outs.append(feat)
        local = torch.cat(layer_outs, dim=1)                  # (B, sum(h), N)
        global_feat = self.fuse(local).max(dim=2, keepdim=True).values
        fused = torch.cat([local, global_feat.expand(-1, -1, local.shape[2])], dim=1)
        displacement = self.displacement_head(fused).transpose(1, 2)   # (B, N, 3)
        logits = self.outlier_head(fused).squeeze(1)                   # (B, N)
        return x + displacement, logits
