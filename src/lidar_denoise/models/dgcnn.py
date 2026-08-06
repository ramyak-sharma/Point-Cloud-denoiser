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


def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-NN indices (B, N, k) for points/features x: (B, C, N), self excluded."""
    # indices are discrete so no grad is needed here (also avoids a
    # cdist-backward bug on the MPS backend)
    with torch.no_grad():
        d = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))  # (B, N, N)
        return d.topk(k + 1, largest=False).indices[:, :, 1:]  # skip self


def knn_graph_features(x: torch.Tensor, k: int, idx: torch.Tensor | None = None) -> torch.Tensor:
    """EdgeConv feature construction: concat(x_i, x_j - x_i) over k-NN.

    x: (B, C, N) -> (B, 2C, N, k). If `idx` is given, that fixed graph is
    used instead of recomputing k-NN in feature space (static-graph mode —
    the main inference-latency optimisation, since neighbourhood search
    dominates EdgeConv runtime).
    """
    b, c, n = x.shape
    if idx is None:
        idx = knn_indices(x, k)
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

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        feat = knn_graph_features(x, self.k, idx)
        return self.conv(feat).max(dim=3).values              # (B, out_dim, N)


class DGCNNDenoiser(nn.Module):
    """DGCNN with two graph modes.

    static_graph=True computes the k-NN graph ONCE from the input XYZ
    coordinates and reuses it in every EdgeConv layer. This trades DGCNN's
    dynamic feature-space grouping for a 2-3x inference speedup — the graph
    build is the latency bottleneck — and is the configuration used for the
    real-time results. static_graph=False is the original dynamic behaviour.
    """

    def __init__(
        self,
        k: int = 16,
        hidden_dims: list[int] | None = None,
        emb_dim: int = 256,
        static_graph: bool = True,
    ):
        super().__init__()
        self.static_graph = static_graph
        self.k = k
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
        idx = knn_indices(feat, self.k) if self.static_graph else None
        layer_outs = []
        for conv in self.convs:
            feat = conv(feat, idx)
            layer_outs.append(feat)
        local = torch.cat(layer_outs, dim=1)                  # (B, sum(h), N)
        global_feat = self.fuse(local).max(dim=2, keepdim=True).values
        fused = torch.cat([local, global_feat.expand(-1, -1, local.shape[2])], dim=1)
        displacement = self.displacement_head(fused).transpose(1, 2)   # (B, N, 3)
        logits = self.outlier_head(fused).squeeze(1)                   # (B, N)
        return x + displacement, logits
