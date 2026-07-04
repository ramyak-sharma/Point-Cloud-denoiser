"""Full-cloud inference for the learning-based denoisers.

A whole KITTI scan (~20-40k points after preprocessing) cannot be fed to the
patch networks directly, so the cloud is covered with overlapping k-NN
patches (overlap_factor patches per point on average). Per-point outputs
from overlapping patches are averaged — the same recombination strategy used
by score-denoise and PointCleanNet.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .models import DGCNNDenoiser, PointNetAutoencoder


def load_checkpoint(model_name: str, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg, k = ckpt["config"], ckpt["k"]
    if model_name == "pointnet_ae":
        model = PointNetAutoencoder(k, mcfg["latent_dim"], mcfg["decoder_hidden"])
    else:
        model = DGCNNDenoiser(mcfg["k"], mcfg["hidden_dims"], mcfg["emb_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, k


def _cover_with_patches(
    points: np.ndarray, points_per_patch: int, overlap_factor: float, rng: np.random.Generator
) -> np.ndarray:
    """Patch index matrix (P, k) covering the cloud with overlapping patches."""
    n_patches = max(1, int(np.ceil(len(points) * overlap_factor / points_per_patch)))
    n_patches = min(n_patches, len(points))
    centres = points[rng.choice(len(points), n_patches, replace=False)]
    tree = cKDTree(points)
    _, idx = tree.query(centres, k=min(points_per_patch, len(points)), workers=-1)
    return np.atleast_2d(idx)


@torch.no_grad()
def denoise_cloud(
    model,
    model_name: str,
    points: np.ndarray,
    cfg: dict,
    device: torch.device,
    batch_size: int = 64,
    outlier_threshold: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Denoise a full cloud; returns (denoised points, elapsed ms).

    DGCNN: per-point outputs are index-aligned with the input, so displaced
    positions and outlier probabilities from overlapping patches are averaged
    per original point; points whose mean outlier probability exceeds the
    threshold are removed.
    PointNet AE: the decoder emits an *unordered* set (no correspondence to
    input indices), so the union of all reconstructed patches is taken and
    deduplicated with a voxel grid at the preprocessing resolution.
    """
    pcfg = cfg["patches"]
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    patch_idx = _cover_with_patches(
        points, pcfg["points_per_patch"], pcfg["overlap_factor"], rng
    )
    patches = points[patch_idx]                                     # (P, k, 3)
    centroids = patches.mean(axis=1, keepdims=True)
    radii = np.linalg.norm(patches - centroids, axis=2).max(axis=1)
    radii = np.maximum(radii, 1e-6)[:, None, None]
    norm = ((patches - centroids) / radii).astype(np.float32)

    # accumulators over the original point index space (DGCNN path)
    pos_sum = np.zeros_like(points, dtype=np.float64)
    prob_sum = np.zeros(len(points), dtype=np.float64)
    count = np.zeros(len(points), dtype=np.float64)
    ae_reconstructions: list[np.ndarray] = []       # PointNet AE path

    for start in range(0, len(norm), batch_size):
        batch = torch.from_numpy(norm[start : start + batch_size]).to(device)
        b_idx = patch_idx[start : start + batch_size]
        b_centroids = centroids[start : start + batch_size]
        b_radii = radii[start : start + batch_size]
        if model_name == "pointnet_ae":
            out = model(batch).cpu().numpy()
            ae_reconstructions.append((out * b_radii + b_centroids).reshape(-1, 3))
        else:
            out_t, logits = model(batch)
            out = out_t.cpu().numpy()
            probs = torch.sigmoid(logits).cpu().numpy()
            restored = out * b_radii + b_centroids                  # back to metres
            np.add.at(pos_sum, b_idx.ravel(), restored.reshape(-1, 3))
            np.add.at(prob_sum, b_idx.ravel(), probs.ravel())
            np.add.at(count, b_idx.ravel(), 1.0)

    if model_name == "pointnet_ae":
        from .preprocessing import voxel_downsample

        union = np.vstack(ae_reconstructions).astype(np.float32)
        denoised = voxel_downsample(union, cfg["preprocessing"]["voxel_size"])
    else:
        covered = count > 0
        denoised = points.astype(np.float64).copy()
        denoised[covered] = pos_sum[covered] / count[covered, None]
        mean_prob = np.zeros(len(points))
        mean_prob[covered] = prob_sum[covered] / count[covered]
        denoised = denoised[mean_prob < outlier_threshold]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return denoised.astype(np.float32), elapsed_ms
