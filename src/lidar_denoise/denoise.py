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
        model = DGCNNDenoiser(mcfg["k"], mcfg["hidden_dims"], mcfg["emb_dim"],
                              mcfg.get("static_graph", True))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, k


def load_range_checkpoint(ckpt_path: Path, device: torch.device):
    """Load a trained RangeRestoreNet plus the config it was trained with."""
    from .models.range_unet import RangeRestoreNet

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    rcfg = ckpt["config"]
    model = RangeRestoreNet(base=rcfg["base"], depth=rcfg["depth"],
                            max_correction=rcfg.get("max_correction", 0.5))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, rcfg


@torch.no_grad()
def denoise_cloud_range(
    model,
    rcfg: dict,
    points: np.ndarray,
    cfg: dict,
    device: torch.device,
    outlier_threshold: float = 0.5,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Denoise a full cloud with the dual-head range-image network.

    Both heads are applied: predicted-noise points are removed, and the
    surviving points are moved along their own ray by the predicted range
    residual. Points that lost a pixel collision inherit their occluder's
    prediction (see `range_image.unproject_labels`), so every input point
    receives a decision.

    Returns (denoised points, elapsed ms, kept indices into `points`).
    """
    from . import range_image as RI

    max_range = cfg["preprocessing"]["max_range"]
    t0 = time.perf_counter()

    ri = RI.project(points, height=rcfg["height"], width=rcfg["width"], max_range=max_range)
    x = torch.from_numpy(ri.image).unsqueeze(0).to(device)
    logits, residual = model(x)
    prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    # residual is a bounded METRE correction; the image plane is normalised
    corrected = ri.image[0] * max_range + residual.squeeze(0).cpu().numpy()

    point_prob = RI.unproject_labels(ri, prob, fill=0.0)
    point_range = RI.unproject_labels(ri, corrected, fill=0.0)

    xyz = points[:, :3]
    r = np.maximum(np.linalg.norm(xyz, axis=1), 1e-6)
    # Move each point along its own ray to the predicted range. Points with
    # no pixel of their own keep their original range rather than adopting a
    # correction derived from a different return.
    scale = np.where(ri.represented, point_range / r, 1.0)
    restored = xyz * scale[:, None]

    kept_idx = np.flatnonzero(point_prob < outlier_threshold)
    out = restored[kept_idx].astype(np.float32)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return out, elapsed_ms, kept_idx


def _cover_with_patches(
    points: np.ndarray,
    points_per_patch: int,
    overlap_factor: float,
    rng: np.random.Generator,
    ensure_coverage: bool = True,
) -> np.ndarray:
    """Patch index matrix (P, k) covering the cloud with overlapping patches.

    Random centres alone do NOT cover every point: with overlap_factor 3 a
    few percent of points fall outside every patch. Those points used to
    pass through untouched with an outlier probability of zero, so the DGCNN
    could never remove an uncovered noise point regardless of how good its
    classifier was — a silent ceiling on its removal recall. Patches are
    therefore seeded greedily on the uncovered remainder until coverage is
    complete.
    """
    n = len(points)
    k = min(points_per_patch, n)
    n_patches = min(max(1, int(np.ceil(n * overlap_factor / points_per_patch))), n)
    tree = cKDTree(points)
    _, idx = tree.query(points[rng.choice(n, n_patches, replace=False)], k=k, workers=-1)
    idx = np.atleast_2d(idx)
    if not ensure_coverage:
        return idx

    covered = np.zeros(n, dtype=bool)
    covered[idx.ravel()] = True
    extra = []
    while not covered.all():
        remaining = np.flatnonzero(~covered)
        n_seeds = max(1, int(np.ceil(len(remaining) / k)))
        seeds = remaining[rng.permutation(len(remaining))[:n_seeds]]
        _, more = tree.query(points[seeds], k=k, workers=-1)
        more = np.atleast_2d(more)
        covered[more.ravel()] = True
        extra.append(more)
    return np.vstack([idx, *extra]) if extra else idx


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

    PointNet AE: the decoder emits an *unordered* set with no correspondence
    to input indices, so its reconstruction must be mapped back before the
    two models can be compared. Two strategies, set by
    ``patches.ae_recombination``:

    * ``nn_project`` (default) — each input point takes the position of its
      nearest point in the reconstruction of the same patch, then averages
      across overlapping patches exactly as the DGCNN path does. Output
      cardinality equals input cardinality.
    * ``union`` (legacy) — concatenate every patch reconstruction and voxel
      -downsample. Because each point is covered by ``overlap_factor``
      patches and every patch emits ``points_per_patch`` points, this
      returns ~1.7x more points than it was given, which inflates the
      Chamfer term from the denoised side and hands the detector a cloud of
      the wrong density. Retained only to reproduce the earlier numbers.

    The two differ by far more than a recombination detail should, so both
    are reported in the ablation rather than one being silently chosen.
    """
    pcfg = cfg["patches"]
    rng = np.random.default_rng(seed)
    recombination = pcfg.get("ae_recombination", "nn_project")
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
            restored = out * b_radii + b_centroids                  # (B, k, 3)
            if recombination == "union":
                ae_reconstructions.append(restored.reshape(-1, 3))
            else:
                # Map the unordered reconstruction back onto the patch's own
                # input points: each input point adopts its nearest
                # reconstructed point, then overlapping patches are averaged.
                src = points[b_idx]                                 # (B, k, 3)
                d = np.linalg.norm(src[:, :, None, :] - restored[:, None, :, :], axis=3)
                nearest = restored[np.arange(len(src))[:, None], d.argmin(axis=2)]
                np.add.at(pos_sum, b_idx.ravel(), nearest.reshape(-1, 3))
                np.add.at(count, b_idx.ravel(), 1.0)
        else:
            out_t, logits = model(batch)
            out = out_t.cpu().numpy()
            probs = torch.sigmoid(logits).cpu().numpy()
            restored = out * b_radii + b_centroids                  # back to metres
            np.add.at(pos_sum, b_idx.ravel(), restored.reshape(-1, 3))
            np.add.at(prob_sum, b_idx.ravel(), probs.ravel())
            np.add.at(count, b_idx.ravel(), 1.0)

    if model_name == "pointnet_ae" and recombination == "union":
        from .preprocessing import voxel_downsample

        union = np.vstack(ae_reconstructions).astype(np.float32)
        denoised = voxel_downsample(union, cfg["preprocessing"]["voxel_size"])
        kept_idx = None                       # no correspondence to the input
    else:
        covered = count > 0
        denoised = points.astype(np.float64).copy()
        denoised[covered] = pos_sum[covered] / count[covered, None]
        if model_name == "pointnet_ae":
            kept_idx = np.arange(len(points))  # displacement only, removes nothing
        else:
            mean_prob = np.zeros(len(points))
            mean_prob[covered] = prob_sum[covered] / count[covered]
            kept_idx = np.flatnonzero(mean_prob < outlier_threshold)
        denoised = denoised[kept_idx]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return denoised.astype(np.float32), elapsed_ms, kept_idx
