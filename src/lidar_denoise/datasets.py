"""Paired (noisy, clean) patch dataset for the learning-based denoisers.

Following the patch-based convention of score-denoise / PointCleanNet:
fixed-size k-NN patches are extracted around centres sampled from the noisy
cloud, normalised to zero mean and unit radius, and paired with

* the patch's nearest clean points (targets for displacement / AE loss), and
* per-point injected-noise labels (targets for the outlier head).

Patches are precomputed per frame and cached as .npz shards so training does
not re-run noise injection every epoch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .kitti_io import load_bin
from .noise import inject_noise
from .preprocessing import preprocess


def extract_patch_pairs(
    clean: np.ndarray,
    noisy: np.ndarray,
    noise_labels: np.ndarray,
    points_per_patch: int,
    max_patches: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample k-NN patches from the noisy cloud with clean targets.

    Two distinct targets are produced, because the two models need
    different supervision:

    * ``clean`` — index-aligned: for every noisy point, its nearest clean
      point. This is the DGCNN's per-point displacement target. Injected
      outliers are masked out of the displacement loss (their correct
      treatment is removal, handled by the classification head), so the
      arbitrary projection an outlier receives here never contributes.

    * ``clean_set`` — the k clean points nearest the patch centroid, i.e. an
      actual clean patch. This is the AE's Chamfer target. It cannot be the
      index-aligned array above: that array projects *every* noisy point,
      outliers included, onto its nearest surface point, so ~15% of the
      target set is duplicate surface points clustered wherever the outliers
      happened to fall. Chamfer against that distribution teaches the AE to
      pile reconstruction mass onto those spots, which is a direct cause of
      the oversmoothed output the earlier runs showed.
    """
    n_patches = min(max_patches, max(1, len(noisy) // points_per_patch))
    centres = noisy[rng.choice(len(noisy), n_patches, replace=False)]

    noisy_tree = cKDTree(noisy)
    _, patch_idx = noisy_tree.query(centres, k=points_per_patch, workers=-1)

    clean_tree = cKDTree(clean)
    _, nn_clean = clean_tree.query(noisy, k=1, workers=-1)
    targets_full = clean[nn_clean]                    # (M, 3) NN-clean per noisy point

    patches = noisy[patch_idx]                        # (P, k, 3)
    targets = targets_full[patch_idx]                 # (P, k, 3)
    labels = noise_labels[patch_idx]                  # (P, k)

    # Per-patch normalisation: centre on the patch centroid, scale by radius.
    centroids = patches.mean(axis=1, keepdims=True)
    radii = np.linalg.norm(patches - centroids, axis=2).max(axis=1)
    radii = np.maximum(radii, 1e-6)[:, None, None]

    # True clean patch for the AE, gathered around the same centroids.
    k_clean = min(points_per_patch, len(clean))
    _, clean_idx = clean_tree.query(centroids[:, 0, :], k=k_clean, workers=-1)
    clean_set = clean[np.atleast_2d(clean_idx)]       # (P, k, 3)
    if k_clean < points_per_patch:                    # pad tiny frames by repetition
        pad = np.repeat(clean_set[:, -1:, :], points_per_patch - k_clean, axis=1)
        clean_set = np.concatenate([clean_set, pad], axis=1)

    return {
        "noisy": ((patches - centroids) / radii).astype(np.float32),
        "clean": ((targets - centroids) / radii).astype(np.float32),
        "clean_set": ((clean_set - centroids) / radii).astype(np.float32),
        "labels": labels.astype(np.float32),
    }


def build_patch_cache(
    scan_paths: list[Path], cfg: dict, cache_path: Path, seed: int = 0
) -> Path:
    """Preprocess + noise-inject every scan and store all patches in one .npz."""
    pcfg = cfg["patches"]
    rng = np.random.default_rng(seed)
    shards: list[dict[str, np.ndarray]] = []
    for i, path in enumerate(scan_paths):
        clean = preprocess(load_bin(path), cfg)
        noisy = inject_noise(clean, cfg, seed=seed + i)
        shards.append(
            extract_patch_pairs(
                clean, noisy.points, noisy.labels,
                pcfg["points_per_patch"], pcfg["max_patches_per_frame"], rng,
            )
        )
    merged = {k: np.concatenate([s[k] for s in shards]) for k in shards[0]}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **merged)
    return cache_path


class PatchDataset(Dataset):
    """Loads a cached .npz of (noisy, clean, clean_set, labels) patch tuples."""

    def __init__(self, cache_path: Path):
        data = np.load(cache_path)
        self.noisy = torch.from_numpy(data["noisy"])
        self.clean = torch.from_numpy(data["clean"])
        # Caches built before the AE-target fix lack clean_set; fall back to
        # the index-aligned targets so old caches still load (and warn).
        if "clean_set" in data:
            self.clean_set = torch.from_numpy(data["clean_set"])
        else:
            import warnings

            warnings.warn(
                f"{cache_path} predates the clean_set AE target; rebuild the "
                "cache (--stage cache) for correct PointNet-AE supervision.",
                stacklevel=2,
            )
            self.clean_set = self.clean
        self.labels = torch.from_numpy(data["labels"])

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, i: int):
        return self.noisy[i], self.clean[i], self.clean_set[i], self.labels[i]
