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

    For every noisy point, the displacement target is the vector to its
    nearest clean point. Injected outliers get a *zero* displacement target
    (masked out of the displacement loss) because their correct treatment is
    removal, handled by the classification head.
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
    return {
        "noisy": ((patches - centroids) / radii).astype(np.float32),
        "clean": ((targets - centroids) / radii).astype(np.float32),
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
    """Loads a cached .npz of (noisy, clean, labels) patch triples."""

    def __init__(self, cache_path: Path):
        data = np.load(cache_path)
        self.noisy = torch.from_numpy(data["noisy"])
        self.clean = torch.from_numpy(data["clean"])
        self.labels = torch.from_numpy(data["labels"])

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, i: int):
        return self.noisy[i], self.clean[i], self.labels[i]
