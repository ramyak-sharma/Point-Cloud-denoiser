"""Step 4 metrics: Chamfer Distance and Hausdorff Distance.

Vectorised with scipy's cKDTree (parallel queries), replacing the original
prototype's per-point Python loop — ~100x faster on full KITTI scans.

Conventions (documented so numbers are comparable across the report):
* chamfer_distance: mean of the two directed mean-NN distances (metres).
* hausdorff_distance: max of the two directed maxima (metres).
* hausdorff95: robust variant using the 95th percentile, less dominated by
  a single residual outlier — reported alongside the strict HD.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _directed_nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point of a to its nearest neighbour in b."""
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1, workers=-1)
    return d


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer Distance: 0.5 * (mean NN(a->b) + mean NN(b->a))."""
    return float(0.5 * (_directed_nn(a, b).mean() + _directed_nn(b, a).mean()))


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff Distance: max of the directed maxima."""
    return float(max(_directed_nn(a, b).max(), _directed_nn(b, a).max()))


def hausdorff95(a: np.ndarray, b: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff Distance (robust variant)."""
    return float(
        max(np.percentile(_directed_nn(a, b), 95), np.percentile(_directed_nn(b, a), 95))
    )


def evaluate_cloud(clean: np.ndarray, denoised: np.ndarray, noisy: np.ndarray) -> dict:
    """All reconstruction metrics for one frame, plus relative improvement."""
    cd_noisy = chamfer_distance(clean, noisy)
    hd_noisy = hausdorff_distance(clean, noisy)
    cd = chamfer_distance(clean, denoised)
    hd = hausdorff_distance(clean, denoised)
    return {
        "cd": cd,
        "hd": hd,
        "hd95": hausdorff95(clean, denoised),
        "cd_noisy": cd_noisy,
        "hd_noisy": hd_noisy,
        "cd_improvement_pct": 100.0 * (cd_noisy - cd) / cd_noisy if cd_noisy else 0.0,
        "hd_improvement_pct": 100.0 * (hd_noisy - hd) / hd_noisy if hd_noisy else 0.0,
        "retention_pct": 100.0 * len(denoised) / len(noisy),
    }
