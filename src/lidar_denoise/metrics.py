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


def _degenerate(a: np.ndarray, b: np.ndarray) -> bool:
    """True if either cloud is empty, making every distance undefined.

    A filter can legitimately empty a cloud (aggressive SOR on a sparse
    frame). Previously cKDTree returned inf here and the inf propagated
    silently into the averaged tables; such frames are now reported as NaN
    and excluded from means with an explicit count.
    """
    return len(a) == 0 or len(b) == 0


def _directed_nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point of a to its nearest neighbour in b."""
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1, workers=-1)
    return d


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer Distance: 0.5 * (mean NN(a->b) + mean NN(b->a))."""
    if _degenerate(a, b):
        return float("nan")
    return float(0.5 * (_directed_nn(a, b).mean() + _directed_nn(b, a).mean()))


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff Distance: max of the directed maxima."""
    if _degenerate(a, b):
        return float("nan")
    return float(max(_directed_nn(a, b).max(), _directed_nn(b, a).max()))


def hausdorff95(a: np.ndarray, b: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff Distance (robust variant).

    Preferred over the strict HD in the report: a single surviving outlier
    at the range gate pins strict HD near 30 m for every method, which is
    why the raw HD column carries almost no signal.
    """
    if _degenerate(a, b):
        return float("nan")
    return float(
        max(np.percentile(_directed_nn(a, b), 95), np.percentile(_directed_nn(b, a), 95))
    )


def oracle_floor(clean: np.ndarray, noisy: np.ndarray, labels: np.ndarray) -> dict:
    """Best achievable scores for a perfect noise classifier.

    Deleting every injected point (labels == 1) and keeping every real one
    is the ceiling for any removal-based method. It is NOT zero error: the
    LISA weather model *deletes* returns whose attenuated power falls below
    the detection threshold, and no denoiser can put those back, while
    surviving points keep their Gaussian range jitter. Reporting this floor
    stops a method being credited for error it could never have removed.
    """
    real = noisy[labels == 0]
    return {
        "cd": chamfer_distance(clean, real),
        "hd95": hausdorff95(clean, real),
        "retention_pct": 100.0 * len(real) / len(noisy) if len(noisy) else float("nan"),
        "points_lost_to_weather": int(len(clean) - len(real)),
    }


def removal_scores(kept_indices: np.ndarray, labels: np.ndarray) -> dict:
    """Precision/recall/F1 of noise removal against the injected-noise labels.

    Defined for any method that yields an index set into the noisy cloud —
    the statistical filters by construction, and the DGCNN via its outlier
    head. Complements the geometric metrics, which cannot distinguish
    "removed the noise" from "moved everything slightly".
    """
    removed = np.ones(len(labels), dtype=bool)
    removed[kept_indices] = False
    noise = labels.astype(bool)
    tp = int((removed & noise).sum())
    fp = int((removed & ~noise).sum())
    fn = int((~removed & noise).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"removal_precision": precision, "removal_recall": recall, "removal_f1": f1}


def evaluate_cloud(clean: np.ndarray, denoised: np.ndarray, noisy: np.ndarray) -> dict:
    """All reconstruction metrics for one frame, plus relative improvement."""
    cd_noisy = chamfer_distance(clean, noisy)
    hd_noisy = hausdorff_distance(clean, noisy)
    cd = chamfer_distance(clean, denoised)
    hd = hausdorff_distance(clean, denoised)
    finite = np.isfinite([cd_noisy, cd]).all()
    return {
        "cd": cd,
        "hd": hd,
        "hd95": hausdorff95(clean, denoised),
        "cd_noisy": cd_noisy,
        "hd_noisy": hd_noisy,
        "cd_improvement_pct": (
            100.0 * (cd_noisy - cd) / cd_noisy if finite and cd_noisy else float("nan")
        ),
        "hd_improvement_pct": (
            100.0 * (hd_noisy - hd) / hd_noisy
            if np.isfinite([hd_noisy, hd]).all() and hd_noisy
            else float("nan")
        ),
        "retention_pct": 100.0 * len(denoised) / len(noisy) if len(noisy) else float("nan"),
        "n_points": int(len(denoised)),
        "degenerate": bool(len(denoised) == 0),
    }
