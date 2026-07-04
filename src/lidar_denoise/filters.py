"""Step 3(a): statistical filtering baselines (PCL conventions via Open3D).

Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR), exactly
as characterised in Section 3.1 of the proposal. Both run in O(n log n)
using a k-d tree and return the surviving index set so precision/recall
against injected-noise labels can be computed.
"""

from __future__ import annotations

import time

import numpy as np
import open3d as o3d


def _to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def sor_filter(
    points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Statistical Outlier Removal.

    Removes points whose mean k-NN distance exceeds mu + std_ratio * sigma
    (global statistics). Returns (kept points, kept indices, elapsed ms).
    """
    pcd = _to_pcd(points)
    t0 = time.perf_counter()
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    idx = np.asarray(idx)
    return points[idx], idx, elapsed_ms


def ror_filter(
    points: np.ndarray, nb_points: int = 4, radius: float = 0.6
) -> tuple[np.ndarray, np.ndarray, float]:
    """Radius Outlier Removal.

    Removes points with fewer than nb_points neighbours within radius metres.
    Returns (kept points, kept indices, elapsed ms).
    """
    pcd = _to_pcd(points)
    t0 = time.perf_counter()
    _, idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    idx = np.asarray(idx)
    return points[idx], idx, elapsed_ms


def sor_ror_pipeline(points: np.ndarray, cfg: dict) -> dict:
    """SOR followed by ROR with config parameters; the statistical baseline.

    Returns a dict with the denoised cloud, kept indices into the input,
    and per-stage timings.
    """
    fcfg = cfg["filters"]
    sor_pts, sor_idx, sor_ms = sor_filter(points, **fcfg["sor"])
    ror_pts, ror_idx, ror_ms = ror_filter(sor_pts, **fcfg["ror"])
    return {
        "points": ror_pts,
        "kept_indices": sor_idx[ror_idx],
        "sor_ms": sor_ms,
        "ror_ms": ror_ms,
        "total_ms": sor_ms + ror_ms,
    }


def removal_precision_recall(
    kept_indices: np.ndarray, noise_labels: np.ndarray
) -> dict[str, float]:
    """How well a filter's removals coincide with the injected-noise labels.

    precision = removed points that were actually noise / all removed
    recall    = noise points removed / all noise points
    """
    n = len(noise_labels)
    removed = np.ones(n, dtype=bool)
    removed[kept_indices] = False
    noise = noise_labels.astype(bool)
    tp = int((removed & noise).sum())
    fp = int((removed & ~noise).sum())
    fn = int((~removed & noise).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall}
