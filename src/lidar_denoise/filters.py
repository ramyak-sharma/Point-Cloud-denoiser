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
from scipy.spatial import cKDTree


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

    The timer spans the NumPy -> Open3D conversion as well as the filter
    itself: the learning-based methods are timed end-to-end from a NumPy
    array, so excluding the conversion here would flatter the baseline in
    the latency table.
    """
    t0 = time.perf_counter()
    pcd = _to_pcd(points)
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    idx = np.asarray(idx)
    kept = points[idx]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return kept, idx, elapsed_ms


def ror_filter(
    points: np.ndarray, nb_points: int = 4, radius: float = 0.6
) -> tuple[np.ndarray, np.ndarray, float]:
    """Radius Outlier Removal.

    Removes points with fewer than nb_points neighbours within radius metres.
    Returns (kept points, kept indices, elapsed ms). Timed end-to-end, as
    in `sor_filter`.
    """
    t0 = time.perf_counter()
    pcd = _to_pcd(points)
    _, idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    idx = np.asarray(idx)
    kept = points[idx]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return kept, idx, elapsed_ms


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


def idsor_filter(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 1.5,
    range_beta: float = 3.0,
    range_ref: float = 30.0,
    intensity_gamma: float = 2.0,
    intensity_floor: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Intensity- and Distance-aware Statistical Outlier Removal.

    Reimplementation of IDSOR [arXiv:2602.05876, 2026]. Plain SOR applies one
    global threshold ``mu + std_ratio * sigma`` to every point's mean k-NN
    distance, which is wrong in two opposite directions at once:

    * **Range.** A LiDAR's angular resolution is fixed, so genuine returns
      grow further apart with distance. A global threshold therefore deletes
      real far-field structure — the "erodes valid structure at long range"
      failure the adverse-weather literature repeatedly notes. The threshold
      is scaled by ``1 + range_beta * r / range_ref`` to stay permissive out
      there.
    * **Intensity.** Weather backscatter returns far less energy than a solid
      surface, so a weak return is more likely to be noise regardless of its
      geometry. The threshold is multiplied by a factor rising with
      normalised intensity, tightening it for weak returns.

    Requires (N, 4) XYZI input; use `sor_filter` for geometry-only clouds.
    Returns (kept points, kept indices, elapsed ms).

    Note: the paper's accessible text describes the mechanism and names the
    coefficients but does not give their values, so the shape parameters are
    exposed here and tuned by grid search on the **validation** split (never
    the test split). The defaults below were selected as the CD optimum among
    configurations that also beat the SOR+ROR baseline on removal F1; on
    validation they give CD 0.189 vs 0.259 and F1 0.653 vs 0.631, at higher
    precision (0.895 vs 0.763) and higher retention (82.0% vs 77.9%).
    ``range_beta`` sat at the edge of the search grid, so a wider sweep may
    do slightly better still.
    """
    if points.shape[1] < 4:
        raise ValueError("idsor_filter needs (N, 4) XYZI input")
    t0 = time.perf_counter()
    xyz, intensity = points[:, :3], points[:, 3]

    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=nb_neighbors + 1, workers=-1)
    mean_d = d[:, 1:].mean(axis=1)                       # exclude self
    r = np.linalg.norm(xyz, axis=1)

    # Range is handled by NORMALISING the statistic, not by inflating the
    # threshold. Neighbour spacing grows roughly linearly with range at fixed
    # angular resolution, so dividing it out makes mean_d comparable across
    # the scan and lets a single global mu/sigma be meaningful. (Scaling the
    # threshold instead makes it grow without bound with range, so far-field
    # points become unremovable — measured recall collapses to 0.04.)
    d_norm = mean_d / (1.0 + range_beta * r / max(range_ref, 1e-6))
    base = d_norm.mean() + std_ratio * d_norm.std()

    # Normalise intensity against a high percentile, not the max: a single
    # specular return would otherwise compress every other point toward zero.
    scale = np.percentile(intensity, 95)
    i_norm = np.clip(intensity / scale, 0.0, 1.0) if scale > 0 else np.zeros_like(intensity)
    f_intensity = (intensity_floor + (1.0 - intensity_floor) * i_norm) ** intensity_gamma

    idx = np.flatnonzero(d_norm <= base * f_intensity)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return points[idx], idx, elapsed_ms


def estimate_ego_motion(
    prev_points: np.ndarray, points: np.ndarray, max_correspondence: float = 2.0
) -> tuple[np.ndarray, float]:
    """Rigid transform aligning `prev_points` onto `points`, via point-to-point ICP.

    Returns (4x4 transform, ICP fitness). Identity is returned if either
    cloud is too small for a stable fit.
    """
    if len(prev_points) < 100 or len(points) < 100:
        return np.eye(4), 0.0
    reg = o3d.pipelines.registration.registration_icp(
        _to_pcd(prev_points),
        _to_pcd(points),
        max_correspondence,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return np.asarray(reg.transformation), float(reg.fitness)


def temporal_filter(
    points: np.ndarray,
    prev_points: np.ndarray,
    radius: float = 0.5,
    compensate_ego_motion: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Temporal-consistency filter using the previous scan of the sequence.

    Real structure persists between consecutive LiDAR sweeps (100 ms apart)
    while weather backscatter does not — the cue exploited by 4DenoiseNet
    [Seppänen et al., RA-L 2023]. A point is kept iff it has at least one
    neighbour within `radius` metres in the previous scan (which is itself
    noisy — no clean data is assumed).

    Ego-motion MUST be compensated first. Measured over these KITTI drives
    the inter-sweep translation is 0.56 m on average and up to 0.94 m
    (3-9 m/s), so on half the frames it exceeds `radius` outright: without
    compensation the filter deletes static structure wholesale rather than
    backscatter. The previous sweep is therefore registered onto the current
    one with ICP (`estimate_ego_motion`) before the consistency test. Set
    compensate_ego_motion=False to reproduce the uncompensated ablation.

    Returns (kept points, kept indices, elapsed ms).
    """
    t0 = time.perf_counter()
    ref = prev_points
    if compensate_ego_motion:
        transform, _ = estimate_ego_motion(prev_points, points)
        ref = prev_points @ transform[:3, :3].T + transform[:3, 3]
    tree = cKDTree(ref)
    d, _ = tree.query(points, k=1, workers=-1, distance_upper_bound=radius)
    idx = np.flatnonzero(np.isfinite(d))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return points[idx], idx, elapsed_ms


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
