"""Step 1 of the proposal methodology: preprocessing.

Normalisation helpers, NaN/Inf removal, range gating, and voxel-grid
downsampling. All functions are pure NumPy/Open3D and operate on (N, 3)
float arrays in metres (KITTI sensor frame).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def remove_invalid(points: np.ndarray) -> np.ndarray:
    """Drop rows containing NaN or Inf coordinates."""
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def range_filter(points: np.ndarray, max_range: float, min_z: float | None = None) -> np.ndarray:
    """Keep points within max_range metres of the sensor (and above min_z if set).

    Far-field KITTI returns are extremely sparse; gating them stabilises both
    the voxel grid and the nearest-neighbour statistics used by SOR.
    """
    mask = np.linalg.norm(points, axis=1) <= max_range
    if min_z is not None:
        mask &= points[:, 2] >= min_z
    return points[mask]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsampling via Open3D (centroid per occupied voxel)."""
    if voxel_size <= 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down.points, dtype=np.float32)


def normalize_unit_sphere(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre the cloud and scale it into the unit sphere.

    Returns (normalised points, centroid, scale) so the transform is
    invertible. The main pipeline evaluates in metres; this is used for
    per-patch normalisation inside the learning models.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    scale = float(np.linalg.norm(centred, axis=1).max())
    scale = max(scale, 1e-12)
    return centred / scale, centroid, scale


def preprocess(points: np.ndarray, cfg: dict) -> np.ndarray:
    """Full Step-1 chain: NaN/Inf removal -> range gate -> voxel downsample."""
    pre = cfg["preprocessing"]
    points = remove_invalid(points)
    points = range_filter(points, pre["max_range"], pre.get("min_z"))
    points = voxel_downsample(points, pre["voxel_size"])
    return points
