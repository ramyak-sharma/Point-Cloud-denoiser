"""Step 1 of the proposal methodology: preprocessing.

Normalisation helpers, NaN/Inf removal, range gating, and voxel-grid
downsampling. All functions are pure NumPy/Open3D and operate on (N, 3)
float arrays in metres (KITTI sensor frame).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


"""All functions below are shape-agnostic: they treat columns 0:3 as XYZ
geometry and carry any further columns (intensity, reflectivity, ...)
through unchanged, so the same chain serves both the geometry-only and the
intensity-aware pipelines."""


def remove_invalid(points: np.ndarray) -> np.ndarray:
    """Drop rows whose XYZ coordinates contain NaN or Inf."""
    mask = np.isfinite(points[:, :3]).all(axis=1)
    return points[mask]


def range_filter(points: np.ndarray, max_range: float, min_z: float | None = None) -> np.ndarray:
    """Keep points within max_range metres of the sensor (and above min_z if set).

    Far-field KITTI returns are extremely sparse; gating them stabilises both
    the voxel grid and the nearest-neighbour statistics used by SOR.
    """
    mask = np.linalg.norm(points[:, :3], axis=1) <= max_range
    if min_z is not None:
        mask &= points[:, 2] >= min_z
    return points[mask]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsampling via Open3D (centroid per occupied voxel).

    Extra channels are smuggled through Open3D's colour field, which is
    averaged per voxel exactly as the coordinates are — so an intensity
    column comes back as the voxel's mean intensity rather than being
    silently dropped. Open3D allows three colour channels, hence the cap.
    """
    if voxel_size <= 0:
        return points
    extra = points[:, 3:]
    if extra.shape[1] > 3:
        raise ValueError("voxel_downsample carries at most 3 extra channels")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    if extra.shape[1]:
        # Open3D treats the colour field as [0, 1]; KITTI intensity already
        # lives there, but clip so an out-of-range channel fails loudly in
        # value rather than silently at the Open3D boundary.
        padded = np.zeros((len(points), 3), dtype=np.float64)
        padded[:, : extra.shape[1]] = np.clip(extra, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(padded)
    down = pcd.voxel_down_sample(voxel_size)
    xyz = np.asarray(down.points, dtype=np.float32)
    if not extra.shape[1]:
        return xyz
    cols = np.asarray(down.colors, dtype=np.float32)[:, : extra.shape[1]]
    return np.hstack([xyz, cols])


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
