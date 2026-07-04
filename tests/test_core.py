"""Unit tests for the core preprocessing, noise, filter, and metric functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidar_denoise.config import load_config
from lidar_denoise.filters import removal_precision_recall, ror_filter, sor_filter
from lidar_denoise.metrics import chamfer_distance, hausdorff95, hausdorff_distance
from lidar_denoise.noise import fog_backscatter, gaussian_jitter, inject_noise, random_outliers
from lidar_denoise.preprocessing import (
    normalize_unit_sphere,
    range_filter,
    remove_invalid,
    voxel_downsample,
)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def plane_cloud():
    """A dense 2D grid on z=0 — a predictable 'surface' for filter tests."""
    g = np.linspace(-5, 5, 50)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1).astype(np.float32)


# ── preprocessing ───────────────────────────────────────────────────────────

def test_remove_invalid_drops_nan_and_inf():
    pts = np.array([[0, 0, 0], [np.nan, 1, 1], [1, np.inf, 1], [2, 2, 2]], dtype=np.float32)
    out = remove_invalid(pts)
    assert len(out) == 2
    assert np.isfinite(out).all()


def test_range_filter():
    pts = np.array([[1, 0, 0], [100, 0, 0], [0, 0, -3]], dtype=np.float32)
    out = range_filter(pts, max_range=60.0)
    assert len(out) == 2
    out = range_filter(pts, max_range=60.0, min_z=-1.0)
    assert len(out) == 1


def test_voxel_downsample_reduces_density(plane_cloud):
    out = voxel_downsample(plane_cloud, voxel_size=1.0)
    assert 0 < len(out) < len(plane_cloud)
    # zero voxel size is a no-op
    assert len(voxel_downsample(plane_cloud, 0)) == len(plane_cloud)


def test_normalize_unit_sphere_roundtrip():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(100, 3)).astype(np.float32) * 5 + 3
    normed, centroid, scale = normalize_unit_sphere(pts)
    assert np.linalg.norm(normed, axis=1).max() <= 1.0 + 1e-6
    restored = normed * scale + centroid
    np.testing.assert_allclose(restored, pts, atol=1e-4)


# ── noise ───────────────────────────────────────────────────────────────────

def test_gaussian_jitter_statistics():
    rng = np.random.default_rng(0)
    pts = np.zeros((20000, 3), dtype=np.float32)
    out = gaussian_jitter(pts, sigma=0.05, rng=rng)
    assert abs(out.std() - 0.05) < 0.005


def test_random_outliers_inside_bbox(plane_cloud):
    rng = np.random.default_rng(0)
    out = random_outliers(plane_cloud, ratio=0.1, rng=rng)
    assert len(out) == int(len(plane_cloud) * 0.1)
    assert (out >= plane_cloud.min(axis=0) - 1e-6).all()
    assert (out <= plane_cloud.max(axis=0) + 1e-6).all()


def test_fog_backscatter_range_bounds(plane_cloud):
    rng = np.random.default_rng(0)
    out = fog_backscatter(plane_cloud, ratio=0.1, mean_range=8.0, max_range=25.0, rng=rng)
    r = np.linalg.norm(out, axis=1)
    assert r.max() <= 25.0 + 1e-6
    assert r.min() >= 0.5 - 1e-6


def test_inject_noise_labels_and_counts(cfg, plane_cloud):
    noisy = inject_noise(plane_cloud, cfg, seed=0)
    b = noisy.breakdown
    assert len(noisy.points) == b["total_noisy"]
    assert b["total_noisy"] == b["original"] + b["outliers_added"] + b["fog_points_added"]
    assert noisy.labels.sum() == b["outliers_added"] + b["fog_points_added"]
    # determinism
    again = inject_noise(plane_cloud, cfg, seed=0)
    np.testing.assert_array_equal(noisy.points, again.points)


# ── filters ─────────────────────────────────────────────────────────────────

def test_sor_removes_isolated_outlier(plane_cloud):
    cloud = np.vstack([plane_cloud, [[0, 0, 50]]]).astype(np.float32)
    kept, idx, _ = sor_filter(cloud, nb_neighbors=10, std_ratio=2.0)
    assert len(cloud) - 1 not in idx           # the far point was removed
    assert len(kept) >= len(plane_cloud) * 0.95


def test_ror_removes_isolated_outlier(plane_cloud):
    cloud = np.vstack([plane_cloud, [[0, 0, 50]]]).astype(np.float32)
    kept, idx, _ = ror_filter(cloud, nb_points=4, radius=1.0)
    assert len(cloud) - 1 not in idx


def test_removal_precision_recall():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    kept = np.array([0, 1])                    # removed exactly the two noise points
    pr = removal_precision_recall(kept, labels)
    assert pr["precision"] == 1.0 and pr["recall"] == 1.0


# ── metrics ─────────────────────────────────────────────────────────────────

def test_chamfer_zero_for_identical():
    pts = np.random.default_rng(0).normal(size=(500, 3))
    assert chamfer_distance(pts, pts) == 0.0
    assert hausdorff_distance(pts, pts) == 0.0


def test_chamfer_known_offset():
    a = np.zeros((10, 3))
    b = np.zeros((10, 3)); b[:, 0] = 2.0
    assert chamfer_distance(a, b) == pytest.approx(2.0)
    assert hausdorff_distance(a, b) == pytest.approx(2.0)


def test_hausdorff_dominated_by_worst_point():
    a = np.zeros((100, 3))
    b = np.zeros((100, 3)); b[0, 0] = 10.0
    assert hausdorff_distance(a, b) == pytest.approx(10.0)
    assert hausdorff95(a, b) < 10.0            # robust variant ignores the tail
    assert chamfer_distance(a, b) < 1.0
