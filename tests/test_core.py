"""Unit tests for the core preprocessing, noise, filter, and metric functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidar_denoise.config import load_config
from lidar_denoise import range_image
from lidar_denoise.filters import (
    estimate_ego_motion,
    idsor_filter,
    removal_precision_recall,
    ror_filter,
    sor_filter,
    temporal_filter,
)
from lidar_denoise.metrics import (
    chamfer_distance,
    hausdorff95,
    hausdorff_distance,
    oracle_floor,
    removal_scores,
)
from lidar_denoise.noise import (
    apply_severity,
    fog_backscatter,
    gaussian_jitter,
    inject_noise,
    lisa_weather,
    random_outliers,
)
from lidar_denoise.pseudo_labels import pseudo_noise_labels, score_pseudo_labels
from lidar_denoise.preprocessing import (
    normalize_unit_sphere,
    preprocess,
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


@pytest.fixture
def scene_cloud():
    """Three orthogonal walls — a scene that constrains rigid motion in 3-D.

    A single plane is useless for registration tests: point-to-point ICP
    cannot observe translation *within* the plane (the cloud slides onto
    itself), and a dense planar grid also stays self-similar under a shift,
    so a temporal filter sees no change. Corner geometry removes both
    degeneracies.
    """
    g = np.linspace(0, 6, 26)
    a, b = np.meshgrid(g, g)
    floor = np.stack([a.ravel(), b.ravel(), np.zeros(a.size)], axis=1)
    wall_x = np.stack([np.zeros(a.size), a.ravel(), b.ravel()], axis=1)
    wall_y = np.stack([a.ravel(), np.zeros(a.size), b.ravel()], axis=1)
    return np.vstack([floor, wall_x, wall_y]).astype(np.float32)


@pytest.fixture
def sparse_cloud():
    """Sparse 3-D corner, ~3 m spacing, jittered.

    Two properties are needed at once. Sparsity: at 3 m spacing a 0.8 m shift
    leaves every point 0.8 m from its own source and 2.2 m from the next, so
    nothing falls inside a 0.5 m radius and the uncompensated filter really
    does delete the cloud (a 1.2 m grid would alias — 1.2 - 0.8 = 0.4 m, and
    points would be spuriously "matched" to their neighbours). Corner
    geometry: so ICP can observe the in-plane translation it must undo.
    """
    # >=100 points in total, or estimate_ego_motion's stability guard trips
    # and returns identity, silently disabling compensation.
    g = np.arange(0, 20, 2.5)
    a, b = np.meshgrid(g, g)
    floor = np.stack([a.ravel(), b.ravel(), np.zeros(a.size)], axis=1)
    wall_x = np.stack([np.zeros(a.size), a.ravel(), b.ravel()], axis=1)
    wall_y = np.stack([a.ravel(), np.zeros(a.size), b.ravel()], axis=1)
    pts = np.vstack([floor, wall_x, wall_y])
    pts += np.random.default_rng(0).normal(0, 0.05, pts.shape)   # break symmetry
    return np.unique(pts.round(4), axis=0).astype(np.float32)


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
    cfg = {**cfg, "noise": {**cfg["noise"], "mode": "classic"}}
    noisy = inject_noise(plane_cloud, cfg, seed=0)
    b = noisy.breakdown
    assert len(noisy.points) == b["total_noisy"]
    assert b["total_noisy"] == b["original"] + b["outliers_added"] + b["fog_points_added"]
    assert noisy.labels.sum() == b["outliers_added"] + b["fog_points_added"]
    # determinism
    again = inject_noise(plane_cloud, cfg, seed=0)
    np.testing.assert_array_equal(noisy.points, again.points)


def test_lisa_weather_physics(plane_cloud):
    """Heavier rain must lose more returns and relocate more of them."""
    shifted = plane_cloud + np.array([20, 0, 0], dtype=np.float32)  # 15-25 m range
    rng = np.random.default_rng(0)
    out_l, lab_l, lost_l, s_l, src_l = lisa_weather(shifted, rain_rate=5.0, rng=rng)
    rng = np.random.default_rng(0)
    out_h, lab_h, lost_h, s_h, src_h = lisa_weather(shifted, rain_rate=50.0, rng=rng)
    # surviving surface returns keep a link to their clean source; relocated
    # droplet returns do not (they no longer correspond to that surface)
    assert (src_l[lab_l == 0] >= 0).all()
    assert (src_l[lab_l == 1] == -1).all()
    assert s_h["lost"] > s_l["lost"]
    assert s_h["scattered"] > s_l["scattered"]
    assert len(out_l) == len(shifted) - s_l["lost"]
    # A droplet return can never lie beyond the target it occluded, so every
    # relocated point must fall inside the cloud's own maximum range (~25.5 m
    # for this corner-of-plane fixture, not a flat 25 m).
    assert (
        np.linalg.norm(out_h[lab_h == 1], axis=1).max()
        <= np.linalg.norm(shifted, axis=1).max() + 1e-4
    )


def test_inject_noise_lisa_mode_counts(cfg, plane_cloud):
    shifted = plane_cloud + np.array([20, 0, 0], dtype=np.float32)
    noisy = inject_noise(shifted, cfg, seed=0)          # default mode is lisa
    b = noisy.breakdown
    assert b["total_noisy"] == (
        b["original"] - b["lost_returns"] + b["outliers_added"]
    )
    assert noisy.labels.sum() == b["backscatter_relocated"] + b["outliers_added"]


def test_apply_severity_overrides(cfg):
    heavy = apply_severity(cfg, "heavy")
    assert heavy["noise"]["rain_rate"] == 50.0
    assert cfg["noise"]["rain_rate"] == 25.0            # original untouched


# ── filters ─────────────────────────────────────────────────────────────────


def test_temporal_filter_drops_transient_points(plane_cloud):
    """Points absent from the previous sweep are removed; persistent kept."""
    transient = np.array([[0, 0, 30]], dtype=np.float32)
    current = np.vstack([plane_cloud, transient])
    kept, idx, _ = temporal_filter(current, prev_points=plane_cloud, radius=0.5)
    assert len(current) - 1 not in idx
    assert len(kept) == len(plane_cloud)

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


def test_ego_motion_compensation_recovers_translation(scene_cloud):
    """ICP must recover a known inter-sweep shift."""
    shift = np.array([0.8, 0.3, 0.0], dtype=np.float32)   # ~8.5 m/s at 10 Hz
    transform, fitness = estimate_ego_motion(scene_cloud, scene_cloud + shift)
    np.testing.assert_allclose(transform[:3, 3], shift, atol=0.05)
    assert fitness > 0.9


def test_temporal_filter_survives_ego_motion(sparse_cloud):
    """Ego-motion beyond `radius` must not wipe out static structure.

    Measured inter-sweep translation on these KITTI drives averages 0.56 m
    and reaches 0.94 m, i.e. above the 0.5 m consistency radius on half the
    frames — uncompensated, the filter deletes the world.
    """
    shift = np.array([0.8, 0.0, 0.0], dtype=np.float32)
    current = sparse_cloud + shift
    kept_raw, _, _ = temporal_filter(
        current, sparse_cloud, radius=0.5, compensate_ego_motion=False
    )
    kept_icp, _, _ = temporal_filter(
        current, sparse_cloud, radius=0.5, compensate_ego_motion=True
    )
    assert len(kept_raw) < 0.5 * len(current)      # uncompensated: mass deletion
    assert len(kept_icp) > 0.95 * len(current)     # compensated: structure survives


def test_removal_scores_f1():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    perfect = removal_scores(np.array([0, 1]), labels)
    assert perfect["removal_f1"] == 1.0
    keep_all = removal_scores(np.arange(4), labels)
    assert keep_all["removal_recall"] == 0.0 and keep_all["removal_f1"] == 0.0


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


def test_metrics_nan_on_empty_cloud():
    """An emptied cloud must yield NaN, not a silent inf that skews means."""
    a = np.zeros((10, 3))
    empty = np.zeros((0, 3))
    assert np.isnan(chamfer_distance(a, empty))
    assert np.isnan(hausdorff_distance(a, empty))
    assert np.isnan(hausdorff95(empty, a))


# ── intensity, IDSOR, range image ───────────────────────────────────────────

def test_intensity_survives_preprocessing(cfg, plane_cloud):
    """Extra channels must ride through the whole preprocessing chain."""
    xyzi = np.hstack([plane_cloud, np.full((len(plane_cloud), 1), 0.4, np.float32)])
    out = preprocess(xyzi, cfg)
    assert out.shape[1] == 4
    assert np.allclose(out[:, 3], 0.4, atol=1e-3)      # voxel averaging preserves it
    assert preprocess(plane_cloud, cfg).shape[1] == 3  # geometry-only still works


def test_lisa_attenuates_intensity(cfg, plane_cloud):
    """Weather noise must return less energy than real surface returns."""
    shifted = plane_cloud + np.array([20, 0, 0], dtype=np.float32)
    xyzi = np.hstack([shifted, np.full((len(shifted), 1), 0.5, np.float32)])
    noisy = inject_noise(xyzi, cfg, seed=0)
    i, lab = noisy.points[:, 3], noisy.labels
    assert i[lab == 1].mean() < i[lab == 0].mean()
    assert (i >= 0).all() and (i <= 1).all()


def test_idsor_needs_intensity(plane_cloud):
    with pytest.raises(ValueError):
        idsor_filter(plane_cloud)                      # (N,3) input must be rejected


def test_idsor_removes_the_dimmer_points(plane_cloud):
    """Among points removed, mean intensity must be below that of survivors.

    Intensity is used RELATIVELY — normalised per cloud against its own 95th
    percentile — so scaling every return by the same gain changes nothing,
    and a uniformly dim cloud is not treated as uniformly noise. The
    mechanism is therefore tested as a within-cloud contrast rather than by
    comparing two differently-lit clouds.
    """
    rng = np.random.default_rng(0)
    geom = np.vstack([plane_cloud, rng.uniform(-5, 5, (300, 3))]).astype(np.float32)
    intensity = rng.uniform(0.0, 1.0, (len(geom), 1)).astype(np.float32)
    cloud = np.hstack([geom, intensity])
    _, kept, _ = idsor_filter(cloud, intensity_gamma=2.0, intensity_floor=0.1)
    removed_mask = np.ones(len(cloud), bool)
    removed_mask[kept] = False
    assert removed_mask.any(), "filter removed nothing; test is vacuous"
    assert intensity[removed_mask].mean() < intensity[~removed_mask].mean()


def test_idsor_range_normalisation_protects_far_field(plane_cloud):
    """Range normalisation must keep distant sparse structure alive.

    Without it the threshold is uniform and naturally-sparse far-field
    returns get eroded — the failure mode the adverse-weather literature
    repeatedly reports for plain SOR.
    """
    near = np.hstack([plane_cloud, np.full((len(plane_cloud), 1), 0.5, np.float32)])
    far = near.copy()
    far[:, 0] += 45.0                                  # same structure, pushed out
    far[:, :3] *= 1.0
    cloud = np.vstack([near, far]).astype(np.float32)
    _, with_norm, _ = idsor_filter(cloud, range_beta=3.0)
    _, without, _ = idsor_filter(cloud, range_beta=0.0)
    far_ids = set(range(len(near), len(cloud)))
    kept_far_with = len(far_ids & set(with_norm.tolist()))
    kept_far_without = len(far_ids & set(without.tolist()))
    assert kept_far_with >= kept_far_without


def test_range_image_roundtrip_preserves_indices(plane_cloud):
    """Every projected point must map back to itself."""
    cloud = plane_cloud + np.array([10, 0, 0], dtype=np.float32)
    ri = range_image.project(cloud, height=64, width=1024)
    assert ri.image.shape == (5, 64, 1024)
    assert ri.represented.sum() > 0
    # a pixel's owner index must point at a point that maps to that pixel
    owners = ri.point_idx.ravel()
    occupied = owners >= 0
    assert (ri.pixel_of[owners[occupied]] == np.flatnonzero(occupied)).all()


def test_range_image_reports_projection_loss(plane_cloud):
    stats = range_image.projection_loss(range_image.project(plane_cloud, 64, 256))
    assert 0 <= stats["represented_pct"] <= 100
    assert stats["represented"] <= stats["n_points"]


def test_recover_rings_finds_64_lasers():
    """Ring recovery must count one wrap per laser revolution.

    Yaw ASCENDS within a ring and drops by 2pi at each boundary — verified
    against these KITTI drives, which show exactly 64 such drops per scan.
    """
    yaw = np.concatenate([np.linspace(-np.pi, np.pi, 100, endpoint=False)] * 64)
    r = np.full(len(yaw), 10.0)
    pts = np.stack([r * np.cos(yaw), r * np.sin(yaw), np.zeros_like(yaw)], axis=1)
    rings = range_image.recover_rings(pts, n_rings=64)
    assert rings.max() == 63


# ── self-supervised pseudo-labels ───────────────────────────────────────────

def test_pseudo_labels_need_intensity(plane_cloud):
    with pytest.raises(ValueError):
        pseudo_noise_labels(plane_cloud)


def test_pseudo_labels_never_read_ground_truth(cfg, plane_cloud, monkeypatch):
    """The pseudo-labeller must derive labels from the scan, not the generator.

    Guarded by a test because the whole self-supervision claim collapses if
    a ground-truth array leaks into the labelling path.
    """
    shifted = plane_cloud + np.array([15, 0, 0], dtype=np.float32)
    xyzi = np.hstack([shifted, np.full((len(shifted), 1), 0.5, np.float32)])
    noisy = inject_noise(xyzi, cfg, seed=0)
    # corrupt the true labels; pseudo-labels must be unchanged
    a, _ = pseudo_noise_labels(noisy.points)
    noisy.labels[:] = 1
    b, _ = pseudo_noise_labels(noisy.points)
    np.testing.assert_array_equal(a, b)


def test_pseudo_labels_flag_isolated_weak_returns(cfg, plane_cloud):
    """Isolated, weak returns must be flagged more often than surface points."""
    surface = plane_cloud + np.array([15, 0, 0], dtype=np.float32)
    surface = np.hstack([surface, np.full((len(surface), 1), 0.6, np.float32)])
    rng = np.random.default_rng(0)
    floaters = np.hstack([
        rng.uniform([5, -6, -3], [25, 6, 4], (400, 3)),
        rng.uniform(0.0, 0.05, (400, 1)),
    ]).astype(np.float32)
    cloud = np.vstack([surface, floaters]).astype(np.float32)
    labels, stats = pseudo_noise_labels(cloud)
    flagged_floaters = labels[len(surface):].mean()
    flagged_surface = labels[: len(surface)].mean()
    assert flagged_floaters > flagged_surface
    assert 0 <= stats["flagged_pct"] <= 100


def test_score_pseudo_labels_is_diagnostic_only():
    truth = np.array([0, 0, 1, 1], np.uint8)
    assert score_pseudo_labels(np.array([0, 0, 1, 1], np.uint8), truth)["f1"] == 1.0
    assert score_pseudo_labels(np.array([0, 0, 0, 0], np.uint8), truth)["recall"] == 0.0


def test_oracle_floor_is_not_zero(plane_cloud, cfg):
    """The floor must be strictly positive: LISA deletes unrecoverable returns."""
    shifted = plane_cloud + np.array([20, 0, 0], dtype=np.float32)
    noisy = inject_noise(shifted, cfg, seed=0)
    floor = oracle_floor(shifted, noisy.points, noisy.labels)
    assert floor["points_lost_to_weather"] > 0
    assert floor["cd"] > 0.0
    # and it must beat every real method's achievable error
    assert floor["cd"] < chamfer_distance(shifted, noisy.points)
