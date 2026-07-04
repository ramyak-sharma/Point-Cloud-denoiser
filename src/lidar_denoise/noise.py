"""Step 2 of the proposal methodology: synthetic noise modelling.

Injects the three noise families identified in the proposal's Need Analysis:

* Gaussian jitter        — sensor vibration / ranging error (every point)
* uniform random outliers — multi-echo interference, spurious returns
* fog/rain backscatter    — laser pulses reflecting off suspended particles,
                            producing phantom points at short range

`inject_noise` returns the noisy cloud together with a per-point label array
(0 = perturbed real point, 1 = injected noise point), which supplies the
supervision signal for the learning-based models and the ground truth for
precision/recall analysis of the filters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoisyCloud:
    points: np.ndarray        # (M, 3) noisy cloud
    labels: np.ndarray        # (M,) uint8: 1 where the point was injected noise
    clean: np.ndarray         # (N, 3) the clean source cloud
    breakdown: dict           # counts per noise type


def gaussian_jitter(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb every point with isotropic Gaussian noise of std sigma (m)."""
    return points + rng.normal(0.0, sigma, points.shape).astype(points.dtype)


def random_outliers(points: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform outliers inside the clean cloud's bounding box."""
    n = int(len(points) * ratio)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return rng.uniform(lo, hi, (n, 3)).astype(points.dtype)


def fog_backscatter(
    points: np.ndarray,
    ratio: float,
    mean_range: float,
    max_range: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Phantom near-range returns from fog/rain.

    Backscatter intensity decays with distance, so ranges are drawn from an
    exponential distribution (truncated at max_range) and directions follow
    the sensor geometry: uniform azimuth, elevation confined to the KITTI
    HDL-64E fan (-24.8° .. +2°).
    """
    n = int(len(points) * ratio)
    r = rng.exponential(mean_range, n)
    r = np.clip(r, 0.5, max_range)
    az = rng.uniform(-np.pi, np.pi, n)
    el = rng.uniform(np.deg2rad(-24.8), np.deg2rad(2.0), n)
    pts = np.stack(
        [r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], axis=1
    )
    return pts.astype(points.dtype)


def inject_noise(clean: np.ndarray, cfg: dict, seed: int | None = None) -> NoisyCloud:
    """Apply all three noise types from the config to a clean cloud."""
    ncfg = cfg["noise"]
    rng = np.random.default_rng(seed)

    jittered = gaussian_jitter(clean, ncfg["gaussian_sigma"], rng)
    outliers = random_outliers(clean, ncfg["outlier_ratio"], rng)
    fog = fog_backscatter(
        clean, ncfg["fog_ratio"], ncfg["fog_mean_range"], ncfg["fog_max_range"], rng
    )

    points = np.vstack([jittered, outliers, fog])
    labels = np.concatenate(
        [
            np.zeros(len(jittered), dtype=np.uint8),
            np.ones(len(outliers) + len(fog), dtype=np.uint8),
        ]
    )
    # Shuffle so injected points are not trivially identifiable by index.
    order = rng.permutation(len(points))
    breakdown = {
        "original": len(clean),
        "outliers_added": len(outliers),
        "fog_points_added": len(fog),
        "total_noisy": len(points),
    }
    return NoisyCloud(points[order], labels[order], clean, breakdown)
