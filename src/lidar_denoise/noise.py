"""Step 2 of the proposal methodology: synthetic noise modelling.

Two weather models are provided, selected by ``noise.mode`` in the config:

* ``classic`` — the original heuristic: exponential-range fog blob around
  the sensor (kept for comparability with earlier runs).
* ``lisa``    — a simplified physics-based model following LISA
  [Kilic et al., "LiDAR Light Scattering Augmentation", arXiv:2107.07004]:
  a Mie-theory extinction coefficient derived from the rain rate attenuates
  each return with range; returns whose attenuated power falls below the
  detection threshold are LOST (deleted); for a fraction of beams a raindrop
  in the beam out-reflects the target and the point is RELOCATED to the
  droplet's range (backscatter); surviving returns get SNR-dependent extra
  range jitter. This adds the point-loss failure mode the heuristic model
  cannot produce.

Both modes also apply Gaussian jitter (sensor vibration / ranging error) and
uniform bounding-box outliers (multi-echo interference), per the proposal's
noise taxonomy. Severity presets (light/moderate/heavy) follow the
three-level convention of the Robo3D corruption benchmark
[Kong et al., ICCV 2023].

`inject_noise` returns the noisy cloud together with a per-point label array
(0 = perturbed real point, 1 = injected/relocated noise point), which
supplies the supervision signal for the learning-based models and the ground
truth for precision/recall analysis of the filters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoisyCloud:
    points: np.ndarray        # (M, 3) or (M, 4) noisy cloud
    labels: np.ndarray        # (M,) uint8: 1 where the point was injected noise
    clean: np.ndarray         # (N, 3) or (N, 4) the clean source cloud
    breakdown: dict           # counts per noise type
    # (M,) int32 index of each output point's clean source, or -1 for points
    # with no clean counterpart (injected outliers, relocated backscatter).
    # This gives an EXACT restoration target — the true position a surviving
    # return should be moved back to — rather than a nearest-neighbour guess,
    # which is what a displacement head needs to be supervised properly.
    source_idx: np.ndarray | None = None


def gaussian_jitter(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb XYZ with isotropic Gaussian noise of std sigma (m).

    Any trailing channels (intensity) pass through untouched — ranging error
    displaces a return, it does not change how much light came back.
    """
    out = points.copy()
    out[:, :3] += rng.normal(0.0, sigma, (len(points), 3)).astype(points.dtype)
    return out


def random_outliers(points: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform outliers inside the clean cloud's bounding box.

    Multi-echo and interference returns are weak, so when the source cloud
    carries intensity these are given low values drawn from the bottom of
    the range rather than the full spread.
    """
    n = int(len(points) * ratio)
    lo, hi = points[:, :3].min(axis=0), points[:, :3].max(axis=0)
    xyz = rng.uniform(lo, hi, (n, 3))
    if points.shape[1] > 3:
        xyz = np.hstack([xyz, rng.uniform(0.0, 0.2, (n, 1))])
    return xyz.astype(points.dtype)


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


def lisa_weather(
    points: np.ndarray, rain_rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Simplified LISA-style hybrid Monte-Carlo rain/fog model.

    Physics, following Kilic et al. (arXiv:2107.07004):
    * extinction coefficient alpha = 0.003 * R^0.6 [1/m], an empirical Mie
      fit for precipitation with spray/fog at 905 nm (R in mm/h); two-way
      transmittance exp(-2 alpha r).
    * BACKSCATTER (decided first — the drop intercepts the beam *before*
      the target): probability 0.35 * (1 - transmittance); the point
      relocates to the drop's range, sampled ~ Exp(1/(2 alpha)) truncated
      to (0.5 m, r) — near ranges dominate, as observed in rain.
    * LOST returns (non-scattered beams only): detection probability falls
      with transmitted power; a return survives with p = transmittance^0.5.
    * surviving returns get extra range jitter scaled by the lost power.

    Returns (kept_or_relocated points, labels 1=relocated phantom,
    lost_mask over the input, stats).
    """
    n = len(points)
    xyz = points[:, :3]
    intensity = points[:, 3] if points.shape[1] > 3 else None
    r = np.linalg.norm(xyz, axis=1)
    r = np.maximum(r, 1e-6)
    direction = xyz / r[:, None]

    alpha = 0.003 * rain_rate**0.6                   # extinction [1/m]
    transmittance = np.exp(-2.0 * alpha * r)         # two-way power ratio

    # 1) backscatter — a drop in the beam out-reflects the target
    p_scatter = 0.35 * (1.0 - transmittance)
    scattered = rng.random(n) < p_scatter
    r_drop = rng.exponential(1.0 / (2.0 * alpha), n)
    r_drop = np.clip(r_drop, 0.5, r)                 # cannot lie beyond target

    # 2) lost returns — weakest attenuated echoes drop below threshold
    survive_p = transmittance**0.5
    lost = (rng.random(n) > survive_p) & ~scattered

    # 3) SNR-dependent range jitter on true survivors
    sigma_r = 0.02 * (1.0 - transmittance)           # up to 2 cm extra
    r_noisy = r + rng.normal(0.0, 1.0, n) * sigma_r

    new_r = np.where(scattered, r_drop, r_noisy)
    out = direction * new_r[:, None]
    keep = ~lost
    labels = scattered[keep].astype(np.uint8)

    if intensity is not None:
        # Intensity follows the same radiometry that drives the geometry.
        # A surviving surface return carries the two-way transmittance it
        # was already attenuated by. A droplet return is weak for two
        # independent reasons: a raindrop's backscatter cross-section is far
        # smaller than a surface's, and the pulse only travelled to the drop
        # (r_drop < r), so the 1/r^2 falloff is milder but the reflectivity
        # penalty dominates. That contrast is exactly the signal IDSOR and
        # the range-image classifier exploit.
        i_surface = intensity * transmittance
        i_drop = (
            rng.uniform(0.0, 0.15, n)                    # weak droplet return
            * np.exp(-2.0 * alpha * r_drop)              # attenuated over its shorter path
        )
        i_out = np.where(scattered, i_drop, i_surface)
        out = np.hstack([out, np.clip(i_out, 0.0, 1.0)[:, None]])

    # Index of the clean point each surviving return came from; a relocated
    # backscatter return no longer corresponds to its original surface, so
    # it is marked -1 along with anything injected later.
    source = np.where(scattered, -1, np.arange(n))[keep].astype(np.int32)

    stats = {
        "rain_rate_mm_h": rain_rate,
        "alpha_per_m": float(alpha),
        "lost": int(lost.sum()),
        "scattered": int(scattered.sum()),
    }
    return out[keep].astype(points.dtype), labels, lost, stats, source


def apply_severity(cfg: dict, severity: str) -> dict:
    """Return a config copy with the noise block set to a severity preset."""
    import copy

    out = copy.deepcopy(cfg)
    preset = cfg["noise"]["severity_presets"][severity]
    out["noise"].update(preset)
    return out


def inject_noise(clean: np.ndarray, cfg: dict, seed: int | None = None) -> NoisyCloud:
    """Apply the configured weather model plus sensor-noise components."""
    ncfg = cfg["noise"]
    rng = np.random.default_rng(seed)
    mode = ncfg.get("mode", "classic")

    if mode == "lisa":
        weather_pts, weather_labels, _, stats, weather_src = lisa_weather(
            clean, ncfg["rain_rate"], rng
        )
        # sensor jitter applies to the (already weather-affected) real returns
        real = weather_labels == 0
        weather_pts[real] = gaussian_jitter(weather_pts[real], ncfg["gaussian_sigma"], rng)
        stats["has_intensity"] = clean.shape[1] > 3
        outliers = random_outliers(clean, ncfg["outlier_ratio"], rng)
        points = np.vstack([weather_pts, outliers])
        labels = np.concatenate(
            [weather_labels, np.ones(len(outliers), dtype=np.uint8)]
        )
        source = np.concatenate(
            [weather_src, np.full(len(outliers), -1, dtype=np.int32)]
        )
        breakdown = {
            "original": len(clean),
            "lost_returns": stats["lost"],
            "backscatter_relocated": stats["scattered"],
            "outliers_added": len(outliers),
            "total_noisy": len(points),
            **{k: v for k, v in stats.items() if k in ("rain_rate_mm_h", "alpha_per_m")},
        }
    else:
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
        source = np.concatenate(
            [
                np.arange(len(jittered), dtype=np.int32),   # jitter preserves identity
                np.full(len(outliers) + len(fog), -1, dtype=np.int32),
            ]
        )
        breakdown = {
            "original": len(clean),
            "outliers_added": len(outliers),
            "fog_points_added": len(fog),
            "total_noisy": len(points),
        }

    # Shuffle so injected points are not trivially identifiable by index.
    order = rng.permutation(len(points))
    return NoisyCloud(points[order], labels[order], clean, breakdown, source[order])
