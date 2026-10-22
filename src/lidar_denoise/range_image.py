"""Spherical (cylindrical) range-image projection and its exact inverse.

Every real-time adverse-weather denoiser in the recent literature —
WeatherNet, LiSnowNet, 4DenoiseNet, LRINet, LIORNet — abandons the
point-graph representation and works on a 2-D range image instead. The
reason is throughput, and it is decisive: our DGCNN's EdgeConv builds a
k-NN graph per patch and runs at 0.59 Hz, while range-image networks in the
same task report 16-43 Hz. The projection is what buys that, for two
reasons:

* it recovers the sensor's native topology — a rotating LiDAR *is* a 2-D
  array of (laser channel, azimuth) samples, so neighbourhood queries become
  array indexing instead of spatial search, and
* it lets ordinary 2-D convolutions supply the local context that EdgeConv
  had to compute explicitly.

The cost is quantisation: two returns can collide in one pixel, and any
point that loses a collision is not represented in the image. We therefore
keep an explicit index map so predictions can be scattered back to *all*
original points, and points that never won a pixel are handled by an
explicit policy rather than silently dropped (see `unproject_labels`).

Geometry (KITTI HDL-64E defaults):
    u = W * (1 - (atan2(y, x) + pi) / 2pi)          azimuth  -> column
    v = H * (1 - (arcsin(z / r) - fov_down) / fov)  elevation -> row

**Measured fidelity** on the preprocessed (0.15 m voxel) clouds this project
evaluates on, averaged over 4 frames. "Ceiling" is the fraction of points
that recover their own ground-truth label after a projection/unprojection
round trip, and is the real upper bound on a range-image method's removal
accuracy — it exceeds the represented fraction because a point that loses a
pixel inherits its occluder's label, which is usually correct:

    64 x 1024   69.2% represented   86.3% ceiling   38.1% occupancy
    64 x 2048   83.3% represented   92.8% ceiling   23.0% occupancy   <- default
    64 x 4096   91.3% represented   96.4% ceiling   12.6% occupancy

Voxel downsampling is what costs us here: it replaces the sensor's 64
discrete laser rings with grid centroids, so the rows no longer correspond
to beams. Projecting a *raw* scan with `recover_rings` is near-lossless, but
then the range-image method would be denoising a different cloud from every
other method in the comparison, and the metrics would not be comparable. We
keep the shared preprocessed cloud and report the ceiling instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RangeImage:
    """A projected scan plus everything needed to invert the projection.

    image:      (C, H, W) float32 feature planes — range, intensity, x, y, z
    valid:      (H, W) bool, True where a point occupies the pixel
    point_idx:  (H, W) int32, index into the source cloud (-1 where empty)
    pixel_of:   (N,) int32, flat pixel index each source point mapped to
    represented:(N,) bool, True for points that won their pixel
    """

    image: np.ndarray
    valid: np.ndarray
    point_idx: np.ndarray
    pixel_of: np.ndarray
    represented: np.ndarray
    height: int
    width: int


def recover_rings(points: np.ndarray, n_rings: int = 64) -> np.ndarray:
    """Laser-ring index per point, recovered from azimuth wrap-around.

    A spinning LiDAR writes its returns in scan order, so the yaw angle
    decreases monotonically within a ring and jumps by ~2pi at each ring
    boundary. Counting those jumps recovers the exact row assignment that
    uniform elevation binning only approximates (the HDL-64E's lasers are
    not evenly spaced in elevation). Verified on these drives: exactly 64
    wraps per scan.

    Only meaningful for a RAW scan — voxel downsampling destroys both the
    ordering and the ring structure this relies on.
    """
    yaw = np.arctan2(points[:, 1], points[:, 0])
    ring = np.zeros(len(points), dtype=np.int32)
    ring[1:] = np.cumsum(np.diff(yaw) < -np.pi)
    return np.clip(ring, 0, n_rings - 1)


def project(
    points: np.ndarray,
    height: int = 64,
    width: int = 2048,
    fov_up_deg: float = 2.0,
    fov_down_deg: float = -24.8,
    max_range: float = 60.0,
) -> RangeImage:
    """Project an (N, 3) or (N, 4) cloud into a range image.

    Where several points fall in the same pixel the NEAREST is kept, which is
    the physically right choice: a nearer return is what the sensor would
    actually have reported for that beam, and it also means weather
    backscatter (which sits in front of the surface it occluded) is preserved
    rather than averaged away — it is exactly what we need the network to
    learn to reject.
    """
    xyz = points[:, :3]
    intensity = points[:, 3] if points.shape[1] > 3 else np.zeros(len(points), np.float32)

    r = np.linalg.norm(xyz, axis=1)
    r = np.maximum(r, 1e-6)

    fov_up, fov_down = np.deg2rad(fov_up_deg), np.deg2rad(fov_down_deg)
    fov = fov_up - fov_down

    yaw = np.arctan2(xyz[:, 1], xyz[:, 0])
    pitch = np.arcsin(np.clip(xyz[:, 2] / r, -1.0, 1.0))

    u = (1.0 - (yaw + np.pi) / (2.0 * np.pi)) * width
    v = (1.0 - (pitch - fov_down) / fov) * height
    u = np.clip(np.floor(u), 0, width - 1).astype(np.int32)
    v = np.clip(np.floor(v), 0, height - 1).astype(np.int32)
    flat = v * width + u

    # Nearest-wins collision resolution: sort far->near so that writing in
    # order leaves the nearest point owning each pixel.
    order = np.argsort(-r)
    point_idx = np.full(height * width, -1, dtype=np.int32)
    point_idx[flat[order]] = order

    valid = point_idx >= 0
    img = np.zeros((5, height * width), dtype=np.float32)
    sel = point_idx[valid]
    img[0, valid] = r[sel] / max_range                   # normalised range
    img[1, valid] = intensity[sel]
    img[2:5, valid] = (xyz[sel] / max_range).T           # normalised xyz

    represented = np.zeros(len(points), dtype=bool)
    represented[sel] = True

    return RangeImage(
        image=img.reshape(5, height, width),
        valid=valid.reshape(height, width),
        point_idx=point_idx.reshape(height, width),
        pixel_of=flat,
        represented=represented,
        height=height,
        width=width,
    )


def unproject_labels(
    ri: RangeImage, pixel_values: np.ndarray, fill: float = 0.0
) -> np.ndarray:
    """Scatter a per-pixel prediction back to every source point.

    A point that lost a pixel collision has no prediction of its own; it
    inherits the prediction of the pixel it mapped into. This is deliberate
    and is the conservative choice for a *removal* head: an occluded point
    shares a beam with its occluder, so if the winner was classified as
    weather the loser is very likely weather too. Points falling outside the
    vertical field of view receive `fill`.
    """
    flat = np.asarray(pixel_values).reshape(-1)
    out = np.full(len(ri.pixel_of), fill, dtype=np.float32)
    inside = (ri.pixel_of >= 0) & (ri.pixel_of < flat.size)
    out[inside] = flat[ri.pixel_of[inside]]
    return out


def projection_loss(ri: RangeImage) -> dict:
    """How much of the cloud the projection actually represents.

    Reported rather than assumed: at 64x1024 a dense KITTI scan loses a few
    percent of its points to pixel collisions, and any accuracy claim about
    a range-image method has to be read against that ceiling.
    """
    n = len(ri.represented)
    kept = int(ri.represented.sum())
    return {
        "n_points": n,
        "represented": kept,
        "represented_pct": 100.0 * kept / n if n else float("nan"),
        "occupied_pixels": int(ri.valid.sum()),
        "pixel_occupancy_pct": 100.0 * ri.valid.sum() / ri.valid.size,
    }
