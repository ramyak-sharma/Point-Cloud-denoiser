"""Reading and writing KITTI raw Velodyne scans (*.bin, float32 x,y,z,intensity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_bin(path: str | Path) -> np.ndarray:
    """Load a KITTI Velodyne scan, returning an (N, 3) float32 XYZ array.

    The intensity channel is dropped: the denoising pipeline and both
    evaluation metrics operate purely on geometry.
    """
    pts = np.fromfile(str(path), dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"{path}: size {pts.size} is not a multiple of 4 floats")
    return pts.reshape(-1, 4)[:, :3]


def save_bin(path: str | Path, points: np.ndarray, intensity: np.ndarray | None = None) -> None:
    """Save an (N, 3) XYZ array as a KITTI-format .bin (intensity zero-filled).

    Keeping the KITTI layout lets denoised clouds feed directly into
    KITTI-trained detectors (mmdetection3d expects 4-channel scans).
    """
    points = np.asarray(points, dtype=np.float32)
    if intensity is None:
        intensity = np.zeros(len(points), dtype=np.float32)
    out = np.hstack([points, intensity.reshape(-1, 1).astype(np.float32)])
    out.tofile(str(path))


def list_scans(directories: list[str | Path]) -> list[Path]:
    """Recursively collect every .bin scan under the given directories, sorted."""
    scans: list[Path] = []
    for d in directories:
        d = Path(d)
        if d.is_dir():
            scans.extend(d.rglob("*.bin"))
    # de-duplicate (data/ may be nested inside a raw_dir) and sort for determinism
    return sorted(set(scans))


def split_scans(
    scans: list[Path], fractions: tuple[float, float, float], seed: int
) -> dict[str, list[Path]]:
    """Deterministic train/val/test split over scan files."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scans))
    n_train = int(len(scans) * fractions[0])
    n_val = int(len(scans) * fractions[1])
    idx = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    return {k: [scans[i] for i in v] for k, v in idx.items()}
