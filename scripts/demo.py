#!/usr/bin/env python
"""Interactive before/after demo — the "show, don't tell" script.

Loads one KITTI scan, injects synthetic noise, runs BOTH denoising
approaches, prints their metrics, and opens an interactive side-by-side
Open3D window (clean | noisy | SOR+ROR | DGCNN) with the clouds laid out
next to each other. Use --save-only on a headless machine to write PNGs
instead of opening a window.

Usage:
  python scripts/demo.py                     # random test frame
  python scripts/demo.py --frame data/0000000042.bin
  python scripts/demo.py --save-only        # writes results/plots/demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lidar_denoise.config import load_config, resolve
from lidar_denoise.denoise import denoise_cloud, load_checkpoint
from lidar_denoise.filters import sor_ror_pipeline
from lidar_denoise.kitti_io import load_bin
from lidar_denoise.metrics import chamfer_distance, hausdorff_distance
from lidar_denoise.noise import inject_noise
from lidar_denoise.preprocessing import preprocess
from lidar_denoise.train import select_device

PALETTE = {
    "clean": (0.27, 0.87, 0.47),
    "noisy": (1.00, 0.27, 0.27),
    "SOR+ROR": (0.27, 0.60, 1.00),
    "pointnet_ae": (1.00, 0.75, 0.25),
    "dgcnn": (0.80, 0.45, 1.00),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=None, help="path to a KITTI .bin scan")
    ap.add_argument("--config", default=None)
    ap.add_argument("--save-only", action="store_true", help="write PNG, no window")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.frame:
        frame = Path(args.frame)
    else:
        from run_pipeline import get_splits

        frame = get_splits(cfg)["test"][0]
    print(f"[demo] frame: {frame}")

    clean = preprocess(load_bin(frame), cfg)
    noisy = inject_noise(clean, cfg, seed=7)
    clouds: dict[str, np.ndarray] = {"clean": clean, "noisy": noisy.points}

    clouds["SOR+ROR"] = sor_ror_pipeline(noisy.points, cfg)["points"]
    device = select_device(cfg["training"]["device"])
    ckpt_dir = resolve(cfg, cfg["training"]["checkpoint_dir"])
    for name in ("pointnet_ae", "dgcnn"):
        ckpt = ckpt_dir / f"{name}.pt"
        if ckpt.exists():
            model, _ = load_checkpoint(name, ckpt, device)
            clouds[name], _ = denoise_cloud(model, name, noisy.points, cfg, device)

    print(f"\n{'method':14s} {'points':>9s} {'CD (m)':>9s} {'HD (m)':>9s}")
    for name, pts in clouds.items():
        if name == "clean":
            print(f"{name:14s} {len(pts):9,d} {'—':>9s} {'—':>9s}")
        else:
            cd = chamfer_distance(clean, pts)
            hd = hausdorff_distance(clean, pts)
            print(f"{name:14s} {len(pts):9,d} {cd:9.4f} {hd:9.4f}")

    if args.save_only:
        from lidar_denoise.visualize import before_after_figure

        out = resolve(cfg, "results/plots") / "demo.png"
        before_after_figure(clouds, out, f"Demo — {frame.name}")
        print(f"\n[demo] figure saved to {out}")
        return

    # side-by-side interactive layout: shift each cloud along +y
    import open3d as o3d

    geoms = []
    spacing = 90.0
    for i, (name, pts) in enumerate(clouds.items()):
        pcd = o3d.geometry.PointCloud()
        shifted = pts.copy()
        shifted[:, 1] += i * spacing
        pcd.points = o3d.utility.Vector3dVector(shifted.astype(np.float64))
        pcd.paint_uniform_color(PALETTE.get(name, (0.7, 0.7, 0.7)))
        geoms.append(pcd)
        print(f"[demo] panel {i}: {name}")
    o3d.visualization.draw_geometries(
        geoms, window_name="LiDAR denoising demo — clean | noisy | denoised (see terminal)",
    )


if __name__ == "__main__":
    main()
