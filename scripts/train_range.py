#!/usr/bin/env python
"""Train the dual-head range-image denoiser (resume-aware)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lidar_denoise.config import load_config
from lidar_denoise.kitti_io import list_scans, split_scans
from lidar_denoise.train_range import train_range_model
from lidar_denoise.config import resolve

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--self-supervised", action="store_true",
                help="train the removal head on physics-derived pseudo-labels only")
a = ap.parse_args()
cfg = load_config(a.config)
if a.epochs:
    cfg["model"]["range_unet"]["epochs"] = a.epochs
if a.self_supervised:
    cfg["model"]["range_unet"]["self_supervised"] = True
scans = list_scans([resolve(cfg, d) for d in cfg["data"]["raw_dirs"]])
splits = split_scans(scans, tuple(cfg["data"]["split"]), cfg["data"]["seed"])
train_range_model(cfg, splits["train"], splits["val"], resume=True)
