#!/usr/bin/env python
"""Downstream detection-consistency evaluation (proposal Step 4, second half).

Runs a KITTI-pretrained PointPillars detector (mmdetection3d) on the clean,
noisy, and denoised versions of held-out test frames and reports how many of
the clean-cloud detections survive each condition.

Setup (once):
  .venv/bin/python -m pip install mmengine mmcv mmdet mmdet3d
  .venv/bin/mim download mmdet3d \
      --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class --dest models/

Usage:
  python scripts/detection_eval.py [--device cpu] [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lidar_denoise.config import load_config, resolve
from lidar_denoise.denoise import denoise_cloud, load_checkpoint
from lidar_denoise.detection import consistency_metrics, detect, load_detector
from lidar_denoise.filters import sor_ror_pipeline
from lidar_denoise.kitti_io import load_bin
from lidar_denoise.noise import inject_noise
from lidar_denoise.preprocessing import preprocess

from run_pipeline import get_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu", help="mmdet3d device (cpu recommended on mac)")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["detection"]

    models_dir = resolve(cfg, args.models_dir)
    det_ckpts = sorted(glob.glob(str(models_dir / "*.pth")))
    # the config ships inside the mmdet3d package (it needs its _base_ tree,
    # so it cannot be copied out standalone)
    import mmdet3d

    pkg_cfg = (Path(mmdet3d.__file__).parent / ".mim" / "configs" / "pointpillars"
               / f"{dcfg['config']}.py")
    det_cfgs = sorted(glob.glob(str(models_dir / f"{dcfg['config']}*.py"))) or (
        [str(pkg_cfg)] if pkg_cfg.exists() else []
    )
    if not det_cfgs or not det_ckpts:
        sys.exit(
            f"Detector config/checkpoint not found in {models_dir}.\n"
            "See README 'Detection evaluation' for the download command."
        )
    detector = load_detector(det_cfgs[0], det_ckpts[0], device=args.device)

    import torch

    from lidar_denoise.train import select_device

    device = select_device(cfg["training"]["device"])
    ckpt_dir = resolve(cfg, cfg["training"]["checkpoint_dir"])
    denoisers = {
        name: load_checkpoint(name, ckpt_dir / f"{name}.pt", device)
        for name in ("pointnet_ae", "dgcnn")
        if (ckpt_dir / f"{name}.pt").exists()
    }

    splits = get_splits(cfg)
    frames = splits["test"][: dcfg["num_frames"]]
    rows: dict[str, list[dict]] = {}
    for i, path in enumerate(frames):
        clean = preprocess(load_bin(path), cfg)
        noisy = inject_noise(clean, cfg, seed=5000 + i)
        reference = detect(detector, clean, dcfg["score_threshold"])

        conditions = {"noisy": noisy.points}
        conditions["SOR+ROR"] = sor_ror_pipeline(noisy.points, cfg)["points"]
        for name, (model, _) in denoisers.items():
            conditions[name], _ = denoise_cloud(model, name, noisy.points, cfg, device)

        for cond, pts in conditions.items():
            pred = detect(detector, pts, dcfg["score_threshold"])
            m = consistency_metrics(reference, pred)
            rows.setdefault(cond, []).append(m)
        print(f"[detect] frame {i + 1}/{len(frames)}: "
              f"{len(reference['boxes'])} reference boxes")

    summary = {
        cond: {
            "recall": float(np.mean([r["recall"] for r in rs])),
            "precision": float(np.mean([r["precision"] for r in rs])),
            "mean_reference_boxes": float(np.mean([r["n_reference"] for r in rs])),
        }
        for cond, rs in rows.items()
    }
    out = resolve(cfg, cfg["evaluation"]["results_dir"]) / "detection_consistency.json"
    with open(out, "w") as f:
        json.dump({"per_frame": rows, "summary": summary}, f, indent=2)
    print(f"[detect] wrote {out}")
    for cond, s in summary.items():
        print(f"  {cond:14s} recall {s['recall']:.3f}  precision {s['precision']:.3f}")


if __name__ == "__main__":
    main()
