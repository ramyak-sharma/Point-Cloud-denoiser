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

from run_pipeline import (  # noqa: E402
    _denoise_all_methods,
    _load_frame,
    _prev_noisy_cloud,
    get_splits,
)
from lidar_denoise.denoise import load_range_checkpoint  # noqa: E402
from lidar_denoise.train import select_device  # noqa: E402


def _paired_recall_tests(rows: dict[str, list[dict]], reference: str) -> dict:
    """Wilcoxon signed-rank on per-frame recall, every condition vs `reference`.

    Detection recall is far noisier than the geometric metrics — per-frame
    values span 0.0 to 0.86 — so a difference of means over a handful of
    frames says very little on its own. An earlier revision of this project
    reported that two methods "beat the perfect-removal oracle" on recall
    from 10 frames; at that sample size neither difference was significant
    (p = 0.25 and p = 0.20). The test is run here so no such claim can be
    made again without support.
    """
    from scipy.stats import wilcoxon

    ref = np.array([r["recall"] for r in rows[reference]])
    out = {}
    for cond, rs in rows.items():
        if cond == reference:
            continue
        x = np.array([r["recall"] for r in rs])
        if len(x) != len(ref) or np.allclose(x, ref):
            continue
        try:
            stat, p = wilcoxon(x, ref)
        except ValueError:
            continue
        out[f"{cond} vs {reference}"] = {
            "n_pairs": int(len(x)),
            "mean_diff": float((x - ref).mean()),
            "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05),
        }
    return out


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

    range_ckpt = ckpt_dir / "range_unet.pt"
    range_model = load_range_checkpoint(range_ckpt, device) if range_ckpt.exists() else None

    splits = get_splits(cfg)
    frames = splits["test"][: dcfg["num_frames"]]
    n_seeds = dcfg.get("num_seeds", 1)
    rows: dict[str, list[dict]] = {}

    for s in range(n_seeds):
        for i, path in enumerate(frames):
            clean = _load_frame(path, cfg)
            noisy = inject_noise(clean, cfg, seed=5000 + 100 * s + i)
            reference = detect(detector, clean[:, :3], dcfg["score_threshold"])
            prev_noisy = _prev_noisy_cloud(path, cfg, seed=9700 + 100 * s + i)

            conditions = {"noisy": noisy.points[:, :3]}
            # Upper bound for a REMOVAL-based method: every injected point
            # deleted, nothing else touched. Weather-deleted returns are
            # still gone, so this is what a perfect noise classifier would
            # hand the detector, not a perfect denoiser.
            conditions["oracle (perfect removal)"] = noisy.points[noisy.labels == 0][:, :3]
            for name, (pts, _, _) in _denoise_all_methods(
                noisy, cfg, denoisers, device, prev_noisy, range_model
            ).items():
                conditions[name] = pts[:, :3]

            for cond, pts in conditions.items():
                pred = detect(detector, pts, dcfg["score_threshold"])
                m = consistency_metrics(reference, pred)
                m["frame"] = path.name
                m["seed"] = s
                rows.setdefault(cond, []).append(m)
            print(f"[detect] seed {s + 1}/{n_seeds} frame {i + 1}/{len(frames)}: "
                  f"{len(reference['boxes'])} reference boxes")

    summary = {
        cond: {
            "recall": float(np.mean([r["recall"] for r in rs])),
            "recall_std": float(np.std([r["recall"] for r in rs])),
            "precision": float(np.mean([r["precision"] for r in rs])),
            "precision_std": float(np.std([r["precision"] for r in rs])),
            "mean_reference_boxes": float(np.mean([r["n_reference"] for r in rs])),
            "n_observations": len(rs),
        }
        for cond, rs in rows.items()
    }
    significance = _paired_recall_tests(rows, "oracle (perfect removal)")
    vs_noisy = _paired_recall_tests(rows, "noisy")

    out = resolve(cfg, cfg["evaluation"]["results_dir"]) / "detection_consistency.json"
    with open(out, "w") as f:
        json.dump({"per_frame": rows, "summary": summary,
                   "significance_vs_oracle": significance,
                   "significance_vs_noisy": vs_noisy,
                   "protocol": {"n_frames": len(frames), "n_seeds": n_seeds,
                                "paired": True, "test": "Wilcoxon signed-rank"}},
                  f, indent=2)
    print(f"[detect] wrote {out}")
    for cond, s_ in summary.items():
        print(f"  {cond:26s} recall {s_['recall']:.3f} ± {s_['recall_std']:.3f}  "
              f"precision {s_['precision']:.3f}")
    print("\n  vs oracle (perfect removal):")
    for k, v in significance.items():
        print(f"    {k:48s} diff {v['mean_diff']:+.3f}  p={v['p_value']:.4f}  "
              f"{'SIG' if v['significant_at_0.05'] else 'ns'}")


if __name__ == "__main__":
    main()
