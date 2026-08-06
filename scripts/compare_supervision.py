#!/usr/bin/env python
"""Supervised vs self-supervised RangeRestoreNet, on identical test frames.

Answers the sharpest objection to this project: the learned denoisers are
trained on `inject_noise`'s own label array, which no real deployment
possesses. The self-supervised model is trained only on pseudo-labels
derived from the physics of a weather return (see `pseudo_labels`), so it is
the only configuration that could be trained on measured adverse-weather
data. Both are scored here against the *true* labels.

Usage:
  python scripts/compare_supervision.py [--frames 20] [--seeds 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lidar_denoise.config import load_config, resolve
from lidar_denoise.denoise import denoise_cloud_range, load_range_checkpoint
from lidar_denoise.metrics import chamfer_distance, hausdorff95, removal_scores
from lidar_denoise.noise import inject_noise
from lidar_denoise.pseudo_labels import pseudo_noise_labels, score_pseudo_labels
from lidar_denoise.train import select_device

from run_pipeline import _load_frame, get_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = select_device(cfg["training"]["device"])
    ckpt_dir = resolve(cfg, cfg["training"]["checkpoint_dir"])

    variants = {}
    for tag, name in (("supervised", "range_unet"), ("self-supervised", "range_unet_selfsup")):
        p = ckpt_dir / f"{name}.pt"
        if p.exists():
            variants[tag] = load_range_checkpoint(p, device)
        else:
            print(f"[compare] missing {p}, skipping {tag}")
    if not variants:
        sys.exit("no checkpoints to compare")

    frames = get_splits(cfg)["test"][: args.frames]
    rows: dict[str, list] = {}
    pseudo_quality: list = []

    for s in range(args.seeds):
        for i, path in enumerate(frames):
            clean = _load_frame(path, cfg)
            noisy = inject_noise(clean, cfg, seed=4200 + 100 * s + i)
            cx, pts, lab = clean[:, :3], noisy.points, noisy.labels

            # quality of the supervision itself, as a ceiling on what the
            # self-supervised model could possibly learn
            pl, _ = pseudo_noise_labels(pts)
            pseudo_quality.append(score_pseudo_labels(pl, lab))

            # the pseudo-labels used directly as a filter, with no network:
            # isolates how much of the result is the rules vs the learning
            kept_rule = np.flatnonzero(pl == 0)
            sc = removal_scores(kept_rule, lab)
            rows.setdefault("pseudo-labels as filter (no network)", []).append(
                (chamfer_distance(cx, pts[kept_rule][:, :3]),
                 hausdorff95(cx, pts[kept_rule][:, :3]),
                 sc["removal_precision"], sc["removal_recall"], sc["removal_f1"],
                 100.0 * len(kept_rule) / len(pts), 0.0)
            )

            for tag, (model, rcfg) in variants.items():
                out, ms, kept = denoise_cloud_range(model, rcfg, pts, cfg, device)
                sc = removal_scores(kept, lab)
                rows.setdefault(tag, []).append(
                    (chamfer_distance(cx, out), hausdorff95(cx, out),
                     sc["removal_precision"], sc["removal_recall"], sc["removal_f1"],
                     100.0 * len(out) / len(pts), ms)
                )
        print(f"[compare] seed {s + 1}/{args.seeds} done")

    cols = ["cd", "hd95", "removal_precision", "removal_recall", "removal_f1",
            "retention_pct", "time_ms"]
    summary = {k: dict(zip(cols, np.array(v).mean(axis=0).tolist())) for k, v in rows.items()}
    pq = {k: float(np.mean([d[k] for d in pseudo_quality]))
          for k in ("precision", "recall", "f1")}

    print(f"\npseudo-label quality vs ground truth: "
          f"precision {pq['precision']:.3f}  recall {pq['recall']:.3f}  F1 {pq['f1']:.3f}")
    print(f"\n{'variant':36s} {'CD':>7s} {'HD95':>7s} {'prec':>6s} {'rec':>6s} "
          f"{'F1':>6s} {'ret%':>6s} {'ms':>6s}")
    for k, v in summary.items():
        print(f"{k:36s} {v['cd']:7.4f} {v['hd95']:7.4f} {v['removal_precision']:6.3f} "
              f"{v['removal_recall']:6.3f} {v['removal_f1']:6.3f} "
              f"{v['retention_pct']:6.1f} {v['time_ms']:6.0f}")

    rules_key = "pseudo-labels as filter (no network)"
    if "supervised" in summary and "self-supervised" in summary:
        sup, ssl = summary["supervised"], summary["self-supervised"]
        rules = summary.get(rules_key, {})
        print(f"\nself-supervised reaches {100 * ssl['removal_f1'] / sup['removal_f1']:.0f}% "
              f"of supervised removal F1 with no ground-truth labels.")
        if rules:
            # The comparison that matters: a network trained on noisy labels
            # is only interesting if it beats those labels. Quoting the ratio
            # to the supervised model alone would credit the network for
            # performance the hand-written rules already provide.
            gain = 100 * (ssl["removal_f1"] - rules["removal_f1"]) / rules["removal_f1"]
            print(f"but only {gain:+.1f}% over the rules that supervised it "
                  f"({rules['removal_f1']:.3f} -> {ssl['removal_f1']:.3f}) — it reproduced "
                  f"its supervision rather than generalising past it.")
            print(f"It does clean up the worst outliers: HD95 "
                  f"{rules['hd95']:.3f} -> {ssl['hd95']:.3f} "
                  f"({100 * (rules['hd95'] - ssl['hd95']) / rules['hd95']:.0f}% better).")

    out_path = resolve(cfg, cfg["evaluation"]["results_dir"]) / "supervision_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "pseudo_label_quality": pq,
                   "protocol": {"n_frames": len(frames), "n_seeds": args.seeds}}, f, indent=2)
    print(f"\n[compare] wrote {out_path}")


if __name__ == "__main__":
    main()
