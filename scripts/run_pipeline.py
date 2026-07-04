#!/usr/bin/env python
"""End-to-end pipeline driver (proposal Figure 1).

Stages (each can be run alone via --stage):
  cache     build the paired-patch training cache from KITTI scans
  train     train pointnet_ae and dgcnn on the cache
  evaluate  CD/HD comparison of all methods on held-out test frames
  ablate    noise-level and SOR-parameter ablations
  optimize  throughput profiling + quantisation comparison

Usage:
  python scripts/run_pipeline.py --stage all
  python scripts/run_pipeline.py --stage train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lidar_denoise.config import load_config, resolve
from lidar_denoise.datasets import build_patch_cache
from lidar_denoise.kitti_io import list_scans, load_bin, split_scans
from lidar_denoise.preprocessing import preprocess
from lidar_denoise.noise import inject_noise
from lidar_denoise.filters import sor_ror_pipeline, sor_filter, removal_precision_recall
from lidar_denoise.metrics import evaluate_cloud, chamfer_distance
from lidar_denoise.train import train_model, select_device
from lidar_denoise.denoise import load_checkpoint, denoise_cloud
from lidar_denoise import visualize as viz


def get_splits(cfg):
    scans = list_scans([resolve(cfg, d) for d in cfg["data"]["raw_dirs"]])
    if not scans:
        sys.exit("No .bin scans found — run the data download first (see README).")
    return split_scans(scans, tuple(cfg["data"]["split"]), cfg["data"]["seed"])


def stage_cache(cfg):
    splits = get_splits(cfg)
    print(f"[cache] scans: {len(splits['train'])} train / {len(splits['val'])} val / "
          f"{len(splits['test'])} test")
    cache = build_patch_cache(
        splits["train"], cfg, resolve(cfg, "results/cache/train_patches.npz"),
        seed=cfg["data"]["seed"],
    )
    print(f"[cache] wrote {cache}")


def stage_train(cfg):
    cache = resolve(cfg, "results/cache/train_patches.npz")
    if not cache.exists():
        stage_cache(cfg)
    for name in ("pointnet_ae", "dgcnn"):
        train_model(name, cache, cfg)
        viz.training_curve(
            resolve(cfg, cfg["training"]["log_dir"]) / f"{name}_train.csv",
            resolve(cfg, "results/plots") / f"{name}_training_curve.png",
            f"{name} training loss",
        )


def _denoise_all_methods(noisy, cfg, models, device):
    """Run every method on one noisy cloud -> {method: (points, ms)}."""
    out = {}
    r = sor_ror_pipeline(noisy.points, cfg)
    out["SOR+ROR"] = (r["points"], r["total_ms"], r["kept_indices"])
    for name, (model, _) in models.items():
        pts, ms = denoise_cloud(model, name, noisy.points, cfg, device)
        out[name] = (pts, ms, None)
    return out


def stage_evaluate(cfg):
    splits = get_splits(cfg)
    device = select_device(cfg["training"]["device"])
    ckpt_dir = resolve(cfg, cfg["training"]["checkpoint_dir"])
    models = {
        name: load_checkpoint(name, ckpt_dir / f"{name}.pt", device)
        for name in ("pointnet_ae", "dgcnn")
        if (ckpt_dir / f"{name}.pt").exists()
    }
    if not models:
        sys.exit("No checkpoints found — run --stage train first.")

    test_frames = splits["test"][: cfg["evaluation"]["num_test_frames"]]
    per_method: dict[str, list[dict]] = {}
    example_saved = False
    results_dir = resolve(cfg, cfg["evaluation"]["results_dir"])

    for i, path in enumerate(test_frames):
        clean = preprocess(load_bin(path), cfg)
        noisy = inject_noise(clean, cfg, seed=1000 + i)
        method_outputs = _denoise_all_methods(noisy, cfg, models, device)
        for method, (pts, ms, kept) in method_outputs.items():
            m = evaluate_cloud(clean, pts, noisy.points)
            m["time_ms"] = ms
            if kept is not None:
                m.update(removal_precision_recall(kept, noisy.labels))
            per_method.setdefault(method, []).append(m)
        if not example_saved:
            # per-method before/after figures + one combined panel
            for method, (pts, _, _) in method_outputs.items():
                viz.before_after_figure(
                    {"clean": clean, "noisy": noisy.points, method: pts},
                    results_dir / "plots" / f"example_{method.replace('+', '_')}.png",
                    f"{path.name} — {method}",
                )
            panels = {"clean": clean, "noisy": noisy.points}
            panels.update({m: pts for m, (pts, _, _) in method_outputs.items()})
            viz.before_after_figure(
                panels, results_dir / "plots" / "example_all_methods.png",
                f"Frame {path.name}: statistical vs learning-based denoising",
            )
            example_saved = True
        print(f"[evaluate] frame {i + 1}/{len(test_frames)} done ({path.name})")

    # aggregate: mean over frames
    summary = {
        method: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for method, rows in per_method.items()
    }
    summary["noisy (no denoising)"] = {
        "cd": float(np.mean([r["cd_noisy"] for r in next(iter(per_method.values()))])),
        "hd": float(np.mean([r["hd_noisy"] for r in next(iter(per_method.values()))])),
    }
    out_path = results_dir / "metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump({"per_frame": per_method, "summary": summary}, f, indent=2)
    print(f"[evaluate] wrote {out_path}")

    # comparison table (markdown) + bar charts
    _write_markdown_table(summary, results_dir / "metrics_table.md")
    plot_table = {m: v for m, v in summary.items() if "cd" in v and "time_ms" in v}
    viz.metric_bars(plot_table, "cd", "Chamfer Distance (m)",
                    results_dir / "plots" / "chamfer_comparison.png",
                    "Chamfer Distance by method (lower is better)")
    viz.metric_bars(plot_table, "hd", "Hausdorff Distance (m)",
                    results_dir / "plots" / "hausdorff_comparison.png",
                    "Hausdorff Distance by method (lower is better)")
    viz.metric_bars(plot_table, "time_ms", "latency (ms/frame)",
                    results_dir / "plots" / "latency_comparison.png",
                    "Per-frame latency by method")


def _write_markdown_table(summary: dict, out_path: Path):
    cols = ["cd", "hd", "hd95", "cd_improvement_pct", "hd_improvement_pct",
            "retention_pct", "time_ms"]
    headers = ["Method", "CD (m)", "HD (m)", "HD95 (m)", "CD impr. %", "HD impr. %",
               "Retention %", "Time (ms)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for method, vals in summary.items():
        row = [method] + [f"{vals[c]:.4f}" if c in vals else "—" for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[evaluate] wrote {out_path}")


def stage_ablate(cfg):
    """Noise-intensity and SOR-parameter sweeps on a few test frames."""
    import copy

    splits = get_splits(cfg)
    device = select_device(cfg["training"]["device"])
    ckpt_dir = resolve(cfg, cfg["training"]["checkpoint_dir"])
    models = {
        name: load_checkpoint(name, ckpt_dir / f"{name}.pt", device)
        for name in ("pointnet_ae", "dgcnn")
        if (ckpt_dir / f"{name}.pt").exists()
    }
    frames = [preprocess(load_bin(p), cfg) for p in splits["test"][:3]]
    results_dir = resolve(cfg, cfg["evaluation"]["results_dir"])

    # ── ablation 1: Gaussian sigma sweep ─────────────────────────────────
    sigmas = [0.02, 0.05, 0.08, 0.12]
    series: dict[str, list[float]] = {}
    for sigma in sigmas:
        c = copy.deepcopy(cfg)
        c["noise"]["gaussian_sigma"] = sigma
        agg: dict[str, list[float]] = {}
        for i, clean in enumerate(frames):
            noisy = inject_noise(clean, c, seed=2000 + i)
            agg.setdefault("noisy", []).append(chamfer_distance(clean, noisy.points))
            r = sor_ror_pipeline(noisy.points, c)
            agg.setdefault("SOR+ROR", []).append(chamfer_distance(clean, r["points"]))
            for name, (model, _) in models.items():
                pts, _ = denoise_cloud(model, name, noisy.points, c, device)
                agg.setdefault(name, []).append(chamfer_distance(clean, pts))
        for k, v in agg.items():
            series.setdefault(k, []).append(float(np.mean(v)))
    viz.ablation_lines(sigmas, series, "Gaussian noise sigma (m)", "Chamfer Distance (m)",
                       results_dir / "plots" / "ablation_noise_sigma.png",
                       "Robustness to noise intensity")
    ablation = {"noise_sigma": {"x": sigmas, "series": series}}

    # ── ablation 2: SOR k / std_ratio sweep ──────────────────────────────
    ks = [8, 16, 20, 32, 48]
    stds = [1.0, 1.5, 2.0, 2.5, 3.0]
    clean = frames[0]
    noisy = inject_noise(clean, cfg, seed=3000)
    k_cd = [chamfer_distance(clean, sor_filter(noisy.points, k, 2.0)[0]) for k in ks]
    s_cd = [chamfer_distance(clean, sor_filter(noisy.points, 20, s)[0]) for s in stds]
    viz.ablation_lines(ks, {"SOR (std_ratio=2.0)": k_cd}, "nb_neighbors (k)",
                       "Chamfer Distance (m)",
                       results_dir / "plots" / "ablation_sor_k.png", "SOR sensitivity to k")
    viz.ablation_lines(stds, {"SOR (k=20)": s_cd}, "std_ratio (alpha)",
                       "Chamfer Distance (m)",
                       results_dir / "plots" / "ablation_sor_std.png",
                       "SOR sensitivity to std_ratio")
    ablation["sor_k"] = {"x": ks, "cd": k_cd}
    ablation["sor_std_ratio"] = {"x": stds, "cd": s_cd}

    with open(results_dir / "ablations.json", "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"[ablate] wrote {results_dir / 'ablations.json'}")


def stage_optimize(cfg):
    import torch

    from lidar_denoise.optimize import model_size_mb, profile_callable, quantize_dynamic

    splits = get_splits(cfg)
    device = select_device(cfg["training"]["device"])
    ckpt_dir = resolve(cfg, cfg["training"]["checkpoint_dir"])
    results_dir = resolve(cfg, cfg["evaluation"]["results_dir"])
    frames = []
    for i, p in enumerate(splits["test"][:5]):
        clean = preprocess(load_bin(p), cfg)
        frames.append(inject_noise(clean, cfg, seed=4000 + i).points)

    report: dict[str, dict] = {}
    report["SOR+ROR (CPU)"] = profile_callable(
        lambda x: sor_ror_pipeline(x, cfg), frames
    )
    for name in ("pointnet_ae", "dgcnn"):
        ckpt = ckpt_dir / f"{name}.pt"
        if not ckpt.exists():
            continue
        model, _ = load_checkpoint(name, ckpt, device)
        report[f"{name} ({device.type})"] = profile_callable(
            lambda x, m=model, n=name: denoise_cloud(m, n, x, cfg, device), frames
        )
        cpu_model, _ = load_checkpoint(name, ckpt, torch.device("cpu"))
        report[f"{name} (cpu fp32)"] = profile_callable(
            lambda x, m=cpu_model, n=name: denoise_cloud(m, n, x, cfg, torch.device("cpu")),
            frames,
        )
        q = quantize_dynamic(cpu_model)
        report[f"{name} (cpu int8-dynamic)"] = profile_callable(
            lambda x, m=q, n=name: denoise_cloud(m, n, x, cfg, torch.device("cpu")), frames
        )
        report[f"{name} (cpu int8-dynamic)"]["size_mb"] = model_size_mb(q)
        report[f"{name} (cpu fp32)"]["size_mb"] = model_size_mb(cpu_model)

    for method, stats in report.items():
        stats["meets_10hz"] = stats["hz"] >= 10.0
    with open(results_dir / "throughput.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[optimize] wrote {results_dir / 'throughput.json'}")
    for method, s in report.items():
        print(f"  {method:32s} {s['mean_ms']:9.1f} ms/frame  {s['hz']:6.2f} Hz"
              f"  {'✓ meets 10 Hz' if s['meets_10hz'] else '✗ below 10 Hz'}")


STAGES = {
    "cache": stage_cache,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "ablate": stage_ablate,
    "optimize": stage_optimize,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for name, fn in STAGES.items():
        if args.stage in (name, "all"):
            fn(cfg)
