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
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lidar_denoise.config import load_config, resolve
from lidar_denoise.datasets import build_patch_cache
from lidar_denoise.kitti_io import list_scans, load_bin, previous_scan, split_scans
from lidar_denoise.preprocessing import preprocess
from lidar_denoise.noise import apply_severity, inject_noise
from lidar_denoise.filters import (
    idsor_filter,
    sor_filter,
    sor_ror_pipeline,
    temporal_filter,
)
from lidar_denoise.metrics import (
    chamfer_distance,
    evaluate_cloud,
    oracle_floor,
    removal_scores,
)
from lidar_denoise.train import train_model, select_device
from lidar_denoise.denoise import (
    denoise_cloud,
    denoise_cloud_range,
    load_checkpoint,
    load_range_checkpoint,
)
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
    for split in ("train", "val"):
        cache = build_patch_cache(
            splits[split], cfg, resolve(cfg, f"results/cache/{split}_patches.npz"),
            seed=cfg["data"]["seed"],
        )
        print(f"[cache] wrote {cache}")


def stage_train(cfg):
    cache = resolve(cfg, "results/cache/train_patches.npz")
    val_cache = resolve(cfg, "results/cache/val_patches.npz")
    if not cache.exists() or not val_cache.exists():
        stage_cache(cfg)
    for name in ("pointnet_ae", "dgcnn"):
        train_model(name, cache, cfg, val_cache_path=val_cache,
                    resume=cfg["training"].get("resume", False))
        viz.training_curve(
            resolve(cfg, cfg["training"]["log_dir"]) / f"{name}_train.csv",
            resolve(cfg, "results/plots") / f"{name}_training_curve.png",
            f"{name} train / validation loss",
        )


def _denoise_all_methods(noisy, cfg, models, device, prev_noisy=None, range_model=None):
    """Run every method on one noisy cloud -> {method: (points, ms, kept_idx)}.

    The noisy cloud is (N, 4) XYZI when intensity is enabled. Geometry-only
    methods are handed `[:, :3]`; IDSOR and the range-image network receive
    the full array, since the whole point of both is that intensity carries
    information geometry does not.

    `prev_noisy` is the (also noisy) previous scan of the sequence; when
    available, the temporal-consistency filter chained with SOR is evaluated
    as an additional statistical method.
    """
    pts_all = noisy.points
    xyz = pts_all[:, :3]
    has_intensity = pts_all.shape[1] > 3
    out = {}

    r = sor_ror_pipeline(xyz, cfg)
    out["SOR+ROR"] = (r["points"], r["total_ms"], r["kept_indices"])

    if has_intensity:
        i_pts, i_idx, i_ms = idsor_filter(pts_all, **cfg["filters"].get("idsor", {}))
        out["IDSOR"] = (i_pts[:, :3], i_ms, i_idx)

    if prev_noisy is not None:
        t_pts, t_idx, t_ms = temporal_filter(xyz, prev_noisy[:, :3])
        s_pts, s_idx, s_ms = sor_filter(t_pts, **cfg["filters"]["sor"])
        out["Temporal+SOR"] = (s_pts, t_ms + s_ms, t_idx[s_idx])

    for name, (model, _) in models.items():
        pts, ms, kept = denoise_cloud(model, name, xyz, cfg, device)
        out[name] = (pts, ms, kept)

    if range_model is not None and has_intensity:
        model, rcfg = range_model
        pts, ms, kept = denoise_cloud_range(model, rcfg, pts_all, cfg, device)
        out["RangeRestoreNet"] = (pts, ms, kept)

    return out


def _load_frame(path, cfg):
    """Preprocessed cloud, carrying intensity when the config asks for it."""
    return preprocess(load_bin(path, with_intensity=cfg["data"].get("use_intensity", False)), cfg)


def _prev_noisy_cloud(path, cfg, seed):
    """Preprocessed + noise-injected previous scan, or None at sequence start.

    The previous sweep gets its own independent noise draw — the temporal
    filter never sees clean data.
    """
    prev = previous_scan(path)
    if prev is None:
        return None
    return inject_noise(_load_frame(prev, cfg), cfg, seed=seed).points


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
    range_ckpt = ckpt_dir / "range_unet.pt"
    range_model = (
        load_range_checkpoint(range_ckpt, device) if range_ckpt.exists() else None
    )
    if range_model is None:
        print("[evaluate] no range_unet.pt — skipping RangeRestoreNet "
              "(train it with scripts/train_range.py)")

    test_frames = splits["test"][: cfg["evaluation"]["num_test_frames"]]
    n_seeds = cfg["evaluation"].get("num_seeds", 1)
    per_method: dict[str, list[dict]] = {}
    example_saved = False
    results_dir = resolve(cfg, cfg["evaluation"]["results_dir"])

    # Every method sees the identical noisy cloud for a given (frame, seed),
    # so CD differences are paired and a paired test is the right one.
    for s in range(n_seeds):
        for i, path in enumerate(test_frames):
            clean = _load_frame(path, cfg)
            noisy = inject_noise(clean, cfg, seed=1000 + 100 * s + i)
            prev_noisy = _prev_noisy_cloud(path, cfg, seed=9000 + 100 * s + i)
            method_outputs = _denoise_all_methods(
                noisy, cfg, models, device, prev_noisy, range_model
            )
            # Metrics are geometric: slice off any intensity channel so a
            # 4-column cloud never reaches a k-d tree built on 3 columns.
            clean_xyz, noisy_xyz = clean[:, :3], noisy.points[:, :3]

            # Ceiling for any removal-based method on this exact frame.
            floor = oracle_floor(clean_xyz, noisy_xyz, noisy.labels)
            floor.update({"time_ms": 0.0, "frame": path.name, "seed": s,
                          "removal_precision": 1.0, "removal_recall": 1.0,
                          "removal_f1": 1.0})
            per_method.setdefault("oracle (perfect removal)", []).append(floor)

            for method, (pts, ms, kept) in method_outputs.items():
                m = evaluate_cloud(clean_xyz, pts[:, :3], noisy_xyz)
                m["time_ms"] = ms
                m["frame"] = path.name
                m["seed"] = s
                if kept is not None:
                    m.update(removal_scores(kept, noisy.labels))
                per_method.setdefault(method, []).append(m)

            if not example_saved:
                for method, (pts, _, _) in method_outputs.items():
                    viz.before_after_figure(
                        {"clean": clean_xyz, "noisy": noisy_xyz, method: pts[:, :3]},
                        results_dir / "plots" / f"example_{method.replace('+', '_')}.png",
                        f"{path.name} — {method}",
                    )
                panels = {"clean": clean_xyz, "noisy": noisy_xyz}
                panels.update({m: pts[:, :3] for m, (pts, _, _) in method_outputs.items()})
                viz.before_after_figure(
                    panels, results_dir / "plots" / "example_all_methods.png",
                    f"Frame {path.name}: statistical vs learning-based denoising",
                )
                example_saved = True
            print(f"[evaluate] seed {s + 1}/{n_seeds} frame {i + 1}/{len(test_frames)} "
                  f"({path.name})")

    # Aggregate with dispersion; nanmean so a degenerate frame is excluded
    # from the mean rather than poisoning it, and counted separately.
    summary = {}
    for method, rows in per_method.items():
        numeric = [k for k in rows[0] if isinstance(rows[0][k], (int, float, bool))]
        stats = {}
        for k in numeric:
            vals = np.array([r[k] for r in rows], dtype=float)
            stats[k] = float(np.nanmean(vals))
            stats[f"{k}_std"] = float(np.nanstd(vals))
        stats["n_observations"] = len(rows)
        stats["n_degenerate"] = int(sum(bool(r.get("degenerate")) for r in rows))
        summary[method] = stats
    # The oracle rows are inserted first but carry no cd_noisy; take the
    # baseline figures from any real method (all see the same noisy cloud).
    ref_rows = next(r for r in per_method.values() if "cd_noisy" in r[0])
    summary["noisy (no denoising)"] = {
        "cd": float(np.nanmean([r["cd_noisy"] for r in ref_rows])),
        "cd_std": float(np.nanstd([r["cd_noisy"] for r in ref_rows])),
        "hd": float(np.nanmean([r["hd_noisy"] for r in ref_rows])),
    }

    _add_oracle_gap(summary)
    significance = _paired_tests(per_method, key="cd")
    out_path = results_dir / "metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {"per_frame": per_method, "summary": summary,
             "significance_cd": significance,
             "protocol": {"n_frames": len(test_frames), "n_seeds": n_seeds,
                          "paired": True, "test": "Wilcoxon signed-rank"}},
            f, indent=2,
        )
    print(f"[evaluate] wrote {out_path}")

    # comparison table (markdown) + bar charts
    _write_markdown_table(summary, results_dir / "metrics_table.md")
    plot_table = {m: v for m, v in summary.items() if "cd" in v and "time_ms" in v}
    viz.metric_bars(plot_table, "cd", "Chamfer Distance (m)",
                    results_dir / "plots" / "chamfer_comparison.png",
                    "Chamfer Distance by method (lower is better)")
    # HD95 rather than strict HD: see _write_markdown_table.
    hd_table = {m: v for m, v in plot_table.items() if np.isfinite(v.get("hd95", np.nan))}
    viz.metric_bars(hd_table, "hd95", "HD95 (m)",
                    results_dir / "plots" / "hausdorff_comparison.png",
                    "95th-percentile Hausdorff Distance by method (lower is better)")
    f1_table = {m: v for m, v in plot_table.items()
                if np.isfinite(v.get("removal_f1", np.nan))}
    if f1_table:
        viz.metric_bars(f1_table, "removal_f1", "noise-removal F1",
                        results_dir / "plots" / "removal_f1_comparison.png",
                        "Noise-removal F1 by method (higher is better)")
    viz.metric_bars(plot_table, "time_ms", "latency (ms/frame)",
                    results_dir / "plots" / "latency_comparison.png",
                    "Per-frame latency by method")


def _add_oracle_gap(summary: dict) -> None:
    """Express each method's error as a fraction of the *recoverable* error.

    The headline framing of this project. A raw Chamfer Distance is
    uninterpretable on its own here, because the weather model deletes ~31%
    of returns that no denoiser can restore, so zero error is unreachable.
    The oracle — delete every injected point, keep every real one — is the
    best any purely removal-based method can do, and

        oracle_gap_pct = 100 * (cd_method - cd_oracle) / (cd_noisy - cd_oracle)

    reports what fraction of the *available* improvement a method left on the
    table. 0% means it matched a perfect noise classifier; 100% means it
    achieved nothing over the corrupted input.

    A method can score BELOW 0%, and that is the interesting case rather than
    a bug: the oracle only removes points, while a method that also corrects
    positions can beat it. Every adverse-weather denoiser in the current
    literature is removal-only, so this measures how much is being left
    behind by that framing.
    """
    if "oracle (perfect removal)" not in summary or "noisy (no denoising)" not in summary:
        return
    cd_o = summary["oracle (perfect removal)"]["cd"]
    cd_n = summary["noisy (no denoising)"]["cd"]
    span = cd_n - cd_o
    if not np.isfinite(span) or span <= 0:
        return
    for name, vals in summary.items():
        cd = vals.get("cd")
        if cd is None or not np.isfinite(cd):
            continue
        vals["oracle_gap_pct"] = float(100.0 * (cd - cd_o) / span)
        vals["beats_removal_oracle"] = bool(cd < cd_o)


def _paired_tests(per_method: dict[str, list[dict]], key: str = "cd") -> dict:
    """Wilcoxon signed-rank test on `key` for every pair of methods.

    Observations are paired by (frame, seed): all methods are run on the
    identical noisy cloud, so the paired test is both valid and far more
    powerful than an unpaired one at n = frames x seeds. Reporting this is
    what licenses a claim like "A beats B" rather than "A scored lower than
    B on our ten frames".
    """
    from scipy.stats import wilcoxon

    keyed = {
        m: {(r["frame"], r["seed"]): r[key] for r in rows}
        for m, rows in per_method.items()
    }
    names = sorted(keyed)
    out = {}
    for a, b in itertools.combinations(names, 2):
        shared = sorted(set(keyed[a]) & set(keyed[b]))
        pairs = [(keyed[a][s], keyed[b][s]) for s in shared]
        pairs = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y)]
        if len(pairs) < 6 or all(x == y for x, y in pairs):
            continue
        xa, xb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        try:
            stat, p = wilcoxon(xa, xb)
        except ValueError:
            continue
        out[f"{a} vs {b}"] = {
            "n_pairs": len(pairs),
            "median_diff": float(np.median(xa - xb)),
            "statistic": float(stat),
            "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05),
            "better": a if np.median(xa - xb) < 0 else b,
        }
    return out


def _write_markdown_table(summary: dict, out_path: Path):
    """Results table: CD/HD95 with dispersion, removal F1, retention, latency.

    Strict HD is dropped from the headline table. It is the max over all
    residual points, so one surviving outlier at the 60 m range gate pins it
    near 30 m for every method — the column ranked nothing. HD95 carries the
    same information robustly and is kept; raw HD stays in the JSON.
    """
    cols = ["cd", "hd95", "oracle_gap_pct", "removal_f1", "retention_pct", "time_ms"]
    headers = ["Method", "CD (m) ↓", "HD95 (m) ↓", "Oracle gap % ↓", "Removal F1 ↑",
               "Retention %", "Time (ms)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for method, vals in summary.items():
        row = [method]
        for c in cols:
            if c not in vals or not np.isfinite(vals[c]):
                row.append("—")
            elif f"{c}_std" in vals and c in ("cd", "hd95"):
                row.append(f"{vals[c]:.3f} ± {vals[f'{c}_std']:.3f}")
            else:
                row.append(f"{vals[c]:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    degenerate = {m: v.get("n_degenerate", 0) for m, v in summary.items()}
    if any(degenerate.values()):
        note = ", ".join(f"{m}: {n}" for m, n in degenerate.items() if n)
        lines += ["", f"Frames left empty by the method (excluded from means) — {note}."]
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
    test_paths = splits["test"][:3]
    frames = [_load_frame(p, cfg) for p in test_paths]
    results_dir = resolve(cfg, cfg["evaluation"]["results_dir"])

    # ── ablation 1: corruption severity sweep (Robo3D convention) ────────
    severities = list(cfg["noise"]["severity_presets"])
    series: dict[str, list[float]] = {}
    for sev in severities:
        c = apply_severity(cfg, sev)
        agg: dict[str, list[float]] = {}
        for i, clean in enumerate(frames):
            noisy = inject_noise(clean, c, seed=2000 + i)
            agg.setdefault("noisy", []).append(chamfer_distance(clean, noisy.points))
            prev_noisy = _prev_noisy_cloud(test_paths[i], c, seed=9500 + i)
            for method, (pts, _, _) in _denoise_all_methods(
                noisy, c, models, device, prev_noisy
            ).items():
                agg.setdefault(method, []).append(chamfer_distance(clean, pts))
        for k, v in agg.items():
            series.setdefault(k, []).append(float(np.mean(v)))
    viz.ablation_lines(list(range(len(severities))), series,
                       f"corruption severity ({' / '.join(severities)})",
                       "Chamfer Distance (m)",
                       results_dir / "plots" / "ablation_severity.png",
                       "Robustness across severity levels (rain rate 10/25/50 mm/h)")
    ablation = {"severity": {"levels": severities, "series": series}}

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

    # ── ablation 3: AE patch recombination ───────────────────────────────
    # Quantifies how much of the PointNet-AE's apparent failure was the
    # recombination step rather than the model: the legacy `union` mode
    # returns ~1.7x the input points, inflating CD from the denoised side.
    if "pointnet_ae" in models:
        recomb = {}
        for mode in ("nn_project", "union"):
            c = copy.deepcopy(cfg)
            c["patches"]["ae_recombination"] = mode
            cds, rets = [], []
            for i, clean in enumerate(frames):
                noisy = inject_noise(clean, c, seed=3100 + i)
                pts, _, _ = denoise_cloud(
                    models["pointnet_ae"][0], "pointnet_ae", noisy.points, c, device
                )
                cds.append(chamfer_distance(clean, pts))
                rets.append(100.0 * len(pts) / len(noisy.points))
            recomb[mode] = {"cd": float(np.mean(cds)),
                            "retention_pct": float(np.mean(rets))}
        ablation["ae_recombination"] = recomb
        print(f"[ablate] AE recombination: {recomb}")

    # ── ablation 4: temporal filter with / without ego-motion ICP ────────
    # Inter-sweep translation on these drives averages 0.56 m against a
    # 0.5 m consistency radius, so this is not a refinement but a
    # correctness requirement.
    ego = {}
    for compensate in (True, False):
        cds, rets = [], []
        for i, clean in enumerate(frames):
            noisy = inject_noise(clean, cfg, seed=3200 + i)
            prev_noisy = _prev_noisy_cloud(test_paths[i], cfg, seed=9600 + i)
            if prev_noisy is None:
                continue
            t_pts, _, _ = temporal_filter(
                noisy.points, prev_noisy, compensate_ego_motion=compensate
            )
            cds.append(chamfer_distance(clean, t_pts))
            rets.append(100.0 * len(t_pts) / len(noisy.points))
        if cds:
            ego["icp" if compensate else "uncompensated"] = {
                "cd": float(np.nanmean(cds)), "retention_pct": float(np.mean(rets))
            }
    if ego:
        ablation["temporal_ego_motion"] = ego
        print(f"[ablate] temporal ego-motion: {ego}")

    # ── ablation 5: DGCNN outlier-probability threshold ──────────────────
    # 0.5 is the default but nothing justified it; this is the model's
    # precision/recall dial and it trades CD against retention directly.
    if "dgcnn" in models:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        sweep = {"x": thresholds, "cd": [], "removal_f1": [], "retention_pct": []}
        for th in thresholds:
            cds, f1s, rets = [], [], []
            for i, clean in enumerate(frames):
                noisy = inject_noise(clean, cfg, seed=3300 + i)
                pts, _, kept = denoise_cloud(
                    models["dgcnn"][0], "dgcnn", noisy.points, cfg, device,
                    outlier_threshold=th,
                )
                cds.append(chamfer_distance(clean, pts))
                f1s.append(removal_scores(kept, noisy.labels)["removal_f1"])
                rets.append(100.0 * len(pts) / len(noisy.points))
            sweep["cd"].append(float(np.nanmean(cds)))
            sweep["removal_f1"].append(float(np.mean(f1s)))
            sweep["retention_pct"].append(float(np.mean(rets)))
        viz.ablation_lines(
            thresholds, {"CD (m)": sweep["cd"], "removal F1": sweep["removal_f1"]},
            "DGCNN outlier probability threshold", "value",
            results_dir / "plots" / "ablation_dgcnn_threshold.png",
            "DGCNN removal threshold: geometry vs classification",
        )
        ablation["dgcnn_threshold"] = sweep
        print(f"[ablate] dgcnn threshold sweep: {sweep}")

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
