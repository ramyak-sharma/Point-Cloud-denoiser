#!/usr/bin/env python
"""Generate every figure used by report/final_report.tex.

Design rules followed throughout (and why, since they are not arbitrary):

* **Magnitude-by-category charts use ONE hue, not a categorical palette.**
  A bar chart of "Chamfer Distance by method" encodes one measure across
  categories; giving each bar its own hue implies six series that do not
  exist. Emphasis is carried by a single accent, and reference values by a
  rule line.
* **Scatter plots carry identity in direct labels, not colour.** With seven
  methods on screen no eight-hue palette clears colour-vision separation on
  the all-pairs list, so colour would be decorative at best and misleading
  at worst. Every point is labelled instead.
* **Line charts may use the categorical order** (their pairlist is adjacent
  only), and are direct-labelled because three of the slots sit below 3:1
  against the surface.
* Vector PDF for anything with text; PNG only for the point-cloud rasters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "report" / "figures"

# ── design tokens (validated: see scripts/validate_palette.js in the skill) ──
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ACCENT = "#2a78d6"
ACCENT_HI = "#eb6834"
REFERENCE = "#52514e"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.edgecolor": "#d8d7d2",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#e8e7e2",
    "grid.linewidth": 0.8,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "figure.dpi": 150,
})

ORDER = ["oracle (perfect removal)", "dgcnn", "IDSOR", "RangeRestoreNet",
         "SOR+ROR", "Temporal+SOR", "pointnet_ae"]
PRETTY = {
    "oracle (perfect removal)": "Oracle",
    "dgcnn": "DGCNN",
    "IDSOR": "IDSOR",
    "RangeRestoreNet": "RangeRestoreNet",
    "SOR+ROR": "SOR+ROR",
    "Temporal+SOR": "Temporal+SOR",
    "pointnet_ae": "PointNet-AE",
    "noisy (no denoising)": "Noisy input",
    "noisy": "Noisy input",
}


def load():
    r = ROOT / "results"
    j = lambda n: json.load(open(r / n))
    return (j("metrics_summary.json"), j("detection_consistency.json"),
            j("ablations.json"), j("supervision_comparison.json"))


def _barlabels(ax, bars, values, fmt="{:.3f}", dy=0.01, color=INK):
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * span,
                fmt.format(v), ha="center", va="bottom", fontsize=8, color=color)


# ── 1. reconstruction quality + oracle gap ──────────────────────────────────
def fig_reconstruction(m):
    s = m["summary"]
    names = [n for n in ORDER if n in s]
    cd = [s[n]["cd"] for n in names]
    err = [s[n].get("cd_std", 0.0) for n in names]
    gap = [s[n].get("oracle_gap_pct", np.nan) for n in names]
    noisy = s["noisy (no denoising)"]["cd"]

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    colors = [REFERENCE if n.startswith("oracle") else
              (ACCENT_HI if n in ("dgcnn",) else ACCENT) for n in names]
    bars = ax.bar(range(len(names)), cd, yerr=err, capsize=2.5, width=0.62,
                  color=colors, error_kw=dict(ecolor=MUTED, lw=0.9))
    ax.axhline(noisy, color=CRITICAL, lw=1.4, ls="--", zorder=0)
    ax.text(-0.35, noisy, f"no denoising  {noisy:.3f}", va="bottom",
            ha="left", fontsize=8, color=CRITICAL)
    oracle_cd = s["oracle (perfect removal)"]["cd"]
    ax.axhline(oracle_cd, color=REFERENCE, lw=1.0, ls=":", zorder=0)
    ax.text(len(names) - 0.45, oracle_cd, f"perfect removal  {oracle_cd:.3f}",
            va="bottom", ha="right", fontsize=8, color=REFERENCE)

    for i, (b, g) in enumerate(zip(bars, gap)):
        if np.isfinite(g):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{g:.1f}%", ha="center", va="bottom", fontsize=8,
                    color=INK_2, fontweight="bold" if g < 5 else "normal")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PRETTY[n] for n in names], fontsize=8,
                       rotation=22, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Chamfer Distance (m)")
    ax.set_ylim(0, max(cd) * 1.35)
    ax.set_title("Reconstruction error, with the fraction of recoverable error left behind",
                 loc="left", color=INK)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reconstruction.pdf")
    plt.close(fig)


# ── 2. removal precision / recall / F1 ──────────────────────────────────────
def fig_removal(m):
    s = m["summary"]
    names = [n for n in ORDER if n in s and np.isfinite(s[n].get("removal_f1", np.nan))
             and not n.startswith("oracle")]
    prec = [s[n]["removal_precision"] for n in names]
    rec = [s[n]["removal_recall"] for n in names]
    f1 = [s[n]["removal_f1"] for n in names]

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.bar(x - w, prec, w * 0.92, label="precision", color=CAT[0])
    ax.bar(x, rec, w * 0.92, label="recall", color=CAT[1])
    ax.bar(x + w, f1, w * 0.92, label="F1", color=CAT[2])
    for xi, v in zip(x + w, f1):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in names], fontsize=8,
                       rotation=22, ha="right", rotation_mode="anchor")
    ax.set_ylabel("noise-removal score")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    ax.set_title("How well each method identifies injected noise (higher is better)",
                 loc="left", color=INK)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_removal_quality.pdf")
    plt.close(fig)


# ── 3. downstream detection ─────────────────────────────────────────────────
def fig_detection(d):
    s, sig = d["summary"], d["significance_vs_oracle"]
    names = ["noisy"] + [n for n in ORDER if n in s]
    rec = [s[n]["recall"] for n in names]
    err = [s[n]["recall_std"] / np.sqrt(s[n]["n_observations"]) for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    colors = [REFERENCE if n.startswith("oracle") else
              (CRITICAL if n == "pointnet_ae" else
               (MUTED if n == "noisy" else ACCENT)) for n in names]
    bars = ax.bar(range(len(names)), rec, yerr=err, capsize=2.5, width=0.62,
                  color=colors, error_kw=dict(ecolor=MUTED, lw=0.9))
    oracle = s["oracle (perfect removal)"]["recall"]
    ax.axhline(oracle, color=REFERENCE, lw=1.0, ls=":", zorder=0)

    for i, n in enumerate(names):
        key = f"{n} vs oracle (perfect removal)"
        if key in sig:
            mark = "n.s." if not sig[key]["significant_at_0.05"] else "*"
            ax.text(i, rec[i] + err[i] + 0.022, mark, ha="center", fontsize=8,
                    color=GOOD if mark == "n.s." else INK_2,
                    fontweight="bold" if mark == "n.s." else "normal")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PRETTY[n] for n in names], fontsize=8,
                       rotation=22, ha="right", rotation_mode="anchor")
    ax.set_ylabel("detection-consistency recall")
    ax.set_ylim(0, max(rec) * 1.35)
    ax.set_title("Detection recall vs the perfect-removal oracle\n"
                 "* significantly worse   ·   n.s. = indistinguishable from it",
                 loc="left", color=INK, fontsize=9.5)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.text(0.012, 0.015, "84 paired observations · error bars are standard error · "
             "Wilcoxon signed-rank vs oracle", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / "fig_detection.pdf")
    plt.close(fig)


# ── 4. the headline divergence: CD vs detection ─────────────────────────────
def fig_cd_vs_detection(m, d):
    ms, ds = m["summary"], d["summary"]
    names = [n for n in ORDER if n in ms and n in ds]
    cd = [ms[n]["cd"] for n in names]
    rec = [ds[n]["recall"] for n in names]
    n_cd, n_rec = ms["noisy (no denoising)"]["cd"], ds["noisy"]["recall"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # identity carried by labels, not colour — see module docstring
    ax.scatter(cd, rec, s=64, color=ACCENT, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter([n_cd], [n_rec], s=64, color=MUTED, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter([ms["pointnet_ae"]["cd"]], [ds["pointnet_ae"]["recall"]], s=64,
               color=CRITICAL, zorder=4, edgecolor=SURFACE, linewidth=1.5)

    # Hand-placed offsets: seven labels in a small space collide under any
    # automatic rule, and the whole point of the figure is that each method
    # is individually identifiable.
    offsets = {
        "oracle (perfect removal)": (10, 2, "left"),
        "dgcnn": (-6, -12, "right"),
        "IDSOR": (0, -16, "center"),
        "RangeRestoreNet": (0, 14, "center"),
        "SOR+ROR": (0, -16, "center"),
        "Temporal+SOR": (10, 2, "left"),
        "pointnet_ae": (-10, 2, "right"),
        "noisy (no denoising)": (-10, 2, "right"),
    }
    for n, x, y in list(zip(names, cd, rec)) + [("noisy (no denoising)", n_cd, n_rec)]:
        dx, dy, ha = offsets.get(n, (8, 6, "left"))
        ax.annotate(PRETTY[n].replace("\n", " "), (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=INK_2,
                    ha=ha, va="center")

    # No "better on both" arrow: with seven labelled points there is no space
    # for it that does not collide, and both axis labels already state their
    # direction. Chart junk that fights the data loses.

    ax.set_xlabel("Chamfer Distance (m) — lower is better")
    ax.set_ylabel("detection recall — higher is better")
    ax.set_title("Reconstruction error does not predict perception quality",
                 loc="left", color=INK)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(0.05, 0.62)
    ax.set_ylim(0.0, 0.58)
    fig.text(0.012, 0.015, "PointNet-AE improves CD over the noisy input yet loses "
             "76% of the detector's recall (p < 0.0001).", fontsize=7.5, color=CRITICAL)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig_cd_vs_detection.pdf")
    plt.close(fig)


# ── 5. accuracy / latency trade-off ─────────────────────────────────────────
def fig_latency(m):
    s = m["summary"]
    names = [n for n in ORDER if n in s and s[n].get("time_ms", 0) > 0]
    ms_ = [s[n]["time_ms"] for n in names]
    gap = [s[n]["oracle_gap_pct"] for n in names]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axvspan(1, 100, color="#eef5fd", zorder=0)
    ax.axvline(100, color=GOOD, lw=1.2, ls="--", zorder=1)
    ax.text(97, 88, "10 Hz budget", rotation=90, va="top", ha="right",
            fontsize=8, color=GOOD)
    ax.scatter(ms_, gap, s=64, color=ACCENT, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for n, x, y in zip(names, ms_, gap):
        ha = "right" if x > 500 else "left"
        ax.annotate(f"  {PRETTY[n]}  ", (x, y), fontsize=8, color=INK_2,
                    ha=ha, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("latency per frame (ms, log scale) — lower is better")
    ax.set_ylabel("oracle gap (%) — lower is better")
    ax.set_title("Accuracy against cost; shaded region meets the 10 Hz target",
                 loc="left", color=INK)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-5, 95)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency_accuracy.pdf")
    plt.close(fig)


# ── 6. training curves ──────────────────────────────────────────────────────
def fig_training():
    import csv as _csv
    logs = [("pointnet_ae_train.csv", "PointNet-AE"),
            ("dgcnn_train.csv", "DGCNN"),
            ("range_unet_train.csv", "RangeRestoreNet")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (fn, title) in zip(axes, logs):
        path = ROOT / "results" / "logs" / fn
        if not path.exists():
            ax.axis("off"); continue
        ep, tr, va = [], [], []
        with open(path) as f:
            for row in _csv.DictReader(f):
                ep.append(int(row["epoch"])); tr.append(float(row["loss"]))
                va.append(float(row.get("val_loss") or "nan"))
        ax.plot(ep, tr, color=CAT[0], lw=1.6, label="train")
        ax.plot(ep, va, color=CAT[1], lw=1.6, ls="--", label="validation")
        if np.isfinite(va).any():
            b = int(np.nanargmin(va))
            ax.scatter([ep[b]], [va[b]], s=28, color=CAT[1], zorder=4,
                       edgecolor=SURFACE, linewidth=1.2)
            ax.annotate(f"best e{ep[b]}", (ep[b], va[b]), fontsize=7,
                        color=INK_2, xytext=(0, -12), textcoords="offset points",
                        ha="center")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(True); ax.set_axisbelow(True)
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Training and validation loss; the retained checkpoint is the validation minimum",
                 x=0.012, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig_training.pdf")
    plt.close(fig)


# ── 7. severity ablation ────────────────────────────────────────────────────
def fig_severity(a):
    sev = a["severity"]
    levels, series = sev["levels"], sev["series"]
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    keys = [k for k in ["noisy", "dgcnn", "IDSOR", "RangeRestoreNet", "SOR+ROR",
                        "Temporal+SOR", "pointnet_ae"] if k in series][:6]
    for i, k in enumerate(keys):
        ax.plot(x, series[k], lw=1.8, color=CAT[i % len(CAT)], marker="o", ms=4)
        ax.annotate(f" {PRETTY.get(k, k)}", (x[-1], series[k][-1]), fontsize=8,
                    color=INK_2, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{r} mm/h" for l, r in zip(levels, [10, 25, 50])], fontsize=8)
    ax.set_ylabel("Chamfer Distance (m)")
    ax.set_xlim(-0.15, len(levels) - 0.35)
    ax.set_title("Robustness across corruption severity", loc="left", color=INK)
    ax.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation_severity.pdf")
    plt.close(fig)


# ── 8. parameter sensitivity ────────────────────────────────────────────────
def fig_params(a):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axes[0].plot(a["sor_k"]["x"], a["sor_k"]["cd"], color=CAT[0], lw=1.8, marker="o", ms=4)
    axes[0].set_xlabel("SOR $k$"); axes[0].set_title("SOR neighbours", loc="left", fontsize=9)
    axes[1].plot(a["sor_std_ratio"]["x"], a["sor_std_ratio"]["cd"], color=CAT[0],
                 lw=1.8, marker="o", ms=4)
    axes[1].set_xlabel(r"SOR $\alpha$"); axes[1].set_title("SOR threshold", loc="left", fontsize=9)
    if "dgcnn_threshold" in a:
        t = a["dgcnn_threshold"]
        axes[2].plot(t["x"], t["cd"], color=CAT[0], lw=1.8, marker="o", ms=4)
        axes[2].annotate(" CD", (t["x"][-1], t["cd"][-1]), fontsize=8, color=INK_2)
        ax2 = axes[2]
        ax2.set_xlabel("DGCNN removal threshold")
        ax2.set_title("DGCNN operating point", loc="left", fontsize=9)
        best = int(np.argmin(t["cd"]))
        ax2.scatter([t["x"][best]], [t["cd"][best]], s=34, color=ACCENT_HI, zorder=4)
    for ax in axes:
        ax.grid(True); ax.set_axisbelow(True)
    axes[0].set_ylabel("Chamfer Distance (m)")
    fig.suptitle("Parameter sensitivity: none of the three is a knife-edge",
                 x=0.012, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig_ablation_params.pdf")
    plt.close(fig)


# ── 9. the two fixes that changed results ───────────────────────────────────
def fig_fixes(a):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    r = a["ae_recombination"]
    labels = ["nn_project\n(corrected)", "union\n(legacy)"]
    cd = [r["nn_project"]["cd"], r["union"]["cd"]]
    ret = [r["nn_project"]["retention_pct"], r["union"]["retention_pct"]]
    b = axes[0].bar(labels, cd, width=0.5, color=[ACCENT, CRITICAL])
    _barlabels(axes[0], b, cd)
    for xi, v in enumerate(ret):
        axes[0].text(xi, 0.02, f"{v:.0f}% of input points", ha="center",
                     fontsize=7.5, color=SURFACE, fontweight="bold")
    axes[0].set_ylabel("Chamfer Distance (m)")
    axes[0].set_title("Chamfer Distance is gameable by densification",
                      loc="left", fontsize=9)
    axes[0].set_ylim(0, max(cd) * 1.25)

    e = a["temporal_ego_motion"]
    labels2 = ["ICP compensated", "uncompensated"]
    cd2 = [e["icp"]["cd"], e["uncompensated"]["cd"]]
    b2 = axes[1].bar(labels2, cd2, width=0.5, color=[ACCENT, CRITICAL])
    _barlabels(axes[1], b2, cd2)
    for xi, k in enumerate(["icp", "uncompensated"]):
        axes[1].text(xi, 0.008, f"{e[k]['retention_pct']:.0f}% retained", ha="center",
                     fontsize=7.5, color=SURFACE, fontweight="bold")
    axes[1].set_title("Ego-motion compensation is required, not optional",
                      loc="left", fontsize=9)
    axes[1].set_ylim(0, max(cd2) * 1.25)
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation_fixes.pdf")
    plt.close(fig)


# ── 10. self-supervision ────────────────────────────────────────────────────
def fig_selfsup(c):
    s = c["summary"]
    keys = ["pseudo-labels as filter (no network)", "self-supervised", "supervised"]
    lbl = ["Rules only\n(no network)", "Self-supervised\nnetwork", "Supervised\nnetwork"]
    f1 = [s[k]["removal_f1"] for k in keys]
    cd = [s[k]["cd"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    b = axes[0].bar(lbl, f1, width=0.55, color=[MUTED, CAT[1], ACCENT])
    _barlabels(axes[0], b, f1)
    axes[0].set_ylabel("noise-removal F1")
    axes[0].set_ylim(0, 1.0)
    axes[0].annotate("", xy=(1, f1[1] + 0.06), xytext=(0, f1[0] + 0.06),
                     arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.2))
    axes[0].text(0.5, f1[0] + 0.11, "+1.4% only", ha="center", fontsize=8, color=CRITICAL)
    axes[0].set_title("The network reproduces its supervision", loc="left", fontsize=9)

    b2 = axes[1].bar(lbl, cd, width=0.55, color=[MUTED, CAT[1], ACCENT])
    _barlabels(axes[1], b2, cd)
    axes[1].set_ylabel("Chamfer Distance (m)")
    axes[1].set_ylim(0, max(cd) * 1.25)
    axes[1].set_title("Cost of losing ground-truth labels", loc="left", fontsize=9)
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_self_supervision.pdf")
    plt.close(fig)


# ── 11 & 12. qualitative rasters ────────────────────────────────────────────
def _bev(ax, pts, title, sub, color):
    ax.scatter(pts[:, 0], pts[:, 1], s=0.05, c=color, linewidths=0, rasterized=True)
    ax.set_title(f"{title}\n{sub}", fontsize=8.5, color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-45, 45); ax.set_ylim(-32, 32)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#d8d7d2")


def fig_qualitative():
    import torch

    from lidar_denoise.config import load_config
    from lidar_denoise.denoise import (denoise_cloud, denoise_cloud_range,
                                       load_checkpoint, load_range_checkpoint)
    from lidar_denoise.filters import idsor_filter
    from lidar_denoise.kitti_io import list_scans, load_bin, split_scans
    from lidar_denoise.noise import inject_noise
    from lidar_denoise.preprocessing import preprocess
    from lidar_denoise.train import select_device
    from lidar_denoise import range_image as RI

    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    dev = select_device(cfg["training"]["device"])
    splits = split_scans(list_scans([ROOT / "data" / "raw"]),
                         tuple(cfg["data"]["split"]), cfg["data"]["seed"])
    path = splits["test"][0]
    clean = preprocess(load_bin(path, with_intensity=True), cfg)
    noisy = inject_noise(clean, cfg, seed=7)
    pts = noisy.points

    panels = [("Clean (ground truth)", clean[:, :3], GOOD),
              ("Corrupted (25 mm/h rain)", pts[:, :3], CRITICAL)]
    _, i_idx, _ = idsor_filter(pts, **cfg["filters"]["idsor"])
    panels.append(("IDSOR", pts[i_idx][:, :3], ACCENT))
    ck = ROOT / "results" / "checkpoints"
    mdl, _ = load_checkpoint("dgcnn", ck / "dgcnn.pt", dev)
    d_out, _, _ = denoise_cloud(mdl, "dgcnn", pts[:, :3], cfg, dev)
    panels.append(("DGCNN", d_out, ACCENT))
    rmdl, rcfg = load_range_checkpoint(ck / "range_unet.pt", dev)
    r_out, _, _ = denoise_cloud_range(rmdl, rcfg, pts, cfg, dev)
    panels.append(("RangeRestoreNet", r_out, ACCENT))

    fig, axes = plt.subplots(1, len(panels), figsize=(11.0, 2.5))
    for ax, (t, p, c) in zip(axes, panels):
        _bev(ax, p, t, f"{len(p):,} points", c)
    fig.suptitle(f"Bird's-eye view of frame {path.name} through each denoiser",
                 x=0.008, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig_qualitative.png", dpi=220)
    plt.close(fig)

    # range-image representation
    ri = RI.project(pts, height=rcfg["height"], width=rcfg["width"],
                    max_range=cfg["preprocessing"]["max_range"])
    sl = slice(0, 64), slice(600, 1400)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 2.6))
    axes[0].imshow(ri.image[0][sl], aspect="auto", cmap="Blues", interpolation="nearest")
    axes[0].set_title("Range channel", loc="left", fontsize=8.5)
    axes[1].imshow(ri.image[1][sl], aspect="auto", cmap="Oranges", interpolation="nearest")
    axes[1].set_title("Intensity channel", loc="left", fontsize=8.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The range-image representation (64 laser channels x azimuth, cropped)",
                 x=0.008, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / "fig_range_image.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m, d, a, c = load()
    fig_reconstruction(m); print("  fig_reconstruction.pdf")
    fig_removal(m);        print("  fig_removal_quality.pdf")
    fig_detection(d);      print("  fig_detection.pdf")
    fig_cd_vs_detection(m, d); print("  fig_cd_vs_detection.pdf")
    fig_latency(m);        print("  fig_latency_accuracy.pdf")
    fig_training();        print("  fig_training.pdf")
    fig_severity(a);       print("  fig_ablation_severity.pdf")
    fig_params(a);         print("  fig_ablation_params.pdf")
    fig_fixes(a);          print("  fig_ablation_fixes.pdf")
    fig_selfsup(c);        print("  fig_self_supervision.pdf")
    fig_qualitative();     print("  fig_qualitative.png, fig_range_image.png")
    print(f"[figures] wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
