"""Publication-style figures: before/after clouds, metric comparisons, ablations.

Matplotlib only (headless-safe); interactive 3D viewing is provided
separately by scripts/demo.py using PyVista/Open3D windows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _bev_axes(ax, points: np.ndarray, title: str, color_by_height: bool = True, lim: float = 40):
    """Bird's-eye-view scatter (x forward, y left), coloured by height."""
    c = points[:, 2] if color_by_height else "steelblue"
    ax.scatter(points[:, 0], points[:, 1], s=0.3, c=c, cmap="viridis", linewidths=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{len(points):,} pts", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def before_after_figure(
    clouds: dict[str, np.ndarray], out_path: Path, suptitle: str = ""
) -> None:
    """Side-by-side BEV panels, one per named cloud (clean/noisy/denoised...)."""
    n = len(clouds)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (name, pts) in zip(axes, clouds.items()):
        _bev_axes(ax, pts, name)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))   # keep panel titles clear of it
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def metric_bars(
    table: dict[str, dict[str, float]], metric: str, ylabel: str, out_path: Path, title: str
) -> None:
    """Bar chart of one metric across methods. table: method -> {metric: value}."""
    methods = list(table.keys())
    values = [table[m][metric] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(methods, values, color=["#888", "#4477aa", "#66aa55", "#cc6644"][: len(methods)])
    ax.bar_label(bars, fmt="%.4f", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def ablation_lines(
    x: list[float], series: dict[str, list[float]], xlabel: str, ylabel: str,
    out_path: Path, title: str,
) -> None:
    """Line plot for ablations (e.g. CD vs noise sigma per method)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, ys in series.items():
        ax.plot(x, ys, marker="o", markersize=4, label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def training_curve(csv_path: Path, out_path: Path, title: str) -> None:
    """Loss-vs-epoch curve from a training CSV log."""
    import csv as _csv

    epochs, losses = [], []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            epochs.append(int(row["epoch"])); losses.append(float(row["loss"]))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(epochs, losses)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
