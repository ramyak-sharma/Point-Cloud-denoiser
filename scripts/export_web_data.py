#!/usr/bin/env python
"""Export point clouds and results for the showcase web app.

Two kinds of payload, and the split is deliberate:

* **The corrupted cloud, shipped raw.** The statistical filters (SOR, ROR,
  IDSOR) are simple geometric algorithms, so the site reimplements them in
  JavaScript and runs them live as the visitor drags a slider. That needs the
  input cloud and its ground-truth noise labels, nothing more.
* **The learned models' outputs, precomputed.** PyTorch cannot run in a
  browser, so DGCNN / PointNet-AE / RangeRestoreNet results are exported as
  finished clouds. The site says so rather than pretending otherwise.

Positions are quantised to Int16 centimetres. At a 60 m range gate that is
lossless to within 5 mm — far below the 0.15 m voxel size the whole pipeline
already operates at — and halves the transfer against Float32.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "site" / "data"

MAX_POINTS = 16000       # renders smoothly on integrated GPUs
SCALE = 100.0            # metres -> centimetres for Int16


def quantise(points: np.ndarray) -> np.ndarray:
    """(N,3) float metres -> (N,3) int16 centimetres."""
    q = np.round(points[:, :3] * SCALE)
    return np.clip(q, -32768, 32767).astype(np.int16)


def decimate(n: int, keep: int, seed: int = 0) -> np.ndarray:
    """Uniform random subsample indices, or all of them if already small."""
    if n <= keep:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, keep, replace=False))


def write_cloud(name: str, points: np.ndarray, extras: dict | None = None) -> dict:
    """Write positions (+ optional per-point uint8 channels) as raw binaries."""
    OUT.mkdir(parents=True, exist_ok=True)
    quantise(points).tofile(OUT / f"{name}.xyz.i16")
    entry = {"n": int(len(points)), "scale": SCALE}
    for key, arr in (extras or {}).items():
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        arr.tofile(OUT / f"{name}.{key}.u8")
        entry[key] = True
    return entry


def main() -> None:
    from lidar_denoise.config import load_config
    from lidar_denoise.denoise import (denoise_cloud, denoise_cloud_range,
                                       load_checkpoint, load_range_checkpoint)
    from lidar_denoise.filters import sor_ror_pipeline, idsor_filter
    from lidar_denoise.kitti_io import list_scans, load_bin, split_scans
    from lidar_denoise.metrics import chamfer_distance, removal_scores
    from lidar_denoise.noise import apply_severity, inject_noise
    from lidar_denoise.preprocessing import preprocess
    from lidar_denoise.train import select_device

    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    device = select_device(cfg["training"]["device"])
    splits = split_scans(list_scans([ROOT / "data" / "raw"]),
                         tuple(cfg["data"]["split"]), cfg["data"]["seed"])
    frame = splits["test"][0]
    ck = ROOT / "results" / "checkpoints"

    dg, _ = load_checkpoint("dgcnn", ck / "dgcnn.pt", device)
    ae, _ = load_checkpoint("pointnet_ae", ck / "pointnet_ae.pt", device)
    rn, rcfg = load_range_checkpoint(ck / "range_unet.pt", device)

    manifest: dict = {"frame": frame.name, "severities": {}, "max_points": MAX_POINTS}

    for sev in ("light", "moderate", "heavy"):
        c = apply_severity(cfg, sev)
        clean = preprocess(load_bin(frame, with_intensity=True), c)
        noisy = inject_noise(clean, c, seed=7)
        pts, labels = noisy.points, noisy.labels

        keep = decimate(len(pts), MAX_POINTS)
        keep_set = np.zeros(len(pts), bool)
        keep_set[keep] = True
        sub_pts, sub_lab = pts[keep], labels[keep]
        # map original index -> position in the decimated array, -1 if dropped
        remap = np.full(len(pts), -1, np.int32)
        remap[keep] = np.arange(len(keep))

        entry: dict = {"clouds": {}, "metrics": {}}
        ck_clean = decimate(len(clean), MAX_POINTS, seed=1)
        entry["clouds"]["clean"] = write_cloud(
            f"{sev}_clean", clean[ck_clean],
            {"intensity": clean[ck_clean, 3]})
        entry["clouds"]["noisy"] = write_cloud(
            f"{sev}_noisy", sub_pts,
            {"intensity": sub_pts[:, 3], "label": sub_lab})

        # ── methods that only REMOVE: ship the surviving index set, so the
        #    site can render them without a second copy of the geometry
        removal = {}
        r = sor_ror_pipeline(pts[:, :3], c)
        removal["SOR+ROR"] = r["kept_indices"]
        removal["IDSOR"] = idsor_filter(pts, **c["filters"]["idsor"])[1]

        for name, kept in removal.items():
            m = remap[kept]
            m = m[m >= 0].astype(np.int32)
            m.tofile(OUT / f"{sev}_{name.replace('+', '')}.keep.i32")
            s = removal_scores(kept, labels)
            entry["clouds"][name] = {"keep_indices": True, "n": int(len(m))}
            entry["metrics"][name] = {
                "cd": float(chamfer_distance(clean[:, :3], pts[kept][:, :3])),
                "retention_pct": 100.0 * len(kept) / len(pts),
                **{k: float(v) for k, v in s.items()},
            }

        # ── learned models: precomputed clouds (cannot run in a browser)
        for name, fn in (
            ("DGCNN", lambda: denoise_cloud(dg, "dgcnn", pts[:, :3], c, device)),
            ("PointNet-AE", lambda: denoise_cloud(ae, "pointnet_ae", pts[:, :3], c, device)),
            ("RangeRestoreNet", lambda: denoise_cloud_range(rn, rcfg, pts, c, device)),
        ):
            out, ms, kept = fn()
            sel = decimate(len(out), MAX_POINTS, seed=2)
            entry["clouds"][name] = write_cloud(f"{sev}_{name.replace('-', '')}", out[sel])
            s = removal_scores(kept, labels) if kept is not None else {}
            entry["metrics"][name] = {
                "cd": float(chamfer_distance(clean[:, :3], out)),
                "retention_pct": 100.0 * len(out) / len(pts),
                "time_ms": float(ms),
                **{k: float(v) for k, v in s.items()},
            }

        entry["metrics"]["noisy"] = {
            "cd": float(chamfer_distance(clean[:, :3], pts[:, :3])),
            "retention_pct": 100.0,
        }
        entry["metrics"]["oracle"] = {
            "cd": float(chamfer_distance(clean[:, :3], pts[labels == 0][:, :3])),
            "retention_pct": 100.0 * (labels == 0).mean(),
            "removal_precision": 1.0, "removal_recall": 1.0, "removal_f1": 1.0,
        }
        entry["n_full"] = int(len(pts))
        entry["rain_rate"] = c["noise"]["rain_rate"]
        manifest["severities"][sev] = entry
        print(f"[export] {sev}: {len(pts)} pts -> {len(sub_pts)} shipped")

    # results the dashboard charts read directly
    res = ROOT / "results"
    for f in ("metrics_summary.json", "detection_consistency.json",
              "ablations.json", "supervision_comparison.json", "throughput.json"):
        src = json.load(open(res / f))
        if f == "metrics_summary.json":
            src = {"summary": src["summary"], "significance_cd": src["significance_cd"],
                   "protocol": src["protocol"]}
        elif f == "detection_consistency.json":
            src = {"summary": src["summary"],
                   "significance_vs_oracle": src["significance_vs_oracle"],
                   "significance_vs_noisy": src["significance_vs_noisy"],
                   "protocol": src["protocol"]}
        json.dump(src, open(OUT / f, "w"), separators=(",", ":"))

    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"[export] {len(list(OUT.iterdir()))} files, {total/1e6:.2f} MB -> {OUT}")


if __name__ == "__main__":
    main()
