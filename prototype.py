"""
LiDAR Point Cloud Denoising Prototype
======================================
Capstone Project - CPG No. 5
Thapar Institute of Engineering and Technology, Patiala

Pipeline (mirrors Figure 1 from proposal):
  Step 1: Load & Preprocess
  Step 2: Inject Synthetic Noise
  Step 3: Denoise (SOR + ROR baseline)
  Step 4: Evaluate (Chamfer Distance, point retention)
  Step 5: Export results for visualisation dashboard
"""

import numpy as np
import open3d as o3d
import json, time, os

# ─────────────────────────────────────────────
# STEP 0 — reproducibility
# ─────────────────────────────────────────────
np.random.seed(42)
RESULTS_PATH = "results.json"


# ─────────────────────────────────────────────
# STEP 1 — Load & preprocess point cloud
# ─────────────────────────────────────────────
def load_point_cloud(bin_path="data/0000000001.bin"):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts = pts[:, :3]  # drop intensity column, keep XYZ only
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # normalise same as before
    p = np.asarray(pcd.points)
    p -= p.mean(axis=0)
    p /= np.max(np.linalg.norm(p, axis=1))
    pcd.points = o3d.utility.Vector3dVector(p)
    return pcd


# ─────────────────────────────────────────────
# STEP 2 — Synthetic noise injection
# ─────────────────────────────────────────────
def inject_noise(pcd, gaussian_sigma=0.015, outlier_ratio=0.08, fog_ratio=0.04):
    """
    Three noise types from proposal's Need Analysis section:
      • Gaussian jitter   — sensor vibration / calibration error
      • Random outliers   — multi-echo interference
      • Fog/rain points   — backscatter from suspended particles
    Returns noisy pcd and a breakdown dict.
    """
    print("[Step 2] Injecting synthetic noise...")
    pts = np.asarray(pcd.points).copy()
    n = len(pts)

    # — Gaussian jitter on every point
    gaussian_noise = np.random.normal(0, gaussian_sigma, pts.shape)
    pts += gaussian_noise

    # — Random outliers scattered in the scene bounding box
    n_outliers = int(n * outlier_ratio)
    outliers = np.random.uniform(-1.5, 1.5, (n_outliers, 3))
    pts = np.vstack([pts, outliers])

    # — Fog simulation: spurious near-range points clustered around origin
    n_fog = int(n * fog_ratio)
    fog_pts = np.random.normal(0, 0.3, (n_fog, 3))
    pts = np.vstack([pts, fog_pts])

    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(pts)

    breakdown = {
        "original": n,
        "gaussian_perturbed": n,
        "outliers_added": n_outliers,
        "fog_points_added": n_fog,
        "total_noisy": len(pts)
    }
    print(f"    Noisy cloud: {len(pts):,} points "
          f"({n_outliers} outliers + {n_fog} fog pts + Gaussian jitter)")
    return noisy_pcd, breakdown


# ─────────────────────────────────────────────
# STEP 3 — Denoising (SOR + ROR baseline)
# ─────────────────────────────────────────────
def denoise_sor(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical Outlier Removal — fast O(n log n) baseline."""
    print("[Step 3a] Applying Statistical Outlier Removal (SOR)...")
    t0 = time.time()
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    elapsed = (time.time() - t0) * 1000
    print(f"    SOR kept {len(ind):,} / {len(pcd.points):,} points  [{elapsed:.1f} ms]")
    return cl, elapsed


def denoise_ror(pcd, nb_points=6, radius=0.05):
    """Radius Outlier Removal — complementary pass."""
    print("[Step 3b] Applying Radius Outlier Removal (ROR)...")
    t0 = time.time()
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    elapsed = (time.time() - t0) * 1000
    print(f"    ROR kept {len(ind):,} / {len(pcd.points):,} points  [{elapsed:.1f} ms]")
    return cl, elapsed


# ─────────────────────────────────────────────
# STEP 4 — Evaluation metrics
# ─────────────────────────────────────────────
def chamfer_distance(pcd_a, pcd_b):
    """
    Chamfer Distance: mean of nearest-neighbour distances in both directions.
    Lower is better — measures geometric reconstruction fidelity.
    """
    pts_a = np.asarray(pcd_a.points)
    pts_b = np.asarray(pcd_b.points)

    tree_b = o3d.geometry.KDTreeFlann(pcd_b)
    tree_a = o3d.geometry.KDTreeFlann(pcd_a)

    d_ab = []
    for p in pts_a:
        _, idx, _ = tree_b.search_knn_vector_3d(p, 1)
        d_ab.append(np.linalg.norm(p - pts_b[idx[0]]))

    d_ba = []
    for p in pts_b:
        _, idx, _ = tree_a.search_knn_vector_3d(p, 1)
        d_ba.append(np.linalg.norm(p - pts_a[idx[0]]))

    return float(np.mean(d_ab) + np.mean(d_ba))


def evaluate(clean_pcd, noisy_pcd, denoised_pcd, label="SOR"):
    """Compute CD before and after denoising."""
    print(f"[Step 4] Evaluating {label}...")
    cd_before = chamfer_distance(clean_pcd, noisy_pcd)
    cd_after  = chamfer_distance(clean_pcd, denoised_pcd)
    improvement = (cd_before - cd_after) / cd_before * 100
    retention   = len(denoised_pcd.points) / len(noisy_pcd.points) * 100

    print(f"    CD before denoising : {cd_before:.6f}")
    print(f"    CD after  denoising : {cd_after:.6f}")
    print(f"    CD improvement      : {improvement:.1f}%")
    print(f"    Point retention     : {retention:.1f}%")
    return cd_before, cd_after, improvement, retention


# ─────────────────────────────────────────────
# STEP 5 — Save results for dashboard
# ─────────────────────────────────────────────
def save_results(results: dict):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Step 5] Results saved to {RESULTS_PATH}")


def colorize(pcd, color):
    """Paint all points a single RGB colour for visualisation."""
    pcd.paint_uniform_color(color)
    return pcd


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  LiDAR Point Cloud Denoising Prototype")
    print("  CPG No. 5 — Thapar Institute, Patiala")
    print("=" * 55)

    # Step 1
    clean_pcd = load_point_cloud()

    # Step 2
    noisy_pcd, noise_breakdown = inject_noise(
        clean_pcd,
        gaussian_sigma=0.015,
        outlier_ratio=0.08,
        fog_ratio=0.04
    )

    # Step 3 — SOR then ROR in sequence
    sor_pcd, sor_time = denoise_sor(noisy_pcd, nb_neighbors=20, std_ratio=2.0)
    final_pcd, ror_time = denoise_ror(sor_pcd, nb_points=6, radius=0.05)
    total_time = sor_time + ror_time

    # Step 4
    cd_before, cd_sor, imp_sor, ret_sor = evaluate(clean_pcd, noisy_pcd, sor_pcd, "SOR")
    _,         cd_final, imp_final, ret_final = evaluate(clean_pcd, noisy_pcd, final_pcd, "SOR+ROR")

    # Step 5
    results = {
        "noise_breakdown": noise_breakdown,
        "metrics": {
            "cd_noisy_vs_clean":     round(cd_before, 6),
            "cd_sor_vs_clean":       round(cd_sor, 6),
            "cd_sor_ror_vs_clean":   round(cd_final, 6),
            "improvement_sor_pct":   round(imp_sor, 2),
            "improvement_final_pct": round(imp_final, 2),
            "retention_sor_pct":     round(ret_sor, 2),
            "retention_final_pct":   round(ret_final, 2),
            "sor_time_ms":           round(sor_time, 2),
            "ror_time_ms":           round(ror_time, 2),
            "total_time_ms":         round(total_time, 2),
        },
        "point_counts": {
            "clean":    len(clean_pcd.points),
            "noisy":    len(noisy_pcd.points),
            "after_sor": len(sor_pcd.points),
            "after_sor_ror": len(final_pcd.points),
        }
    }
    save_results(results)

    import pyvista as pv

    print("\n[Visualise] Opening 3 windows simultaneously...")

    clean_pts  = np.asarray(clean_pcd.points)
    noisy_pts  = np.asarray(noisy_pcd.points)
    final_pts  = np.asarray(final_pcd.points)

    def make_cloud(pts, color):
        cloud = pv.PolyData(pts)
        p = pv.Plotter()
        p.background_color = "#111111"
        p.add_points(cloud, color=color, point_size=3, render_points_as_spheres=True)
        return p

    p1 = make_cloud(clean_pts,  "#44dd77")
    p2 = make_cloud(noisy_pts,  "#ff4444")
    p3 = make_cloud(final_pts,  "#4499ff")

    p1.show(title=f"CLEAN — {len(clean_pts):,} pts",     auto_close=False)
    p2.show(title=f"NOISY — {len(noisy_pts):,} pts",     auto_close=False)
    p3.show(title=f"DENOISED — {len(final_pts):,} pts",  auto_close=False)


if __name__ == "__main__":
    main()