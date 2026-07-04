# LiDAR Point Cloud Denoising for Autonomous Driving Applications

Capstone Project **CPG No. 5** — Thapar Institute of Engineering and Technology, Patiala.
Mentor: Dr. Lovi Dhamija.

A complete implementation of the proposal's methodology (Section 5, Figure 1):
preprocessing → synthetic noise modelling → **two** denoising approaches
(statistical SOR+ROR baseline and learning-based PointNet-AE / DGCNN) →
reconstruction + downstream-detection evaluation → real-time optimisation.

## Repository layout

```
configs/default.yaml       all paths & hyperparameters (edit here, not in code)
src/lidar_denoise/         the library
  kitti_io.py              KITTI .bin reading/writing, dataset splits
  preprocessing.py         Step 1: NaN/Inf removal, range gate, voxel downsample
  noise.py                 Step 2: Gaussian / outlier / fog-backscatter injection
  filters.py               Step 3a: SOR + ROR statistical baseline (Open3D)
  models/pointnet_ae.py    Step 3b: PointNet autoencoder (Chamfer-trained)
  models/dgcnn.py          Step 3b: DGCNN displacement + outlier-classification
  datasets.py              paired (noisy, clean) patch dataset + cache
  train.py                 training loops with CSV logging
  denoise.py               full-cloud patch-based inference
  metrics.py               Step 4: Chamfer / Hausdorff / HD95 (vectorised)
  detection.py             Step 4: detection-consistency protocol (mmdet3d)
  optimize.py              Step 5: profiling + int8 dynamic quantisation
  visualize.py             BEV before/after figures, metric plots, ablations
scripts/
  run_pipeline.py          driver: --stage cache|train|evaluate|ablate|optimize|all
  detection_eval.py        downstream PointPillars consistency evaluation
  demo.py                  interactive side-by-side before/after demo
tests/test_core.py         unit tests for preprocessing / noise / filters / metrics
results/                   metrics tables (json/md), plots, checkpoints, logs
report/proposal.tex        recreated proposal source (citations fixed) + PDF
report/technical_summary.* results write-up for the final report
prototype.py               the original single-file prototype (kept for reference)
data/                      5 sample KITTI scans (repo) + data/raw/ (downloaded)
```

## Setup

```bash
cd "Point-Cloud denoiser"
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
uv pip install -p .venv/bin/python numpy scipy open3d torch matplotlib \
    pyyaml pytest tqdm pandas        # core pipeline (see requirements.txt for pins)
```

### Data

The pipeline uses **KITTI raw** Velodyne scans (public, no registration):

```bash
cd data/raw
for d in 2011_09_26_drive_0001 2011_09_26_drive_0005; do
  curl -sL -O "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${d}/${d}_sync.zip"
  unzip -q ${d}_sync.zip "*velodyne_points*" -d . && rm ${d}_sync.zip
done
```

This yields 262 scans (~490 MB). **Why this subset:** the official KITTI 3D
object benchmark's Velodyne archive is a single 29 GB login-gated download,
far beyond what denoising experiments need; the raw-split drives are public,
small, and contain identical sensor data. Ground-truth 3D boxes do not exist
for the raw split, which is why detection is evaluated with the
*consistency protocol* described below rather than absolute mAP.

## Running the pipeline

```bash
.venv/bin/python -m pytest tests/            # sanity check
.venv/bin/python scripts/run_pipeline.py --stage cache      # build patch pairs
.venv/bin/python scripts/run_pipeline.py --stage train      # train both models
.venv/bin/python scripts/run_pipeline.py --stage evaluate   # CD/HD tables + plots
.venv/bin/python scripts/run_pipeline.py --stage ablate     # noise & SOR sweeps
.venv/bin/python scripts/run_pipeline.py --stage optimize   # throughput + int8
.venv/bin/python scripts/demo.py                            # interactive demo
```

Outputs land in `results/`:
`metrics_summary.json` / `metrics_table.md` (Step 4 tables),
`throughput.json` (Step 5), `ablations.json`, `plots/*.png`
(before/after BEV panels, metric bar charts, ablation curves, training
curves), `checkpoints/*.pt`, `logs/*_train.csv`.

### Training scale note (GPU vs this repo's checkpoints)

The committed checkpoints were trained on an Apple M1 (MPS backend), the
largest run feasible on team hardware: **PointNet-AE 30 epochs (~10 min),
DGCNN 10 epochs (~45 min)** over ~39k patches from 213 scans. The code is
device-agnostic — on a CUDA machine set `training.device: cuda` (or leave
`auto`) and raise `training.epochs` to 30–50 for the DGCNN; per the proposal,
8 GB VRAM is ample (models are 0.2M–0.8M parameters).

## Detection evaluation (optional stage)

Uses a **KITTI-pretrained PointPillars** from MMDetection3D, CPU inference.

```bash
uv pip install -p .venv/bin/python mmengine && \
uv pip install -p .venv/bin/python --no-build-isolation mmcv && \
uv pip install -p .venv/bin/python --no-deps mmdet mmdet3d && \
uv pip install -p .venv/bin/python numba scikit-image shapely pycocotools \
    trimesh plyfile terminaltables
# mmdet/mmdet3d pin mmcv<2.2 but work with 2.2.0 — relax the assert:
sed -i '' "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    .venv/lib/python3.12/site-packages/mmdet/__init__.py \
    .venv/lib/python3.12/site-packages/mmdet3d/__init__.py
# checkpoint (19 MB):
curl -sL -o models/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
  "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"

.venv/bin/python scripts/detection_eval.py
```

**Protocol.** KITTI raw frames have no box labels, so we measure *detection
consistency*: the detector's boxes on the clean cloud are the reference; we
report what fraction survive (recall) and how many spurious boxes appear
(precision) on the noisy vs denoised clouds. This directly quantifies how
much detector performance the denoiser recovers, without fabricating ground
truth. Results: `results/detection_consistency.json`.

## Headline results (10 held-out KITTI frames, Apple M1)

| Method | CD (m) ↓ | HD95 (m) ↓ | CD impr. | Det. recall ↑ | Det. precision ↑ | Latency |
|---|---|---|---|---|---|---|
| Noisy input | 0.403 | — | — | 0.760 | 0.853 | — |
| **SOR+ROR** | **0.095** | **0.136** | **75.5%** | 0.733 | **0.975** | **104 ms** |
| PointNet-AE | 0.374 | 1.835 | 5.4% | 0.067 | 0.750 | 282 ms (MPS) |
| **DGCNN** | 0.162 | 0.156 | 59.1% | **0.788** | 0.870 | 3.0 s (MPS) |

Full discussion (ablations, quantisation, honest 10 Hz assessment) in
`report/technical_summary.pdf`. Key finding: Chamfer Distance alone does not
predict downstream detection quality — the AE is near-noisy on CD yet
catastrophic for the detector, while DGCNN recovers the most detections.

## Deferred: hardware component

The proposal's YDLIDAR X2 / Raspberry Pi live-capture component is
**out of scope for this software deliverable** — it requires the physical
sensor. The pipeline is sensor-agnostic ((N,3) arrays in), so a future
`ydlidar` reader can plug into `kitti_io.load_bin`'s place unchanged.

## Reproducibility

All randomness is seeded (`data.seed`, per-frame noise seeds). Training
writes CSV logs to `results/logs/`. `configs/default.yaml` is the single
source of truth for paths and hyperparameters.
