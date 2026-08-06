# LiDAR Point Cloud Denoising for Autonomous Driving Applications

Capstone Project **CPG No. 5** — Thapar Institute of Engineering and Technology, Patiala.
Mentor: Dr. Lovi Dhamija.

A complete implementation of the proposal's methodology (Section 5, Figure 1):
preprocessing → physics-based weather modelling → denoising by **six**
methods (SOR+ROR, IDSOR, temporal-consistency filtering, PointNet-AE, DGCNN,
and RangeRestoreNet) → reconstruction + downstream-detection evaluation →
real-time optimisation.

Three of the six were added after a survey of the 2024–2026 literature:
**IDSOR** (an intensity- and range-adaptive statistical filter),
**RangeRestoreNet** (a dual-head range-image network developed here, the
only method that meets the 10 Hz target while learning), and a
**self-supervised** variant trained without any ground-truth labels.

## Repository layout

```
configs/default.yaml       all paths & hyperparameters (edit here, not in code)
src/lidar_denoise/         the library
  kitti_io.py              KITTI .bin reading/writing, dataset splits
  preprocessing.py         Step 1: NaN/Inf removal, range gate, voxel downsample
  noise.py                 Step 2: LISA physics-based rain/fog + Gaussian jitter
                           + outliers; severity presets (light/moderate/heavy)
  filters.py               Step 3a: SOR + ROR baseline, IDSOR (intensity- and
                           range-adaptive), temporal filter with ICP ego-motion
  range_image.py           spherical projection + exact inverse, ring recovery
  pseudo_labels.py         physics-derived noise labels (no ground truth)
  models/pointnet_ae.py    Step 3b: PointNet autoencoder (Chamfer-trained)
  models/dgcnn.py          Step 3b: DGCNN displacement + outlier-classification
  models/range_unet.py     Step 3c: dual-head range-image net (this project's)
  datasets.py              paired (noisy, clean) patch dataset + cache
  train.py                 training loops, per-epoch validation, CSV logging
  train_range.py           range-image training (on-the-fly augmentation)
  denoise.py               full-cloud patch-based inference (coverage-complete)
  metrics.py               Step 4: Chamfer / Hausdorff / HD95, oracle floor,
                           noise-removal precision/recall/F1 (vectorised)
  detection.py             Step 4: detection-consistency protocol (mmdet3d)
  optimize.py              Step 5: profiling + int8 dynamic quantisation
  visualize.py             BEV before/after figures, metric plots, ablations
scripts/
  run_pipeline.py          driver: --stage cache|train|evaluate|ablate|optimize|all
  train_range.py           trains the dual-head range-image denoiser
                           (--self-supervised for the no-labels variant)
  compare_supervision.py   supervised vs self-supervised vs rules-only
  detection_eval.py        downstream PointPillars consistency evaluation
  demo.py                  interactive side-by-side before/after demo
tests/test_core.py         35 unit tests: preprocessing, weather physics,
                           filters, ego-motion registration, range projection,
                           pseudo-labels, metrics, degenerate-input guards
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
.venv/bin/python scripts/run_pipeline.py --stage cache      # train + val patch pairs
.venv/bin/python scripts/run_pipeline.py --stage train      # AE + DGCNN
.venv/bin/python scripts/train_range.py                     # RangeRestoreNet
.venv/bin/python scripts/run_pipeline.py --stage evaluate   # CD/HD95 tables + plots
.venv/bin/python scripts/run_pipeline.py --stage ablate     # 5 ablation sweeps
.venv/bin/python scripts/run_pipeline.py --stage optimize   # throughput + int8
.venv/bin/python scripts/demo.py                            # interactive demo
```

Outputs land in `results/`:
`metrics_summary.json` / `metrics_table.md` (Step 4 tables, with per-method
dispersion and pairwise significance tests), `throughput.json` (Step 5),
`ablations.json`, `plots/*.png` (before/after BEV panels, metric bar charts,
ablation curves, train/validation curves), `checkpoints/*.pt`,
`logs/*_train.csv`.

> **Note on stdout.** The stage scripts print progress but Python
> block-buffers stdout when it is redirected to a file, so a long
> `--stage train > log` run can look frozen for many minutes. The CSV logs
> in `results/logs/` are flushed per epoch — `tail` those to watch real
> progress, or run Python with `-u`.

### Ablations produced by `--stage ablate`

| Sweep | Question it answers |
|---|---|
| corruption severity | do the methods degrade gracefully from light to heavy rain? |
| SOR `k` / `std_ratio` | how sensitive is the baseline to its two parameters? |
| AE recombination | how much of the AE's error is the model vs the patch-recombination step? |
| temporal ego-motion | does ICP compensation actually matter, or is it a refinement? |
| DGCNN threshold | where does the outlier head's precision/recall dial belong? |

### Training scale note (GPU vs this repo's checkpoints)

The committed checkpoints were trained on an Apple M1 (MPS backend), the
largest run feasible on team hardware: **10 epochs each** over 28,510
patches from 213 training scans, validated against 3,415 patches from 26
held-out scans. The AE trains at ~22 s/epoch; the DGCNN at ~5.5 min/epoch
(its EdgeConv graph build dominates). The code is device-agnostic — on a
CUDA machine set
`training.device: cuda` (or leave `auto`) and raise `training.epochs` to
30–50 for the DGCNN; per the proposal, 8 GB VRAM is ample (models are
0.2M–0.8M parameters).

Training keeps the **best-on-validation** checkpoint rather than the last
epoch, which is not cosmetic: in the reference run the AE's validation loss
was lowest at epoch 8 (0.0338) and rose again by epoch 10 (0.0359), so
last-epoch selection would have shipped a measurably worse model.

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
truth. An `oracle (perfect removal)` condition is evaluated alongside the
real methods — every injected point deleted, nothing else touched — which
answers what perfect denoising would actually be worth to the detector, and
bounds what any method here could achieve. Results:
`results/detection_consistency.json`.

## Evaluation protocol

Four choices make the numbers in `results/` comparable across methods, and
each is worth stating explicitly when the table is presented:

**Paired, multi-seed.** Every method is run on the *identical* noisy cloud
for a given (frame, seed), over `evaluation.num_seeds` independent noise
draws × `evaluation.num_test_frames` frames. Results carry a standard
deviation, and every pairwise ranking claim is backed by a Wilcoxon
signed-rank test recorded under `significance_cd` in
`results/metrics_summary.json`. A method scoring lower on ten frames is not
by itself evidence that it is better.

**An oracle floor, not zero.** The LISA weather model *deletes* returns
whose attenuated power drops below the detection threshold; no denoiser can
put them back. The `oracle (perfect removal)` row is what a perfect noise
classifier would score — every injected point removed, every real point
kept — and it is the ceiling every other row should be read against.

**Removal F1 alongside geometry.** Chamfer Distance cannot distinguish
"removed the noise" from "moved everything slightly". Every method that
yields an index set into the noisy cloud — all three filters, the DGCNN via
its outlier head, and RangeRestoreNet via its removal head — also reports
noise-removal precision/recall/F1 against the ground-truth injection labels.

**HD95 over strict HD.** Strict Hausdorff is the max over all residual
points, so a single survivor near the 60 m range gate pins it around 30 m
for every method — the column ranked nothing and is kept only in the JSON.

## Headline results

Moderate severity (25 mm/h), Apple M1. Geometry: 10 frames × 3 seeds (30
paired observations), every pairwise CD difference significant at p < 10⁻⁵.
Detection: **28 frames × 3 seeds (84 paired observations)**. All tests are
paired Wilcoxon signed-rank, recorded in `results/metrics_summary.json` and
`results/detection_consistency.json`.

| Method | CD (m) ↓ | HD95 ↓ | Oracle gap ↓ | Removal F1 ↑ | Retention | Det. recall ↑ | Latency |
|---|---|---|---|---|---|---|---|
| Noisy input | 0.572 ± 0.111 | — | 100% | — | — | 0.259 ± 0.231 | — |
| *oracle (perfect removal)* | *0.104 ± 0.003* | *0.353* | *0%* | *1.000* | *68.9%* | *0.479 ± 0.325* | — |
| **DGCNN** | **0.111 ± 0.008** | **0.342** | **1.6%** | 0.788 | 78.7% | **0.452 ± 0.298** | 1,806 ms |
| **IDSOR** | **0.174 ± 0.013** | 0.827 | 15.0% | 0.665 | 81.3% | 0.368 ± 0.291 | **49 ms (20 Hz)** |
| **RangeRestoreNet** | 0.215 ± 0.035 | 0.459 | 23.8% | **0.814** | 65.6% | 0.403 ± 0.320 | **45 ms (22 Hz)** |
| SOR+ROR | 0.251 ± 0.020 | 1.529 | 31.5% | 0.628 | 78.7% | 0.392 ± 0.303 | 65 ms |
| Temporal+SOR | 0.317 ± 0.035 | 2.173 | 45.5% | 0.567 | 81.4% | 0.389 ± 0.269 | 278 ms |
| PointNet-AE | 0.521 ± 0.092 | 3.353 | 89.1% | 0.000 | 100.0% | **0.062 ± 0.104** | 1,192 ms |

*Oracle gap* is the fraction of the **recoverable** error a method leaves
behind: `(CD_method − CD_oracle) / (CD_noisy − CD_oracle)`. 0% means it
matched a perfect noise classifier. It exists because raw CD is
uninterpretable when 31% of returns are destroyed beyond recovery.

**Five findings.**

**1. Perfect removal IS the ceiling, and the DGCNN nearly reaches it.**
The DGCNN closes 98.4% of the recoverable geometric error (oracle gap
1.6%). On downstream detection it is the only method statistically
indistinguishable from the oracle (0.452 vs 0.479, p = 0.15); **every other
method is significantly worse** (p ≤ 0.0002).

This corrects an earlier revision of this README, which claimed the
opposite — that SOR+ROR and the DGCNN *beat* a perfect remover, and that
position correction was why. That claim came from 10 frames without a
significance test. At 84 paired observations the ranking reverses: the
oracle is best, and nothing exceeds it. The lesson is in
*Methodology corrections*.

**2. Chamfer Distance fails to predict detection quality.** The PointNet-AE
improves CD by 8.9% over the noisy input while collapsing detection recall
from 0.259 to **0.062** — a 76% loss, **p < 0.0001** over 84 paired
observations. It is the only method that makes detection *worse* than doing
nothing. This is not a point-count artefact: after the recombination fix the
AE emits exactly 100% of the input cardinality. It is a displacement-only
model that smooths scan-line structure into a surface the detector cannot
parse, and CD rewards it for doing so.

A second, milder inversion points the same way: **IDSOR has clearly better
CD than SOR+ROR (0.174 vs 0.251) yet slightly lower detection recall** (0.368
vs 0.392, p = 0.069). Across all six methods the rank correlation between CD
and detection recall is only ρ = −0.60 (p = 0.21) — directionally right,
statistically weak. Reconstruction error is a rough guide at best.

**3. Chamfer Distance can be gamed by emitting more points.** The
recombination ablation is the sharpest evidence: the legacy `union` mode
scores a *better* CD (0.525) than the corrected `nn_project` mode (0.569)
while returning **232% of the input points**. Densifying a cloud shortens
the clean→denoised nearest-neighbour term faster than it lengthens the
reverse term. Any CD comparison that does not also report retention is
therefore unsafe — which is why the table above carries that column.

**4. Position restoration does not work on this corruption — a negative
result.** Both learned models carry a displacement head, and neither does
anything useful:

| | mean position error of surviving real points |
|---|---|
| corrupted input | 0.0804 m |
| after DGCNN displacement | 0.0783 m (−2.6%) |
| after RangeRestoreNet restoration | 0.0804 m (no change) |

Removing the DGCNN's displacement entirely leaves CD unchanged (0.1158 vs
0.1157). This is not undertraining. Weather range jitter is **zero-mean**,
so given a single corrupted observation the error-minimising prediction *is*
zero — the restoration head converges there correctly. Undoing it needs a
surface prior over neighbouring returns, which neither a per-point patch MSE
nor a per-pixel image L1 induces. Anyone extending this work should treat
position restoration as an open problem requiring an explicit smoothness or
surface-consistency term, not as a head you can bolt on.

**5. The 10 Hz target is met — including, for the first time, by a learned
model.** `RangeRestoreNet` runs end-to-end in **45 ms (22 Hz)** against the
DGCNN's 1,806 ms: a **40× speedup** from replacing per-patch k-NN graph
construction with 2-D convolution over a range image, which is the change
the recent literature converged on. It also has the **best noise-removal F1
of any method (0.814)** and recovers detection recall to 0.403 — better than
SOR+ROR and IDSOR, at a third of SOR+ROR's cost.

Its weakness is over-deletion: retention 65.6%, *below* the oracle's 68.9%,
because it is recall-heavy (it finds noise well but removes real points with
it), which is why its CD trails the DGCNN's. The honest summary is a
three-way trade: DGCNN for accuracy, RangeRestoreNet for speed, IDSOR for
the best accuracy-per-millisecond among methods needing no training.

Supporting detail in `results/metrics_table.md`,
`results/detection_consistency.json`, `results/ablations.json`, and
`report/technical_summary.pdf`.

## Self-supervision: training without ground-truth labels

The sharpest objection to everything above is that the supervision is
circular. We inject the noise, so of course we can find it — every learned
model here trains on `inject_noise`'s own label array, which no real
deployment has. And there is no way around it by finding better data: you
cannot record the same scene simultaneously with and without rain, so a
paired real dataset does not exist.

Following [LIORNet](https://arxiv.org/html/2603.19936v1), `pseudo_labels.py`
derives noise labels from the *physics* of a weather return instead, using
only a single corrupted scan:

1. **Range-conditioned intensity anomaly** — received power falls as ρ/r², so
   a surface at 40 m legitimately returns less than one at 10 m. A global
   intensity threshold would just relabel "far" as "noise", so the median
   intensity is estimated *per range bin* from the scan itself and a return
   is anomalous only relative to its own range.
2. **Neighbourhood range coherence** — a real surface is continuous in the
   range image; a droplet floats in front of what it occluded, so its
   neighbours disagree with it.
3. **Edge protection** — genuine object silhouettes also create range
   discontinuities, but coherent ones that persist across adjacent pixels.
   Isolated discontinuities are not protected; extended ones are.

LIORNet also uses zero-reflectivity, which Ouster sensors report and the
KITTI HDL-64E does not, so that rule is omitted rather than faked.

**The supervision this yields is poor and we report it as such:** precision
0.500, recall 0.470, **F1 0.484** against the true labels. Roughly half of
what the self-supervised model is told is wrong. Both rules contribute
independently (isolation alone F1 0.423, intensity alone 0.388).

The restoration head is switched off entirely in this configuration — its
target is the clean range of a known source point, which is ground truth by
construction. Keeping it would make the "no labels needed" claim false.

### Result: the network does not beat its own supervision

20 test frames × 2 seeds:

| Variant | CD ↓ | HD95 ↓ | Removal F1 ↑ | Retention | Latency |
|---|---|---|---|---|---|
| Pseudo-label rules alone (no network) | 0.465 | 3.297 | 0.484 | 70.5% | — |
| Self-supervised network | 0.436 | 2.564 | **0.491** | 67.7% | 40 ms |
| **Supervised network** | **0.227** | **0.460** | **0.813** | 65.3% | 45 ms |

Self-supervision reaches 60% of the supervised model's removal F1 — but
**that 60% comes from the hand-written rules, not from learning.** The
network moved F1 only from 0.484 to 0.491 (+1.4% relative): it learned to
imitate its supervision rather than to see past it. Quoting "retains 60% of
supervised performance" without this comparison would credit the network for
what three lines of physics already provide.

The one place learning helps is the tail: **HD95 improves 3.297 → 2.564
(22%)** and CD 0.465 → 0.436. The network is better than the rules at
catching the *most damaging* surviving outliers, even though its average
classification accuracy is no better.

**Read honestly, this is a negative result for self-supervision as
implemented**, and it quantifies precisely what the synthetic labels are
worth: the gap between F1 0.49 and 0.81 is the value of knowing the ground
truth, and no amount of training on these rules closes it. Better pseudo-labels
— multi-echo returns, temporal consistency across sweeps, or the
zero-reflectivity channel that Ouster sensors provide and KITTI does not —
are the prerequisite, not more epochs.

Run `scripts/train_range.py --self-supervised`, then
`scripts/compare_supervision.py`; results land in
`results/supervision_comparison.json`.

## Methodology corrections

Seven defects were found and fixed in review; each affected reported
numbers, so they are listed here rather than buried in the git log. Five
were confirmed by measurement rather than inspection.

| Defect | Effect on results | Evidence |
|---|---|---|
| AE patch reconstructions were unioned and voxel-downsampled | AE output had **165% of the input point count**, inflating its Chamfer term and handing the detector a cloud of the wrong density | retention column of the old table |
| AE Chamfer target was the NN-projection of *every* noisy point, outliers included | ~15% of the target set was surface points clustered at outlier locations — the AE was trained to pile mass there | `datasets.extract_patch_pairs` |
| Temporal filter assumed ego-motion ≪ 0.5 m radius | uncompensated it deletes static structure, not backscatter | measured **mean 0.56 m, max 0.94 m; 50% of frames exceed the radius** |
| Patch centres were sampled randomly with no coverage guarantee | uncovered points kept probability 0 and could never be removed, capping DGCNN removal recall | measured **2,147 of 36,033 points (6%) uncovered** |
| SOR/ROR timers started after the NumPy→Open3D conversion | baseline latency was flattered against end-to-end learned timings | `filters.py` timer placement |
| Empty clouds returned `inf` from the metrics | a frame emptied by aggressive SOR silently skewed the averages | now `NaN`, excluded and counted |
| **Detection claims made from means without a significance test** | produced a headline claim ("removal is not the ceiling") that **reversed** when tested properly | 10 frames → **28 frames × 3 seeds**; every detection claim now carries a paired Wilcoxon p-value |

The last one is worth dwelling on, because it is the most instructive
mistake in the project. Detection recall varies enormously between frames
(σ ≈ 0.3 on a mean of 0.4), so at n = 10 a difference of 0.05 between means
looks convincing and means nothing. Two methods appeared to beat a
perfect-removal oracle; at n = 84 both are significantly *worse* than it.
The geometric metrics never had this problem — their between-frame variance
is an order of magnitude smaller — which is exactly why the same sample size
was wrongly assumed to be adequate for both.

**Outcome of the re-measurement.** Corrections (1) and (2) both bear on the
project's headline claim that Chamfer Distance fails to predict downstream
detection quality, because that claim rested largely on the PointNet-AE's
collapse and the AE was handicapped by two bugs unrelated to its
architecture. Re-measured after the fixes, **the claim holds and is better
supported than before**: the AE now emits exactly 100% of the input point
count, is measurably better than the noisy input on CD (+8.9%), and *still*
loses 76% of the detector's recall (p < 0.0001). The previous objection —
that its collapse was an artefact of 165% densification — no longer applies.
The recombination ablation additionally shows CD improving as point count
rises to 232%, a second and independent demonstration of the same weakness
in the metric.

Corrections (3) and (4) changed results in the direction the analysis
predicted. ICP compensation improves the temporal filter from CD 0.227 to
**0.177** while retaining 5 percentage points more of the cloud (83.7% vs
78.4%) — i.e. uncompensated it really was deleting real structure. Closing
the coverage gap costs **+27% patches (+38 ms/frame)**, a latency regression
accepted in exchange for a removal recall no longer capped by a sampling
artefact; the DGCNN's CD improvement over the previous run rose from 59% to
80%.

## Deferred: hardware component

The proposal's YDLIDAR X2 / Raspberry Pi live-capture component is
**out of scope for this software deliverable** — it requires the physical
sensor. The pipeline is sensor-agnostic ((N,3) or (N,4) XYZI arrays in), so
a future `ydlidar` reader can plug into `kitti_io.load_bin`'s place
unchanged.

## Known limitations

Stated plainly, because each bounds what the results above can be taken to
show:

- **The noise is synthetic.** LISA is a physically motivated approximation,
  not measured adverse weather. Validation on real corrupted data (STF,
  CADC, WADS) is the natural next step and has not been done.
- **Detection is measured by consistency, not accuracy.** It bounds
  recoverable performance against the detector's own clean-cloud output; it
  is not absolute mAP, because the raw split has no ground-truth boxes.
- **The dataset is small and homogeneous** — 262 scans from two drives of a
  single urban daytime recording session.
- **Models are not converged.** The DGCNN's validation loss was still
  descending at epoch 10, so its figures are a floor on what the
  architecture achieves, not a ceiling.
- **Self-supervision does not yet work** (see above): the network reproduces
  its pseudo-labels rather than improving on them.
- **The hardware component is deferred**, as noted above.

## Reproducibility

All randomness is seeded (`data.seed`, per-frame noise seeds). Training
writes CSV logs to `results/logs/` including per-epoch validation loss; the
checkpoint kept is the best-on-validation one, not the last.
`configs/default.yaml` is the single source of truth for paths and
hyperparameters.
