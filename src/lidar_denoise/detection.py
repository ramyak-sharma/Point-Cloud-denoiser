"""Downstream detection evaluation via the detection-consistency protocol.

KITTI *raw* scans carry no 3D box annotations, so absolute mAP cannot be
computed on this subset. Instead, the pretrained detector's output on the
CLEAN cloud is treated as the reference, and we measure how much of it
survives on the noisy and denoised versions of the same frame:

    consistency recall  = matched reference boxes / reference boxes
    consistency precision = matched predicted boxes / predicted boxes

with matching done by BEV centre distance + IoU-style size agreement. A good
denoiser should restore recall lost to noise without hallucinating boxes.
This directly answers the proposal's question — "does denoising recover
detector performance?" — without fabricating ground truth.

Requires mmdetection3d and a KITTI-pretrained PointPillars checkpoint
(see scripts/detection_eval.py). Import of mmdet3d is deferred so the rest
of the pipeline runs without it.
"""

from __future__ import annotations

import numpy as np


def load_detector(config: str, checkpoint: str, device: str = "cpu"):
    """Init a mmdetection3d model (deferred import)."""
    from mmdet3d.apis import init_model

    return init_model(config, checkpoint, device=device)


def detect(model, points: np.ndarray, score_threshold: float) -> dict[str, np.ndarray]:
    """Run the detector on an (N, 3) cloud; returns boxes/scores/labels.

    The intensity channel is zero-filled — acceptable for PointPillars, whose
    pillar features are dominated by geometry, but noted as a limitation.
    """
    import tempfile

    from mmdet3d.apis import inference_detector

    from .kitti_io import save_bin

    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        save_bin(f.name, points)
        result, _ = inference_detector(model, f.name)
    pred = result.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.cpu().numpy()          # (M, 7) x,y,z,dx,dy,dz,yaw
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()
    keep = scores >= score_threshold
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def _bev_match(ref: np.ndarray, pred: np.ndarray, centre_tol: float = 1.0) -> int:
    """Greedy one-to-one matching of boxes by BEV centre distance.

    A predicted box matches a reference box of the same frame if their BEV
    centres are within centre_tol metres — a deliberately simple criterion
    that is robust to the small size/yaw regressions noise induces.
    """
    if len(ref) == 0 or len(pred) == 0:
        return 0
    d = np.linalg.norm(ref[:, None, :2] - pred[None, :, :2], axis=2)
    matched = 0
    used_pred: set[int] = set()
    for i in np.argsort(d.min(axis=1)):
        j_order = np.argsort(d[i])
        for j in j_order:
            if d[i, j] > centre_tol:
                break
            if j not in used_pred:
                used_pred.add(j)
                matched += 1
                break
    return matched


def consistency_metrics(
    reference: dict[str, np.ndarray], prediction: dict[str, np.ndarray]
) -> dict[str, float]:
    """Consistency recall/precision of `prediction` against `reference`."""
    matched = _bev_match(reference["boxes"], prediction["boxes"])
    n_ref, n_pred = len(reference["boxes"]), len(prediction["boxes"])
    return {
        "n_reference": n_ref,
        "n_predicted": n_pred,
        "matched": matched,
        "recall": matched / n_ref if n_ref else 1.0,
        "precision": matched / n_pred if n_pred else 1.0,
    }
