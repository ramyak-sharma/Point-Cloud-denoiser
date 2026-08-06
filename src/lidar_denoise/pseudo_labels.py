"""Physics-based noise pseudo-labels — supervision without ground truth.

The single strongest objection to this project is that its supervision is
circular: we inject the noise, so of course we can identify it. Every
learned denoiser here is trained on `inject_noise`'s own label array, which
no real deployment would ever have. There is no measured dataset of the same
scene with and without rain, so paired supervision cannot simply be
sourced elsewhere.

The escape, following LIORNet [Seppänen-style self-supervision, arXiv
2603.19936], is to derive labels from the *physics* of a weather return
rather than from the generator. Three rules are used, all computable from a
single corrupted scan:

1. **Range-conditioned intensity anomaly.** Received power falls as
   :math:`\\rho / r^2`, so a solid surface at 40 m legitimately returns less
   energy than one at 10 m. Comparing raw intensity against a global
   threshold would therefore just relabel "far" as "noise". Instead the
   median intensity is estimated *per range bin* from the scan itself, and a
   return is anomalous when it falls well below what its own range predicts.

2. **Neighbourhood range coherence.** A real surface is locally continuous
   in the range image: adjacent pixels see nearly the same distance. A
   droplet return floats in front of whatever it occluded, so its neighbours
   disagree with it. This is the range-image analogue of the density rule,
   and it is the strongest of the three signals.

3. **Edge protection.** Genuine object boundaries — a car silhouette against
   background — also produce range discontinuities and would be caught by
   rule 2. Real edges are *coherent*: they persist across a run of adjacent
   pixels. Isolated discontinuities are not protected; extended ones are.

LIORNet additionally exploits zero-reflectivity, which Ouster sensors report
and the KITTI HDL-64E does not, so that rule is omitted rather than faked.

Nothing here reads `NoisyCloud.labels`. Those are used only to *score* the
pseudo-labels afterwards, which is a diagnostic, not supervision.
"""

from __future__ import annotations

import numpy as np

from . import range_image as RI


def _range_conditioned_intensity_ratio(
    r: np.ndarray, intensity: np.ndarray, n_bins: int = 24
) -> np.ndarray:
    """intensity / (median intensity of returns at a similar range).

    Values well below 1 mean "weaker than its own range explains". Bins with
    too few samples fall back to the global median so a sparse far-field bin
    cannot manufacture anomalies out of noise in the estimate itself.
    """
    ratio = np.ones_like(intensity)
    if len(r) == 0:
        return ratio
    edges = np.linspace(0.0, max(r.max(), 1e-6), n_bins + 1)
    which = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    global_median = float(np.median(intensity)) or 1.0
    for b in range(n_bins):
        m = which == b
        if m.sum() < 20:
            ref = global_median
        else:
            ref = float(np.median(intensity[m])) or global_median
        ratio[m] = intensity[m] / max(ref, 1e-6)
    return ratio


def _neighbour_disagreement(
    range_img: np.ndarray, valid: np.ndarray, window: int = 2, tol: float = 0.5
) -> np.ndarray:
    """Fraction of occupied neighbours whose range differs by more than `tol`.

    Computed by shifting the image rather than looping — a (2*window+1)^2
    stack of np.roll views, which keeps this a few milliseconds per frame.
    Points on a continuous surface score near 0; a floating droplet return
    scores near 1.
    """
    h, w = range_img.shape
    agree = np.zeros((h, w), np.float32)
    total = np.zeros((h, w), np.float32)
    for dv in range(-window, window + 1):
        for du in range(-window, window + 1):
            if dv == 0 and du == 0:
                continue
            nr = np.roll(np.roll(range_img, dv, axis=0), du, axis=1)
            nv = np.roll(np.roll(valid, dv, axis=0), du, axis=1)
            close = nv & (np.abs(nr - range_img) < tol)
            agree += close
            total += nv
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, agree / np.maximum(total, 1), 0.0)
    return (1.0 - frac).astype(np.float32)


def _edge_coherence(range_img: np.ndarray, valid: np.ndarray, tol: float = 0.5) -> np.ndarray:
    """True where a range discontinuity is part of an extended boundary.

    A real object silhouette produces a discontinuity that continues along
    the neighbouring rows/columns; an isolated droplet produces one that does
    not. Coherence is measured as agreement of the horizontal step sign with
    the rows above and below.
    """
    step = np.abs(np.diff(range_img, axis=1, prepend=range_img[:, :1])) > tol
    step &= valid
    up = np.roll(step, 1, axis=0)
    down = np.roll(step, -1, axis=0)
    return (step & (up | down)).astype(bool)


def pseudo_noise_labels(
    points: np.ndarray,
    height: int = 64,
    width: int = 2048,
    max_range: float = 60.0,
    disagreement_thresh: float = 0.85,
    intensity_ratio_thresh: float = 0.6,
    combine: str = "or",
) -> tuple[np.ndarray, dict]:
    """Per-point noise pseudo-labels for an (N, 4) XYZI cloud.

    Returns (labels uint8, diagnostics). A point is flagged when its
    neighbourhood disagrees with it *or* (by default) its return is far
    weaker than its range predicts — subject to edge protection, which
    rescues points sitting on coherent object boundaries.

    `combine="and"` demands both signals, which raises precision at a large
    cost in recall; "or" is the default because the downstream network is
    tolerant of pseudo-label noise but cannot learn from noise it never sees.

    Defaults were grid-searched on the **validation** split against the true
    labels, maximising pseudo-label F1. At those settings the rules reach
    precision 0.495 / recall 0.483 / F1 0.489, with each rule contributing
    independently (isolation alone F1 0.423, intensity alone 0.388). That is
    deliberately reported: it is the ceiling on the supervision a
    self-supervised model receives, and roughly half of what it is told is
    wrong.
    """
    if points.shape[1] < 4:
        raise ValueError("pseudo_noise_labels needs (N, 4) XYZI input")

    ri = RI.project(points, height=height, width=width, max_range=max_range)
    range_img = ri.image[0] * max_range
    valid = ri.valid

    disagreement = _neighbour_disagreement(range_img, valid)
    protected = _edge_coherence(range_img, valid)

    r = np.linalg.norm(points[:, :3], axis=1)
    ratio = _range_conditioned_intensity_ratio(r, points[:, 3])

    px_disagreement = RI.unproject_labels(ri, disagreement, fill=0.0)
    px_protected = RI.unproject_labels(ri, protected.astype(np.float32), fill=0.0) > 0.5

    isolated = px_disagreement > disagreement_thresh
    weak = ratio < intensity_ratio_thresh
    flagged = (isolated | weak) if combine == "or" else (isolated & weak)
    flagged &= ~px_protected

    labels = flagged.astype(np.uint8)
    return labels, {
        "flagged_pct": float(100.0 * labels.mean()),
        "isolated_pct": float(100.0 * isolated.mean()),
        "weak_pct": float(100.0 * weak.mean()),
        "protected_pct": float(100.0 * px_protected.mean()),
    }


def score_pseudo_labels(pseudo: np.ndarray, truth: np.ndarray) -> dict:
    """Precision/recall/F1 of the pseudo-labels against the real ones.

    A diagnostic only — this comparison never reaches the training loop. It
    answers "how good is the supervision we could obtain without a
    generator?", which bounds what the self-supervised model can learn.
    """
    p, t = pseudo.astype(bool), truth.astype(bool)
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "flagged_pct": float(100.0 * p.mean()), "true_noise_pct": float(100.0 * t.mean())}
