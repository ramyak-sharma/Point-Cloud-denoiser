"""Training for the dual-head range-image denoiser.

Range images are built on the fly rather than cached. Each is ~2.6 MB, so a
213-frame cache would be over half a gigabyte, and re-projecting costs only
~10 ms. More importantly, generating the corruption fresh every epoch means
the network sees a new noise draw of each frame each time — free
augmentation that a fixed patch cache cannot provide, and the reason this
model tolerates a 213-frame training set.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from . import range_image as RI
from .config import resolve
from .kitti_io import load_bin
from .models.range_unet import RangeRestoreNet, dual_head_loss
from .noise import inject_noise
from .preprocessing import preprocess
from .train import select_device


def build_targets(clean: np.ndarray, noisy, height: int, width: int, max_range: float,
                  self_supervised: bool = False):
    """Project one corrupted frame and derive both heads' targets.

    Returns (image, noise_target, range_target, valid, range_in_m) as float32
    arrays on the image grid. The range target is the *clean* range of each
    real return's known source point — an exact target, available because
    `inject_noise` records the correspondence.

    With ``self_supervised=True`` the noise target comes from
    `pseudo_labels.pseudo_noise_labels` instead: derived from the corrupted
    scan's own physics, with no access to the generator's labels. That is the
    only configuration that could be trained on real weather data, where no
    ground truth exists.
    """
    ri = RI.project(noisy.points, height=height, width=width, max_range=max_range)
    pi = ri.point_idx.ravel()
    occupied = pi >= 0
    idx = pi[occupied]

    if self_supervised:
        from .pseudo_labels import pseudo_noise_labels

        point_labels, _ = pseudo_noise_labels(
            noisy.points, height=height, width=width, max_range=max_range
        )
    else:
        point_labels = noisy.labels

    noise_t = np.zeros(height * width, np.float32)
    noise_t[occupied] = point_labels[idx].astype(np.float32)

    # Clean range in METRES for pixels holding a real return; elsewhere copy
    # the input range so the residual target there is zero and contributes
    # nothing. Metres (not normalised units) because the head predicts a
    # bounded metre residual — see RangeRestoreNet.
    range_in_m = ri.image[0].ravel() * max_range
    range_t = np.zeros(height * width, np.float32)
    range_t[occupied] = range_in_m[occupied]
    src = noisy.source_idx[idx]
    has_src = src >= 0
    if has_src.any():
        clean_r = np.linalg.norm(clean[src[has_src], :3], axis=1)
        target_slots = np.flatnonzero(occupied)[has_src]
        range_t[target_slots] = clean_r.astype(np.float32)

    return (
        ri.image,
        noise_t.reshape(height, width),
        range_t.reshape(height, width),
        ri.valid.astype(np.float32),
        range_in_m.reshape(height, width).astype(np.float32),
    )


def _frame_batch(paths, cfg, seed_base, height, width, max_range, self_supervised=False):
    """Load, corrupt and project a set of frames into one stacked batch."""
    imgs, noise_ts, range_ts, valids, range_ins = [], [], [], [], []
    for i, p in enumerate(paths):
        clean = preprocess(load_bin(p, with_intensity=True), cfg)
        noisy = inject_noise(clean, cfg, seed=seed_base + i)
        img, nt, rt, v, rin = build_targets(clean, noisy, height, width, max_range,
                                            self_supervised)
        imgs.append(img); noise_ts.append(nt); range_ts.append(rt)
        valids.append(v); range_ins.append(rin)
    to = lambda a: torch.from_numpy(np.stack(a))
    return to(imgs), to(noise_ts), to(range_ts), to(valids), to(range_ins)


def train_range_model(cfg: dict, train_paths, val_paths, resume: bool = False) -> Path:
    """Train RangeRestoreNet; keeps the best-on-validation checkpoint."""
    rcfg = cfg["model"]["range_unet"]
    tcfg = cfg["training"]
    H, W = rcfg["height"], rcfg["width"]
    max_range = cfg["preprocessing"]["max_range"]
    device = select_device(tcfg["device"])

    model = RangeRestoreNet(base=rcfg["base"], depth=rcfg["depth"],
                            max_correction=rcfg.get("max_correction", 0.5)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rcfg.get("lr", tcfg["lr"]),
                           weight_decay=tcfg["weight_decay"])

    ckpt_dir = resolve(cfg, tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    self_sup = bool(rcfg.get("self_supervised", False))
    tag = "range_unet_selfsup" if self_sup else "range_unet"
    ckpt_path = ckpt_dir / f"{tag}.pt"
    # The restoration target is the clean range of a known source point —
    # ground truth by construction. A self-supervised run must not use it,
    # or the "no labels needed" claim is false. The head is switched off
    # rather than fed a surrogate; it was shown to contribute nothing even
    # when fully supervised, so nothing is lost by dropping it here.
    restore_weight = 0.0 if self_sup else rcfg.get("restore_weight", 1.0)
    start_epoch, best_val = 1, float("inf")
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch, best_val = int(ck.get("epoch", 0)) + 1, float(ck.get("val_loss", np.inf))
        print(f"[range] resuming from epoch {start_epoch} (best val {best_val:.6f})")

    log_path = resolve(cfg, tcfg["log_dir"]) / f"{tag}_train.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = resume and log_path.exists()
    f = open(log_path, "a" if existing else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "bce", "l1", "val_loss", "seconds"])
    if not existing:
        writer.writeheader()

    bs = rcfg.get("batch_size", 2)
    epochs = rcfg.get("epochs", tcfg["epochs"])
    print(f"[range] {len(train_paths)} train / {len(val_paths)} val frames on {device}, "
          f"{H}x{W}, batch {bs}, epochs {start_epoch}-{epochs}"
          + ("  [SELF-SUPERVISED: pseudo-labels only]" if self_sup else ""))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_paths))
        tot = tot_b = tot_l = 0.0
        nb = 0
        for s in range(0, len(order), bs):
            paths = [train_paths[j] for j in order[s : s + bs]]
            # fresh noise seed per epoch: augmentation, not a fixed cache
            img, nt, rt, v, rin = _frame_batch(paths, cfg, 10_000 * epoch + s, H, W, max_range, self_sup)
            img, nt, rt, v, rin = (t.to(device) for t in (img, nt, rt, v, rin))
            opt.zero_grad()
            logits, res = model(img)
            loss, bce, l1 = dual_head_loss(
                logits, res, nt, rt, rin, v,
                restore_weight=restore_weight,
                pos_weight=rcfg.get("pos_weight", 2.0),
            )
            loss.backward()
            opt.step()
            tot += loss.item(); tot_b += bce.item(); tot_l += l1.item(); nb += 1

        model.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(val_paths), bs):
                paths = val_paths[s : s + bs]
                img, nt, rt, v, rin = _frame_batch(paths, cfg, 777_000 + s, H, W, max_range, self_sup)
                img, nt, rt, v, rin = (t.to(device) for t in (img, nt, rt, v, rin))
                logits, res = model(img)
                loss, _, _ = dual_head_loss(
                    logits, res, nt, rt, rin, v,
                    restore_weight=restore_weight,
                    pos_weight=rcfg.get("pos_weight", 2.0),
                )
                vtot += loss.item(); vnb += 1
        val_loss = vtot / max(vnb, 1)
        secs = time.perf_counter() - t0
        writer.writerow({"epoch": epoch, "loss": tot / nb, "bce": tot_b / nb,
                         "l1": tot_l / nb, "val_loss": val_loss, "seconds": round(secs, 2)})
        f.flush()
        print(f"  epoch {epoch:3d}/{epochs}  train {tot / nb:.5f} "
              f"(bce {tot_b / nb:.5f}, l1 {tot_l / nb:.5f})  val {val_loss:.5f}  ({secs:.0f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "config": rcfg, "epoch": epoch, "val_loss": val_loss}, ckpt_path)
    f.close()
    print(f"[range] best val {best_val:.6f} -> {ckpt_path}")
    return ckpt_path
