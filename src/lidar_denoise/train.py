"""Training loops for both learning-based denoisers, with CSV logging.

Losses:
* PointNet AE — squared Chamfer between the reconstructed patch and the
  clean target patch (the AE must also learn to place points *away* from
  injected outliers, since outliers have no nearby clean target).
* DGCNN — masked MSE on displacements for real points, plus BCE on the
  outlier head for all points (weighted by config).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import resolve
from .datasets import PatchDataset
from .models import DGCNNDenoiser, PointNetAutoencoder
from .models.pointnet_ae import chamfer_loss


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_logger(log_path: Path, fields: list[str], append: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = append and log_path.exists()
    f = open(log_path, "a" if existing else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    if not existing:
        writer.writeheader()

    def log(row: dict) -> None:
        writer.writerow(row)
        f.flush()

    return log


def _run_batch(model, model_name, batch, device, bce, cfg):
    """Forward pass + loss for one batch; shared by train and validation."""
    noisy, clean, clean_set, labels = (t.to(device) for t in batch)
    if model_name == "pointnet_ae":
        # Chamfer against a REAL clean patch (clean_set), not the
        # index-aligned NN projection: see datasets.extract_patch_pairs.
        recon = model(noisy)
        recon_loss = chamfer_loss(recon, clean_set)
        out_loss = torch.zeros((), device=device)
        return recon_loss, recon_loss, out_loss
    denoised, logits = model(noisy)
    real = labels < 0.5                                      # (B, N)
    diff = (denoised - clean).pow(2).sum(dim=2)              # (B, N)
    recon_loss = (diff * real).sum() / real.sum().clamp(min=1)
    out_loss = bce(logits, labels)
    loss = recon_loss + cfg["training"]["outlier_loss_weight"] * out_loss
    return loss, recon_loss, out_loss


def train_model(
    model_name: str,
    cache_path: Path,
    cfg: dict,
    val_cache_path: Path | None = None,
    resume: bool = False,
) -> Path:
    """Train 'pointnet_ae' or 'dgcnn' on a patch cache; returns checkpoint path.

    When `val_cache_path` is given, validation loss is computed each epoch and
    the checkpoint kept is the best-on-validation one, not simply the last.
    The val split was previously carved out by `split_scans` and then never
    used, so nothing distinguished a converged model from an overfitted one.
    """
    tcfg = cfg["training"]
    device = select_device(tcfg["device"])
    dataset = PatchDataset(cache_path)
    loader = DataLoader(
        dataset, batch_size=tcfg["batch_size"], shuffle=True, num_workers=tcfg["num_workers"]
    )
    val_loader = None
    if val_cache_path is not None and Path(val_cache_path).exists():
        val_loader = DataLoader(
            PatchDataset(val_cache_path), batch_size=tcfg["batch_size"],
            shuffle=False, num_workers=tcfg["num_workers"],
        )

    k = cfg["patches"]["points_per_patch"]
    if model_name == "pointnet_ae":
        mcfg = cfg["model"]["pointnet_ae"]
        model = PointNetAutoencoder(k, mcfg["latent_dim"], mcfg["decoder_hidden"])
    elif model_name == "dgcnn":
        mcfg = cfg["model"]["dgcnn"]
        model = DGCNNDenoiser(mcfg["k"], mcfg["hidden_dims"], mcfg["emb_dim"],
                              mcfg.get("static_graph", True))
    else:
        raise ValueError(f"unknown model: {model_name}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss()

    ckpt_dir = resolve(cfg, tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}.pt"

    # Resume from an existing checkpoint when asked. The DGCNN takes ~5 min
    # per epoch on MPS, long enough that an interrupted run would otherwise
    # throw away the whole thing; restarting from scratch is not an
    # acceptable failure mode for a multi-hour training budget.
    start_epoch, best_val, best_epoch = 1, float("inf"), 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("val_loss", float("inf")))
        best_epoch = int(ckpt.get("epoch", 0))
        print(f"[train] resuming {model_name} from epoch {start_epoch} "
              f"(best val {best_val:.6f} at epoch {best_epoch})")
        if start_epoch > tcfg["epochs"]:
            print(f"[train] already at {tcfg['epochs']} epochs; nothing to do")
            return ckpt_path

    log = _make_logger(
        resolve(cfg, tcfg["log_dir"]) / f"{model_name}_train.csv",
        ["epoch", "loss", "recon_loss", "outlier_loss", "val_loss", "seconds"],
        append=start_epoch > 1,
    )

    print(f"[train] {model_name} on {device}, {len(dataset)} patches, "
          f"epochs {start_epoch}-{tcfg['epochs']}"
          + (f", val {len(val_loader.dataset)} patches" if val_loader else ", no val set"))
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        tot = tot_recon = tot_out = 0.0
        for batch in loader:
            opt.zero_grad()
            loss, recon_loss, out_loss = _run_batch(model, model_name, batch, device, bce, cfg)
            loss.backward()
            opt.step()
            tot += loss.item(); tot_recon += recon_loss.item(); tot_out += out_loss.item()
        n = len(loader)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    _run_batch(model, model_name, b, device, bce, cfg)[0].item()
                    for b in val_loader
                ) / len(val_loader)

        secs = time.perf_counter() - t0
        log({"epoch": epoch, "loss": tot / n, "recon_loss": tot_recon / n,
             "outlier_loss": tot_out / n, "val_loss": val_loss, "seconds": round(secs, 2)})
        if epoch % 5 == 0 or epoch == 1 or epoch == tcfg["epochs"]:
            print(f"  epoch {epoch:3d}/{tcfg['epochs']}  train {tot / n:.6f}  "
                  f"val {val_loss:.6f}  ({secs:.1f}s)")

        # Keep the best-on-validation weights; with no val set, keep the last.
        # The optimizer state rides along so an interrupted run can resume
        # with its Adam moments intact rather than restarting cold.
        if val_loader is None or val_loss < best_val:
            best_val, best_epoch = (val_loss if val_loader else float("nan")), epoch
            torch.save(
                {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "config": mcfg, "k": k, "epoch": epoch, "val_loss": val_loss},
                ckpt_path,
            )
    if val_loader is not None:
        print(f"[train] best epoch {best_epoch} (val {best_val:.6f}) -> {ckpt_path}")
    else:
        print(f"[train] checkpoint saved to {ckpt_path}")
    return ckpt_path
