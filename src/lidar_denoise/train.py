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


def _make_logger(log_path: Path, fields: list[str]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    def log(row: dict) -> None:
        writer.writerow(row)
        f.flush()

    return log


def train_model(model_name: str, cache_path: Path, cfg: dict) -> Path:
    """Train 'pointnet_ae' or 'dgcnn' on a patch cache; returns checkpoint path."""
    tcfg = cfg["training"]
    device = select_device(tcfg["device"])
    dataset = PatchDataset(cache_path)
    loader = DataLoader(
        dataset, batch_size=tcfg["batch_size"], shuffle=True, num_workers=tcfg["num_workers"]
    )

    k = cfg["patches"]["points_per_patch"]
    if model_name == "pointnet_ae":
        mcfg = cfg["model"]["pointnet_ae"]
        model = PointNetAutoencoder(k, mcfg["latent_dim"], mcfg["decoder_hidden"])
    elif model_name == "dgcnn":
        mcfg = cfg["model"]["dgcnn"]
        model = DGCNNDenoiser(mcfg["k"], mcfg["hidden_dims"], mcfg["emb_dim"])
    else:
        raise ValueError(f"unknown model: {model_name}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss()

    log = _make_logger(
        resolve(cfg, tcfg["log_dir"]) / f"{model_name}_train.csv",
        ["epoch", "loss", "recon_loss", "outlier_loss", "seconds"],
    )
    ckpt_dir = resolve(cfg, tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}.pt"

    print(f"[train] {model_name} on {device}, {len(dataset)} patches, "
          f"{tcfg['epochs']} epochs")
    model.train()
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.perf_counter()
        tot = tot_recon = tot_out = 0.0
        for noisy, clean, labels in loader:
            noisy, clean, labels = noisy.to(device), clean.to(device), labels.to(device)
            opt.zero_grad()
            if model_name == "pointnet_ae":
                recon = model(noisy)
                # AE target: the clean patch points only (outlier targets are
                # duplicates of far-away surface points; Chamfer handles this).
                recon_loss = chamfer_loss(recon, clean)
                out_loss = torch.zeros((), device=device)
                loss = recon_loss
            else:
                denoised, logits = model(noisy)
                real = labels < 0.5                                  # (B, N)
                diff = (denoised - clean).pow(2).sum(dim=2)          # (B, N)
                recon_loss = (diff * real).sum() / real.sum().clamp(min=1)
                out_loss = bce(logits, labels)
                loss = recon_loss + cfg["training"]["outlier_loss_weight"] * out_loss
            loss.backward()
            opt.step()
            tot += loss.item(); tot_recon += recon_loss.item(); tot_out += out_loss.item()
        n = len(loader)
        secs = time.perf_counter() - t0
        log({"epoch": epoch, "loss": tot / n, "recon_loss": tot_recon / n,
             "outlier_loss": tot_out / n, "seconds": round(secs, 2)})
        if epoch % 5 == 0 or epoch == 1 or epoch == tcfg["epochs"]:
            print(f"  epoch {epoch:3d}/{tcfg['epochs']}  loss {tot / n:.6f}  ({secs:.1f}s)")
        torch.save({"model": model.state_dict(), "config": mcfg, "k": k}, ckpt_path)
    print(f"[train] checkpoint saved to {ckpt_path}")
    return ckpt_path
