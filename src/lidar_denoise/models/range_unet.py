"""RangeRestoreNet — a dual-head range-image denoiser.

**Motivation (this is the project's own contribution).**

The recent adverse-weather LiDAR literature is essentially unanimous on two
choices. It works on a 2-D range image, because that is what makes real-time
throughput possible (WeatherNet, LiSnowNet, 4DenoiseNet, LRINet, LIORNet,
TripleMixer all report 16-43 Hz where our point-graph DGCNN manages 0.59 Hz).
And it treats denoising as **point-wise classification** — every one of those
methods emits a binary noise mask and is scored with IoU/mIoU. The August
2025 survey of the field files them all under "outlier removal", and notes
in passing that outlier removal "can only drop noisy points without
adjusting their positions".

Our oracle analysis says that limitation is not passing at all — it binds.
An oracle that deletes every injected point and keeps every real one reaches
only 0.451 detection-consistency recall, while our SOR+ROR (0.499) and DGCNN
(0.521) both *beat* it, because they also move surviving points back toward
the surface. **Perfect classification is not the ceiling.** Removal and
restoration are complementary, and the fast representation the field
standardised on has so far only been used for the first of them.

This module combines the two: a lightweight U-Net over the range image with

* a **removal head** — per-pixel noise logit, as the literature does; and
* a **restoration head** — per-pixel *range residual*, predicting how far
  along its own ray a surviving return should move to undo weather-induced
  range error.

Predicting a scalar residual along the ray rather than a free 3-D
displacement is the natural parameterisation here: the sensor's angular
measurement is essentially exact, and it is the time-of-flight that
precipitation corrupts. It also keeps the output on the image grid, so the
whole method stays a 2-D convolution and inherits the speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BN + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RangeRestoreNet(nn.Module):
    """Dual-head U-Net over a (B, 5, H, W) range image.

    Downsampling is applied to the azimuth axis only in the first stage: a
    64-row image is already shallow, and halving height too early destroys
    the vertical laser structure that separates a surface from a floating
    droplet.

    Returns (noise_logits (B, H, W), range_residual (B, H, W)) where the
    residual is in normalised range units, matching `range_image.project`.
    """

    def __init__(self, in_ch: int = 5, base: int = 32, depth: int = 3,
                 max_correction: float = 0.5):
        super().__init__()
        self.depth = depth
        # The restoration head emits a BOUNDED residual in METRES, not in the
        # image's normalised range units. This is not cosmetic: normalised
        # residuals are divided by the 60 m range gate, so an L1 of 0.003 —
        # which looks converged — is 0.18 m of real error, against a true
        # weather displacement of only 0.08 m. Trained that way the head
        # tripled the displacement error it was supposed to remove (0.080 ->
        # 0.219 m) and made CD worse than not restoring at all. Bounding the
        # output with tanh to +/- max_correction puts the head's dynamic
        # range on the same scale as the correction it must learn.
        self.max_correction = max_correction
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            out = base * (2**d)
            self.encoders.append(ConvBlock(ch, out))
            # first stage: azimuth-only downsample, preserving the 64 rows
            self.pools.append(nn.MaxPool2d((1, 2) if d == 0 else (2, 2)))
            ch = out

        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in reversed(range(depth)):
            skip = base * (2**d)
            self.ups.append(
                nn.ConvTranspose2d(ch, skip, (1, 2) if d == 0 else (2, 2),
                                   stride=(1, 2) if d == 0 else (2, 2))
            )
            self.decoders.append(ConvBlock(skip * 2, skip))
            ch = skip

        self.removal_head = nn.Conv2d(ch, 1, 1)
        self.restore_head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:            # guard odd sizes
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = dec(torch.cat([x, skip], dim=1))
        residual_m = torch.tanh(self.restore_head(x).squeeze(1)) * self.max_correction
        return self.removal_head(x).squeeze(1), residual_m


def dual_head_loss(
    logits: torch.Tensor,
    residual: torch.Tensor,
    noise_target: torch.Tensor,
    range_target: torch.Tensor,
    range_input: torch.Tensor,
    valid: torch.Tensor,
    restore_weight: float = 1.0,
    pos_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked BCE on the removal head + masked L1 on the restoration head.

    Both terms are evaluated only on occupied pixels, and the restoration
    term additionally only on pixels holding a *real* return: asking the
    network to regress a corrected range for a phantom droplet return would
    be meaningless, and it is the removal head's job to delete those.

    `residual`, `range_input` and `range_target` are all in METRES here — see
    the note in RangeRestoreNet on why normalised units made this head
    unlearnable.

    `pos_weight` counters the class imbalance — injected points are roughly a
    third of the cloud, and an unweighted BCE drifts toward predicting the
    majority class early in training.
    """
    valid = valid.bool()
    real = valid & (noise_target < 0.5)

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits[valid], noise_target[valid],
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    if real.any():
        predicted = range_input[real] + residual[real]
        l1 = nn.functional.l1_loss(predicted, range_target[real])
    else:
        l1 = torch.zeros((), device=logits.device)
    return bce + restore_weight * l1, bce, l1
