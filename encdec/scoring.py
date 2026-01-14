from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


LossType = Literal["mse", "smoothl1"]


def build_loss(loss_type: LossType) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss(reduction="mean")
    if loss_type == "smoothl1":
        return nn.SmoothL1Loss(reduction="mean")
    raise ValueError("training.loss_type must be one of: mse, smoothl1")


@torch.no_grad()
def anomaly_score_mse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    err = (x - recon) ** 2
    return err.mean(dim=(1, 2, 3))

