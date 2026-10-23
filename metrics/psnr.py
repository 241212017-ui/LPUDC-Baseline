"""PSNR metric.

Higher is better.
PAPER FACT: PSNR is one of the evaluation metrics (Tables 1–2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute PSNR for NCHW tensors in [0, data_range].

    Returns a scalar tensor (mean over batch).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"psnr shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    pred = pred.detach().float().clamp(0.0, data_range)
    target = target.detach().float().clamp(0.0, data_range)
    # Per-image MSE
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=eps)
    val = 10.0 * torch.log10((data_range ** 2) / mse)
    return val.mean()


def psnr_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Per-image PSNR, shape (N,)."""
    if pred.shape != target.shape:
        raise ValueError("psnr_per_image shape mismatch")
    pred = pred.detach().float().clamp(0.0, data_range)
    target = target.detach().float().clamp(0.0, data_range)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=eps)
    return 10.0 * torch.log10((data_range ** 2) / mse)
