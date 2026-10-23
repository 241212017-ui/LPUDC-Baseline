"""SSIM metric.

Higher is better.
PAPER FACT: SSIM is one of the evaluation metrics (Tables 1–2).

IMPLEMENTATION ASSUMPTION: Standard SSIM with 11x11 Gaussian window,
sigma=1.5, data_range=1.0 (images in [0,1]), as in the original Wang et al.
definition. The paper does not state window size or data range.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d.view(1, 1, window_size, window_size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute mean SSIM for NCHW tensors in [0, data_range]."""
    if pred.shape != target.shape:
        raise ValueError(f"ssim shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    pred = pred.detach().float().clamp(0.0, data_range)
    target = target.detach().float().clamp(0.0, data_range)

    c = pred.shape[1]
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(c, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_p = F.conv2d(pred, window, padding=pad, groups=c)
    mu_t = F.conv2d(target, window, padding=pad, groups=c)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_p2
    sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=c) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    )
    # Mean over batch, channels, spatial dims (valid region only if desired;
    # we use full map including border, matching common implementations).
    return ssim_map.mean()


def ssim_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Per-image SSIM, shape (N,)."""
    if pred.shape != target.shape:
        raise ValueError("ssim_per_image shape mismatch")
    pred = pred.detach().float().clamp(0.0, data_range)
    target = target.detach().float().clamp(0.0, data_range)

    c = pred.shape[1]
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=c)
    mu_t = F.conv2d(target, window, padding=pad, groups=c)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_p2
    sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=c) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_pt
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))
