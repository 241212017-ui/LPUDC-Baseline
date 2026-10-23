"""Metrics package.

Direction of improvement:
    PSNR  — higher is better
    SSIM  — higher is better
    LPIPS — lower is better
    DISTS — lower is better
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .dists_metric import DISTSMetric, build_dists
from .lpips_metric import LPIPSMetric, build_lpips
from .psnr import psnr, psnr_per_image
from .ssim import ssim, ssim_per_image

__all__ = [
    "psnr",
    "psnr_per_image",
    "ssim",
    "ssim_per_image",
    "LPIPSMetric",
    "DISTSMetric",
    "build_lpips",
    "build_dists",
    "MetricSuite",
]


class MetricSuite:
    """Bundles PSNR / SSIM / LPIPS / DISTS with correct directions."""

    # metric name -> 'higher' | 'lower'
    DIRECTIONS = {
        "PSNR": "higher",
        "SSIM": "higher",
        "LPIPS": "lower",
        "DISTS": "lower",
    }

    def __init__(
        self,
        device: str | torch.device = "cpu",
        lpips_net: str = "alex",
        enable_lpips: bool = True,
        enable_dists: bool = True,
        dists_pretrained: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.lpips: LPIPSMetric | None = None
        self.dists: DISTSMetric | None = None
        if enable_lpips:
            self.lpips = build_lpips(net=lpips_net, device=self.device)
        if enable_dists:
            self.dists = build_dists(pretrained=dists_pretrained, device=self.device)

    @torch.no_grad()
    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float | None]:
        """Compute all metrics for a batch. Missing metrics are None."""
        pred = pred.detach()
        target = target.detach()
        out: dict[str, float | None] = {
            "PSNR": float(psnr(pred, target).item()),
            "SSIM": float(ssim(pred, target).item()),
            "LPIPS": None,
            "DISTS": None,
        }
        if self.lpips is not None:
            out["LPIPS"] = self.lpips(pred, target)
        if self.dists is not None:
            out["DISTS"] = self.dists(pred, target)
        return out
