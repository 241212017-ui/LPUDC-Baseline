"""LPIPS metric wrapper.

Lower is better.
PAPER FACT: LPIPS (Zhang et al.) is one of the evaluation metrics.
IMPLEMENTATION ASSUMPTION: Uses the official `lpips` package when available
(backbone default 'alex'). The paper does not state the backbone. If the
package or weights are unavailable, metrics are reported as None with a
clear warning — we never fabricate values.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn


class LPIPSMetric(nn.Module):
    """LPIPS wrapper.

    Args:
        net: 'alex' | 'vgg' | 'squeeze' (paper does not specify; default alex)
        device: torch device
    """

    def __init__(self, net: str = "alex", device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.net_name = net
        self.device = torch.device(device)
        self.available = False
        self._lpips: nn.Module | None = None
        try:
            import lpips  # type: ignore

            self._lpips = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"LPIPS unavailable ({exc}). Install `lpips` to enable this metric. "
                "Values will be reported as null — never fabricated.",
                RuntimeWarning,
                stacklevel=2,
            )

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float | None:
        """Return mean LPIPS (lower is better) or None if unavailable.

        Expects NCHW tensors in [0, 1]. Internally maps to [-1, 1].
        """
        if not self.available or self._lpips is None:
            return None
        pred = pred.detach().to(self.device).float().clamp(0.0, 1.0)
        target = target.detach().to(self.device).float().clamp(0.0, 1.0)
        pred_n = pred * 2.0 - 1.0
        target_n = target * 2.0 - 1.0
        val = self._lpips(pred_n, target_n)
        return float(val.mean().item())


def build_lpips(net: str = "alex", device: str | torch.device = "cpu") -> LPIPSMetric:
    return LPIPSMetric(net=net, device=device)
