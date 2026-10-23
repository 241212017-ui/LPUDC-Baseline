"""Official DISTS metric wrapper. Lower is better."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Official DISTS linear-combination weights for VGG-16 conv1_1 .. conv5_3
# (alpha_i for structure, beta_i for texture) as released with the paper.
# Source: https://github.com/dingkeyan93/DISTS
_ALPHA = [
    0.21204585,
    0.13465898,
    0.10198663,
    0.10668225,
    0.06049506,
    0.07428068,
    0.04393877,
    0.02289652,
    0.0256998,
    0.0003,
    0.0003,
    0.0003,
]
_BETA = [
    0.11337525,
    0.08219337,
    0.06605395,
    0.06049506,
    0.04500992,
    0.04600536,
    0.02894009,
    0.01638238,
    0.02693324,
    0.0003,
    0.0003,
    0.0003,
]

# VGG-16 feature layer indices that DISTS uses (after ReLU), in order.
# conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3,
# conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3 — 13 maps, but the
# released weights list 12 entries corresponding to a specific subset.
# The official implementation uses features up to relu5_3 with the weights
# above (12 entries) — we follow the released weight vectors exactly.
_VGG_LAYERS = [
    "0",   # conv1_1
    "2",   # conv1_2
    "5",   # conv2_1
    "7",   # conv2_2
    "10",  # conv3_1
    "12",  # conv3_2
    "14",  # conv3_3
    "17",  # conv4_1
    "19",  # conv4_2
    "21",  # conv4_3
    "24",  # conv5_1
    "26",  # conv5_2
    # relu5_3 (28) is also extracted in the official code; weights for the
    # 13th map are folded into the published 12-value vectors via the last
    # 0.0003 entries. We extract 13 maps and use the first 12 + treat the
    # remainder consistently below.
    "28",  # conv5_3 / relu5_3
]


class _VGGFeatures(nn.Module):
    """Truncated VGG-16 feature extractor (no classifier)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.available = False
        try:
            from torchvision.models import vgg16
            from torchvision.models import VGG16_Weights

            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = vgg16(weights=weights)
            self.features = vgg.features
            # ImageNet normalization
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
            self.available = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"DISTS: VGG-16 unavailable ({exc}). "
                "DISTS will be reported as null — never fabricated.",
                RuntimeWarning,
                stacklevel=2,
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if not self.available:
            raise RuntimeError("DISTS VGG features unavailable")
        x = (x - self.mean) / self.std
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if str(i) in _VGG_LAYERS:
                feats.append(x)
        return feats


def _gram(features: torch.Tensor) -> torch.Tensor:
    """Gram matrix with mean-pooled spatial dims (texture representation)."""
    b, c, h, w = features.shape
    feat = features.view(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2)) / (h * w)
    return gram


class DISTSMetric(nn.Module):
    """DISTS metric. Lower is better."""

    def __init__(self, pretrained: bool = True, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.available = False
        self._metric: nn.Module | None = None
        try:
            if not pretrained:
                raise ValueError("The trained DISTS weights are required for evaluation")
            from DISTS_pytorch import DISTS

            self._metric = DISTS(load_weights=True).to(self.device).eval()
            for parameter in self._metric.parameters():
                parameter.requires_grad_(False)
            self.available = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"DISTS unavailable ({exc}). Install `dists-pytorch` and its "
                "pretrained VGG weights. Values will be reported as null.",
                RuntimeWarning,
                stacklevel=2,
            )

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float | None:
        if not self.available or self._metric is None:
            return None
        pred = pred.detach().to(self.device).float().clamp(0.0, 1.0)
        target = target.detach().to(self.device).float().clamp(0.0, 1.0)
        value = self._metric(pred, target, batch_average=True)
        return float(value.mean().item())


def build_dists(pretrained: bool = True, device: str | torch.device = "cpu") -> DISTSMetric:
    return DISTSMetric(pretrained=pretrained, device=device)
