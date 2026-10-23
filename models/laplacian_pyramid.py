"""Laplacian pyramid decomposition and reconstruction.

PAPER FACT (Sec. 2.2):
    Given I0, first generate a low-resolution version I1 by applying a
    fixed-kernel weighted average to neighboring pixels. Then an upsampled
    version Î0 is reconstructed from I1, and the high-frequency residual is
    h0 = I0 − Î0. This process is repeated iteratively on I1 to construct a
    pyramid of high-frequency residuals and a final low-frequency image.

IMPLEMENTATION ASSUMPTION:
    The paper does not specify the exact down/upsample kernels. We use the
    classical Burt–Adelson Laplacian pyramid construction:
      - downsample: 5x5 binomial kernel, stride-2, symmetric padding
      - upsample:   bilinear to target size + same 5x5 binomial blur
    This is differentiable, invertible in the limit, and matches the paper's
    description of a "fixed-kernel weighted average".
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binomial_kernel_1d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    return k / k.sum()


def _binomial_kernel_2d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k1 = _binomial_kernel_1d(device, dtype)
    k2 = torch.outer(k1, k1)
    return k2  # 5x5, sums to 1


def laplacian_downsample(x: torch.Tensor) -> torch.Tensor:
    """Downsample by 2x using a fixed 5x5 binomial kernel.

    Odd spatial sizes are handled by reflective padding then cropping the
    upsampled result back to 2*floor(h/2), 2*floor(w/2) via the residual path.
    We simply pad to the next even size before filtering.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {tuple(x.shape)}")
    h, w = x.shape[-2:]
    ph = (2 - h % 2) % 2
    pw = (2 - w % 2) % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    k = _binomial_kernel_2d(x.device, x.dtype).view(1, 1, 5, 5).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(x, k, padding=2, groups=x.shape[1])
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x


def laplacian_upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Upsample to `size` with bilinear interpolation + binomial blur.

    Matches the classical LP expand step closely enough for restoration
    training while remaining differentiable and shape-exact.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D NCHW tensor, got shape {tuple(x.shape)}")
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    k = _binomial_kernel_2d(x.device, x.dtype).view(1, 1, 5, 5).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(x, k, padding=2, groups=x.shape[1])
    return x


class LaplacianPyramid(nn.Module):
    """Explicit Laplacian pyramid decomposition.

    Forward returns:
        residuals: list of high-frequency residuals, finest first.
                   length = num_levels
        coarse:    lowest-resolution image, shape N x C x H/2^L x W/2^L
        levels:    full pyramid for loss purposes:
                   [residual_0, residual_1, ..., residual_{L-1}, coarse]
                   level i has resolution (H/2^i, W/2^i)
    """

    def __init__(self, num_levels: int = 3) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        self.num_levels = int(num_levels)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        residuals: list[torch.Tensor] = []
        current = x
        for _ in range(self.num_levels):
            down = laplacian_downsample(current)
            up = laplacian_upsample(down, size=current.shape[-2:])
            # Match channels/spatial exactly.
            if up.shape != current.shape:
                up = F.interpolate(up, size=current.shape[-2:], mode="bilinear", align_corners=False)
            residual = current - up
            residuals.append(residual)
            current = down
        levels = residuals + [current]
        return residuals, current, levels

    def reconstruct(self, residuals: Sequence[torch.Tensor], coarse: torch.Tensor) -> torch.Tensor:
        """Reconstruct full-resolution image from residuals (finest first) + coarse."""
        if len(residuals) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} residuals, got {len(residuals)}"
            )
        current = coarse
        # Reconstruct from coarsest to finest.
        for res in reversed(list(residuals)):
            target_hw = res.shape[-2:]
            up = laplacian_upsample(current, size=target_hw)
            if up.shape != res.shape:
                up = F.interpolate(up, size=res.shape[-2:], mode="bilinear", align_corners=False)
            current = up + res
        return current


class PyramidDecompose(nn.Module):
    """Thin wrapper so the pyramid can be used as a submodule in models."""

    def __init__(self, num_levels: int = 3) -> None:
        super().__init__()
        self.pyramid = LaplacianPyramid(num_levels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        _, _, levels = self.pyramid(x)
        return levels
