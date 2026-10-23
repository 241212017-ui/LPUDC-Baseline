"""Pyramid reconstruction utilities for LPUDC.

PAPER FACT: The network restores different pyramid levels and reconstructs
the final high-resolution image progressively. The pyramid reconstruction
loss (Eq. 7) compares the enhanced image Laplacian pyramid against the
ground-truth Laplacian pyramid.

IMPLEMENTATION ASSUMPTION: The model predicts an HF enhancement at each
Laplacian residual level and an LF restoration at the coarsest level. The
full image is reconstructed by the inverse Laplacian transform. For the
pyramid loss we also expose the predicted pyramid (enhanced residuals +
restored coarse) so each level can be supervised directly.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .laplacian_pyramid import LaplacianPyramid, laplacian_upsample


class PyramidReconstructor(nn.Module):
    """Reconstruct a full-resolution image from predicted pyramid levels.

    Args:
        num_levels: number of Laplacian levels L.
    """

    def __init__(self, num_levels: int = 3) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.pyramid = LaplacianPyramid(num_levels)

    def forward(
        self,
        predicted_residuals: Sequence[torch.Tensor],
        predicted_coarse: torch.Tensor,
        residual_base: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Reconstruct the restored image and the predicted pyramid.

        Args:
            predicted_residuals: enhanced HF residuals, finest first, length L.
            predicted_coarse: restored coarsest LF image.
            residual_base: optional original degraded residuals. If provided,
                each predicted residual is treated as an *enhancement delta*
                and added to the base residual before reconstruction
                (matching the HF branch's residual-fusion design).

        Returns:
            restored: N x C x H x W
            predicted_levels: [res_0, ..., res_{L-1}, coarse] at their
                              native resolutions (for L_pyr supervision).
        """
        if len(predicted_residuals) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} predicted residuals, "
                f"got {len(predicted_residuals)}"
            )

        levels: list[torch.Tensor] = []
        fused_residuals: list[torch.Tensor] = []
        for i, res in enumerate(predicted_residuals):
            if residual_base is not None:
                base = residual_base[i]
                if base.shape != res.shape:
                    base = F.interpolate(base, size=res.shape[-2:], mode="bilinear", align_corners=False)
                fused = base + res
            else:
                fused = res
            fused_residuals.append(fused)
            levels.append(fused)

        restored = self.pyramid.reconstruct(fused_residuals, predicted_coarse)
        levels.append(predicted_coarse)
        return restored, levels

    @staticmethod
    def assert_shapes(
        restored: torch.Tensor,
        levels: Sequence[torch.Tensor],
        target_hw: tuple[int, int],
        num_levels: int,
    ) -> None:
        """Debug assertion: restored image matches target resolution and the
        pyramid level resolutions decrease by 2x per level."""
        if restored.shape[-2:] != target_hw:
            raise AssertionError(
                f"Restored spatial size {tuple(restored.shape[-2:])} != target {target_hw}"
            )
        if len(levels) != num_levels + 1:
            raise AssertionError(
                f"Expected {num_levels + 1} pyramid levels, got {len(levels)}"
            )
        h, w = target_hw
        for i, lv in enumerate(levels):
            eh = max(1, h // (2 ** i))
            ew = max(1, w // (2 ** i))
            # Allow off-by-one due to odd padding; require closeness.
            if abs(lv.shape[-2] - eh) > 1 or abs(lv.shape[-1] - ew) > 1:
                raise AssertionError(
                    f"Level {i} shape {tuple(lv.shape[-2:])} inconsistent "
                    f"with expected ~({eh}, {ew})"
                )
