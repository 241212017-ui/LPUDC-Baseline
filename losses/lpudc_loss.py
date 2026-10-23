"""Loss functions for LPUDC.

PAPER FACT (Sec. 3.5, Eqs. 6–8):
    Content loss:        L_con = L1(Y, X)          (Eq. 6, written as a sum
                                                    over pixels in the paper)
    Pyramid loss:        L_pyr = sum_{i=0}^{L} || Î_i − I_i ||_1   (Eq. 7)
                         where Î_i is the i-th level of the enhanced image
                         Laplacian pyramid and I_i is the i-th level of the
                         ground-truth Laplacian pyramid.
    Total loss:          L = L_con + λ * L_pyr     (Eq. 8)
    Weighting factor:    λ = 0.5

IMPLEMENTATION ASSUMPTION:
    The paper writes L_con as a raw sum over N pixels. With full-resolution
    1024x2048 images a raw sum would be enormous and scale-fragile across
    resolutions. We use mean L1 (PyTorch L1Loss(reduction='mean')) for both
    terms, which is the standard practical equivalent. λ = 0.5 exactly as
    stated in the paper.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.laplacian_pyramid import LaplacianPyramid


class LPUDCLoss(nn.Module):
    """Joint content + pyramid reconstruction loss.

    Returns a dict with keys:
        total_loss, content_loss, pyramid_loss
    """

    def __init__(
        self,
        pyramid_weight: float = 0.5,
        use_pyramid_loss: bool = True,
        num_levels: int = 3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")
        self.pyramid_weight = float(pyramid_weight)
        self.use_pyramid_loss = bool(use_pyramid_loss)
        self.num_levels = int(num_levels)
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)
        # GT pyramid is computed on the fly (no params).
        self._gt_pyramid = LaplacianPyramid(self.num_levels)

    def content_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"content_loss shape mismatch: pred {tuple(pred.shape)} "
                f"vs target {tuple(target.shape)}"
            )
        return self.l1(pred, target)

    def pyramid_loss(
        self,
        pred_levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """L1 between each predicted pyramid level and the GT pyramid level.

        Eq. 7: L_pyr = sum_i || Î_i − I_i ||_1
        If `pred_levels` includes the coarsest level, it is matched against
        the GT coarsest level. Levels are matched index-wise (0 = finest).
        """
        _, _, gt_levels = self._gt_pyramid(target)
        if len(pred_levels) != len(gt_levels):
            # If the model returned L residuals + coarse, lengths match L+1.
            raise ValueError(
                f"pyramid_loss: predicted {len(pred_levels)} levels, "
                f"GT has {len(gt_levels)} levels"
            )
        total = gt_levels[0].new_zeros(())
        for i, (p, g) in enumerate(zip(pred_levels, gt_levels)):
            if p.shape != g.shape:
                p = F.interpolate(p, size=g.shape[-2:], mode="bilinear", align_corners=False)
            total = total + self.l1(p, g)
        return total

    def forward(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        pred_levels: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        content = self.content_loss(pred_image, target_image)
        if self.use_pyramid_loss:
            if pred_levels is None:
                # Fall back to decomposing the predicted image itself.
                _, _, pred_levels_t = self._gt_pyramid(pred_image)
            else:
                pred_levels_t = pred_levels
            pyr = self.pyramid_loss(pred_levels_t, target_image)
        else:
            pyr = pred_image.new_zeros(())
        total = content + self.pyramid_weight * pyr
        return {
            "total_loss": total,
            "content_loss": content,
            "pyramid_loss": pyr,
        }


def build_loss(cfg: dict[str, Any] | None = None) -> LPUDCLoss:
    """Build loss from a config dict (`loss` section)."""
    cfg = cfg or {}
    return LPUDCLoss(
        pyramid_weight=float(cfg.get("pyramid_weight", cfg.get("lambda_pyr", 0.5))),
        use_pyramid_loss=bool(cfg.get("use_pyramid_loss", cfg.get("pyramid_loss", True))),
        num_levels=int(cfg.get("pyramid_levels", 3)),
        reduction=str(cfg.get("reduction", "mean")),
    )
