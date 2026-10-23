"""LPUDC: Laplacian Pyramid Neural Network for UDC Image Restoration.

Reconstructs the architecture described in:
    Ding et al., "LPUDC: A Laplacian pyramid neural network for restoring
    images from under-display cameras", Neurocomputing 655 (2025) 131442.

Overall data flow (PAPER FACT, Fig. 2 / Sec. 3):

    INPUT DEGRADED IMAGE
            |
            v
    LAPLACIAN PYRAMID DECOMPOSITION
            |
            +------------------------------+
            |                              |
            v                              v
    LOW-FREQUENCY BRANCH             HIGH-FREQUENCY BRANCH
            |                              ^
            |                              |
            v                              |
    LOW-FREQUENCY RESTORATION         DGF GUIDANCE
            |                              ^
            |                              |
            +----------> DGF --------------+
                           |
                           v
                    HIGH-FREQUENCY
                      ENHANCEMENT
                           |
                           v
                 PYRAMID RECONSTRUCTION
                           |
                           v
                     RESTORED IMAGE

The low-frequency branch is applied at the coarsest Laplacian level.
Each high-frequency level is enhanced under DGF guidance from the level
below, then the image is reconstructed from the predicted pyramid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dgf import DeepGuidedFilter
from .high_frequency import HighFrequencyBranch
from .laplacian_pyramid import LaplacianPyramid
from .low_frequency import LowFrequencyBranch
from .reconstruction import PyramidReconstructor


@dataclass
class LPUDCConfig:
    """Configuration for the LPUDC model.

    Defaults follow the paper's primary configuration (3 pyramid levels)
    with engineering assumptions documented in docs/PAPER_SPECIFICATION.md
    for all unspecified architecture widths.
    """

    pyramid_levels: int = 3
    in_channels: int = 3

    # Low-frequency branch
    lf_base_channels: int = 32
    lf_encoder_channels: tuple[int, ...] = (32, 64, 128)
    lf_stem_drb_count: int = 2
    lf_stem_dilations: tuple[int, ...] = (1, 2)
    lf_guidance_channels: int = 64

    # Deep guided filtering
    dgf_kernel_size: int = 3
    dgf_eps: float = 0.001
    dgf_coeff_channels: int = 32

    # High-frequency branch
    hf_base_channels: int = 32
    hf_unshuffle_factor: int = 2
    hf_pre_drb_count: int = 2
    hf_pre_drb_dilations: tuple[int, ...] = (1, 2)
    hf_hrnet_drb_count: int = 4
    hf_hrnet_drb_dilations: tuple[int, ...] = (1, 2, 4)
    confidence_gate: bool = False

    # Ablation switches (paper Tables 3–4)
    # 0 = no multi-scale interaction
    # 1 = multi-scale interaction without guided filtering (direct concat)
    # 2 = multi-scale interaction + guided filtering (full LPUDC)
    fusion_mode: int = 2

    # Loss
    pyramid_weight: float = 0.5
    use_pyramid_loss: bool = True

    # Debug / memory
    debug_shapes: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LPUDCConfig":
        d = dict(d)
        for k in (
            "lf_encoder_channels",
            "lf_stem_dilations",
            "hf_pre_drb_dilations",
            "hf_hrnet_drb_dilations",
        ):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class LPUDC(nn.Module):
    """Full LPUDC network.

    Returns a dict with:
        restored:        predicted full-resolution image
        pred_levels:     predicted Laplacian pyramid levels (for L_pyr)
        lf_restored:     restored coarsest low-frequency image
        guidance_maps:   DGF intermediate guidance outputs (for visualization)
        degraded_levels: Laplacian pyramid of the degraded input
    """

    def __init__(self, config: LPUDCConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = LPUDCConfig(**kwargs)
        elif kwargs:
            # Allow overriding individual fields.
            base = config.to_dict()
            base.update(kwargs)
            config = LPUDCConfig.from_dict(base)
        self.config = config

        L = config.pyramid_levels
        if L < 1:
            raise ValueError("pyramid_levels must be >= 1")
        if config.confidence_gate and config.fusion_mode != 2:
            raise ValueError("confidence_gate requires fusion_mode=2 (DGF)")

        self.pyramid = LaplacianPyramid(L)
        self.reconstructor = PyramidReconstructor(L)

        # --- Low-frequency branch (applied at coarsest level) ---
        self.low_frequency = LowFrequencyBranch(
            in_channels=config.in_channels,
            base_channels=config.lf_base_channels,
            encoder_channels=tuple(config.lf_encoder_channels),
            stem_drb_dilations=tuple(config.lf_stem_dilations),
            stem_drb_count=config.lf_stem_drb_count,
            guidance_channels=config.lf_guidance_channels,
        )

        # --- DGF modules: one per HF level (coarse -> fine) ---
        # Level index i (0 = finest residual) is guided from level i+1.
        # The coarsest level (index L) has no DGF.
        self.dgf_modules = nn.ModuleList()
        # For ablation mode 1 (multi-scale without GF): project upsampled
        # guidance to hf_base_channels so it can enter the HF branch.
        self.concat_projections = nn.ModuleList()
        for i in range(L):
            guid_ch = config.lf_guidance_channels
            if config.fusion_mode == 2:
                self.dgf_modules.append(
                    DeepGuidedFilter(
                        # Paper roles: G is the LF network input and I its output.
                        guidance_channels=config.in_channels,
                        input_channels=config.in_channels,
                        coefficient_channels=config.dgf_coeff_channels,
                        kernel_size=config.dgf_kernel_size,
                        eps=config.dgf_eps,
                        # Output must match HF-branch guidance width.
                        output_channels=config.hf_base_channels,
                    )
                )
            if config.fusion_mode > 0:
                self.concat_projections.append(
                    nn.Conv2d(guid_ch, config.hf_base_channels, kernel_size=1, bias=True)
                )

        # --- High-frequency branches (one per pyramid residual level) ---
        # After inverse pixel shuffle, guided features must match the
        # unshuffled channel layout. We feed DGF output (hf_base_channels)
        # as additional guidance channels at the *pre-unshuffle* resolution,
        # so HighFrequencyBranch expects guidance_channels = hf_base_channels
        # and guidance is applied after unshuffle via concat inside the HF
        # branch (guided_feat is spatially aligned to unshuffled map).
        self.high_frequency = nn.ModuleList()
        for _ in range(L):
            self.high_frequency.append(
                HighFrequencyBranch(
                    in_channels=config.in_channels,
                    base_channels=config.hf_base_channels,
                    unshuffle_factor=config.hf_unshuffle_factor,
                    pre_drb_count=config.hf_pre_drb_count,
                    pre_drb_dilations=tuple(config.hf_pre_drb_dilations),
                    hrnet_drb_count=config.hf_hrnet_drb_count,
                    hrnet_drb_dilations=tuple(config.hf_hrnet_drb_dilations),
                    guidance_channels=config.hf_base_channels
                    if config.fusion_mode > 0
                    else 0,
                    confidence_gate=config.confidence_gate,
                )
            )

        # Projection that maps LF-branch guidance to a channel width usable
        # at each HF level (DGF guidance input).
        self.guidance_proj = nn.Conv2d(
            config.lf_guidance_channels,
            config.lf_guidance_channels,
            kernel_size=1,
            bias=True,
        )
        # For multi-level propagation, project guidance between levels.
        self.level_guidance_proj = nn.ModuleList(
            [
                nn.Conv2d(
                    config.lf_guidance_channels,
                    config.lf_guidance_channels,
                    kernel_size=1,
                    bias=True,
                )
                for _ in range(max(0, L - 1))
            ]
        )

    def forward(
        self, x: torch.Tensor, return_extras: bool = True
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")

        cfg = self.config
        residuals, coarse, levels = self.pyramid(x)
        # residuals: finest first, length L; coarse = levels[-1]

        # ---- Low-frequency branch at the coarsest level (PAPER FACT) ----
        lf_restored, lf_guidance = self.low_frequency(coarse)
        guidance = self.guidance_proj(lf_guidance)

        guidance_maps: list[torch.Tensor] = []
        pred_residuals: list[torch.Tensor | None] = [None] * cfg.pyramid_levels

        # Process HF levels from coarsest residual (index L-1) to finest (0)
        # so DGF guidance can propagate upward (coarse-to-fine, PAPER FACT).
        for level_idx in range(cfg.pyramid_levels - 1, -1, -1):
            res = residuals[level_idx]  # N x C x H_l x W_l
            hf_net = self.high_frequency[level_idx]

            if cfg.fusion_mode == 0:
                # Ablation A: no multi-scale interaction.
                enhanced = hf_net(res, guided_feat=None)
            elif cfg.fusion_mode == 2:
                # Full LPUDC: DGF guides HF enhancement at this level.
                guided_feat = self.dgf_modules[level_idx](
                    coarse, lf_restored, target_size=res.shape[-2:]
                )
                guidance_maps.append(guided_feat)
                enhanced = hf_net(res, guided_feat=guided_feat)
            else:
                # Ablation B: multi-scale interaction WITHOUT guided filtering
                # (direct feature concatenation / projection).
                g_up = F.interpolate(
                    guidance,
                    size=res.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                guided = self.concat_projections[level_idx](g_up)
                guidance_maps.append(guided)
                enhanced = hf_net(res, guided_feat=guided)

            pred_residuals[level_idx] = enhanced

            # Propagate guidance to the next finer level.
            if level_idx > 0 and cfg.fusion_mode > 0:
                guidance = self.level_guidance_proj[level_idx - 1](guidance)

        assert all(r is not None for r in pred_residuals)
        pred_res_list = [r for r in pred_residuals if r is not None]

        restored, pred_levels = self.reconstructor(
            pred_res_list,
            lf_restored,
        )

        if cfg.debug_shapes:
            PyramidReconstructor.assert_shapes(
                restored, pred_levels, x.shape[-2:], cfg.pyramid_levels
            )

        out: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "restored": restored,
            "pred_levels": pred_levels,
            "lf_restored": lf_restored,
            "degraded_levels": levels,
        }
        if return_extras:
            out["guidance_maps"] = guidance_maps
        return out


def build_lpudc(config_dict: dict[str, Any] | None = None) -> LPUDC:
    """Build LPUDC from a plain dict (e.g. loaded YAML `model` section)."""
    if config_dict is None:
        return LPUDC(LPUDCConfig())
    cfg = LPUDCConfig.from_dict(config_dict)
    return LPUDC(cfg)
