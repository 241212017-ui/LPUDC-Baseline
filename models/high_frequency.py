"""High-frequency enhancement branch of LPUDC.

PAPER FACT (Sec. 3.4):
    1. Inverse pixel shuffle — compresses spatial information into the
       channel dimension; reduces computational overhead while retaining
       local features.
    2. 3x3 convolution — adjusts the channel number to extract basic features.
    3. Residual blocks with dilated convolutions (DRB) — expand the receptive
       field and enhance perception of local and global information.
    4. High-Resolution Network (HRNet) — multiple DRBs connected in series,
       continuously extracting and reinforcing high-frequency details and
       focusing on correcting diffraction artifact areas. Residual connections
       are used throughout.
    5. Final 3x3 convolution — projects enhanced features back to the original
       channel number to fuse with the high-frequency components of the
       Laplacian pyramid.

IMPLEMENTATION ASSUMPTION:
    The paper does not specify the inverse-pixel-shuffle factor, channel
    widths, or the exact number of DRBs in the HRNet stage. We use:
      - pixel-unshuffle factor = 2 (default)
      - base channel width 32 after unshuffle
      - 2 DRBs before the HRNet trunk
      - HRNet trunk = 4 serial DRBs with a global residual (NOT the full
        multi-branch HRNet of Wang et al., which the paper does not describe
        and which would blow the ~4.0M parameter budget)
      - fusion with the Laplacian HF residual is a residual add
        (H_restored = H_Laplacian + projected_features)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBlock,
    ConvNormAct,
    DilatedResidualBlock,
    ResidualConvStack,
)


class InversePixelShuffle(nn.Module):
    """PixelUnshuffle: (N, C, H, W) -> (N, C*r^2, H/r, W/r)."""

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("factor must be >= 1")
        self.factor = int(factor)
        self.op = nn.PixelUnshuffle(self.factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        r = self.factor
        if h % r != 0 or w % r != 0:
            # Pad to the next multiple of r so unshuffle is exact.
            ph = (r - h % r) % r
            pw = (r - w % r) % r
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return self.op(x)


class HighFrequencyBranch(nn.Module):
    """High-frequency enhancement network.

    Args:
        in_channels: channels of the Laplacian high-frequency residual
                     (typically 3 for RGB images).
        base_channels: channel width after the first 3x3 conv following
                       inverse pixel shuffle.
        unshuffle_factor: inverse pixel shuffle factor.
        pre_drb_count / pre_drb_dilations: DRBs between the 3x3 conv and HRNet.
        hrnet_drb_count / hrnet_drb_dilations: serial DRBs inside the
                       "HRNet" high-resolution trunk.
        guidance_channels: channels of DGF-guided features concatenated /
                       added as extra input. If 0, ignored.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        unshuffle_factor: int = 2,
        pre_drb_count: int = 2,
        pre_drb_dilations: tuple[int, ...] = (1, 2),
        hrnet_drb_count: int = 4,
        hrnet_drb_dilations: tuple[int, ...] = (1, 2, 4),
        guidance_channels: int = 0,
        confidence_gate: bool = False,
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.unshuffle_factor = unshuffle_factor
        self.base_channels = base_channels
        unch = in_channels * unshuffle_factor * unshuffle_factor
        self.guidance_channels = guidance_channels
        self.confidence_gate = confidence_gate
        if confidence_gate and guidance_channels == 0:
            raise ValueError("confidence_gate requires guided features")

        # 1) Inverse pixel shuffle (PAPER FACT)
        self.pixel_unshuffle = InversePixelShuffle(unshuffle_factor)

        # 2) 3x3 conv to adjust channels (PAPER FACT)
        self.stem_conv = nn.Conv2d(
            unch + guidance_channels, base_channels, kernel_size=3, padding=1, bias=True
        )
        self.stem_act = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)

        # 3) DRBs to expand receptive field (PAPER FACT)
        self.pre_drbs = ResidualConvStack(
            channels=base_channels,
            num_blocks=pre_drb_count,
            dilations=pre_drb_dilations,
            activation=activation,
            negative_slope=negative_slope,
        )

        # 4) HRNet-style high-resolution trunk: serial DRBs + global residual
        #    (PAPER FACT as described in text; see IMPLEMENTATION ASSUMPTION)
        self.hrnet = ResidualConvStack(
            channels=base_channels,
            num_blocks=hrnet_drb_count,
            dilations=hrnet_drb_dilations,
            activation=activation,
            negative_slope=negative_slope,
            global_residual=True,
        )

        # Also keep an explicit DRB series alias for clarity / ablations.
        self.hrb = nn.ModuleList(
            [
                DilatedResidualBlock(
                    base_channels,
                    dilations=(1, 2),
                    activation=activation,
                    negative_slope=negative_slope,
                )
                for _ in range(max(0, hrnet_drb_count - 1))
            ]
        )

        # 5) Final 3x3 projection back to original channel count (PAPER FACT)
        self.out_conv = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1, bias=True
        )

        # Optional learnable residual scale for the fusion.
        self.fusion_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_conv = (
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
            if confidence_gate
            else None
        )

    def forward(
        self,
        hf_residual: torch.Tensor,
        guided_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Enhance a Laplacian high-frequency residual.

        Args:
            hf_residual: N x C x H x W  (Laplacian HF component)
            guided_feat: optional DGF output, same H, W; if provided and
                         guidance_channels > 0, it is used as additional
                         input channels after inverse pixel shuffle.
        Returns:
            enhanced HF residual, N x C x H x W
        """
        x = self.pixel_unshuffle(hf_residual)  # N, C*r^2, H/r, W/r

        g = None
        if guided_feat is not None and self.guidance_channels > 0:
            g = guided_feat
            if g.shape[-2:] != x.shape[-2:]:
                g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if g.shape[1] != self.guidance_channels:
                # Adapt channel count on the fly (should not happen in normal use).
                if g.shape[1] > self.guidance_channels:
                    g = g[:, : self.guidance_channels]
                else:
                    pad = self.guidance_channels - g.shape[1]
                    g = F.pad(g, (0, 0, 0, 0, 0, pad))

        if self.confidence_gate:
            if g is None:
                g = x.new_zeros(
                    x.shape[0], self.guidance_channels, x.shape[2], x.shape[3]
                )
            hf_input = torch.cat([x, torch.zeros_like(g)], dim=1)
            dgf_input = torch.cat([torch.zeros_like(x), g], dim=1)
            hf_features = self.stem_conv(hf_input)
            dgf_features = self.stem_conv(dgf_input)
            gate = torch.sigmoid(
                self.gate_conv(torch.cat([dgf_features, hf_features], dim=1))
            )
            fused_features = gate * dgf_features + (1.0 - gate) * hf_features
        else:
            if self.guidance_channels > 0:
                if g is None:
                    g = x.new_zeros(
                        x.shape[0], self.guidance_channels, x.shape[2], x.shape[3]
                    )
                x = torch.cat([x, g], dim=1)
            fused_features = self.stem_conv(x)

        h = self.stem_act(fused_features)
        h = self.pre_drbs(h)
        h = self.hrnet(h)
        for blk in self.hrb:
            h = blk(h)

        # Upsample back to the residual's spatial resolution.
        if h.shape[-2:] != hf_residual.shape[-2:]:
            h = F.interpolate(
                h, size=hf_residual.shape[-2:], mode="bilinear", align_corners=False
            )
        proj = self.out_conv(h)
        # Residual fusion with the Laplacian HF component (PAPER FACT).
        return hf_residual + self.fusion_scale * proj
