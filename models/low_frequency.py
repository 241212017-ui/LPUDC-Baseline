"""Low-frequency restoration branch of LPUDC.

PAPER FACT (Sec. 3.2):
    The low-frequency image restoration network is primarily responsible for
    global reconstruction, focusing on correcting brightness attenuation and
    color distortion caused by the display layer. The network consists of:
      1. a feature extraction module: 3x3 convolution + LeakyReLU
      2. a Dilated Residual Block (DRB) to capture multi-scale contextual info
      3. an encoder–decoder structure for feature compression and recovery
      4. a symmetric reconstruction module that restores global color and
         illumination information in the low-frequency image.

IMPLEMENTATION ASSUMPTION:
    Exact channel widths, number of DRBs, and encoder/decoder depths are not
    specified. We use a compact multi-level encoder–decoder with configurable
    widths (default [32, 64, 128]) and DRBs in the stem, chosen to approach
    the paper's reported ~4.0M total parameters when combined with the HF branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBlock,
    Downsample,
    ResidualConvStack,
    Upsample,
)


class EncoderStage(nn.Module):
    """One encoder stage: downsample -> channel projection -> residual stack."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        drb_dilations: tuple[int, ...] = (1, 2),
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.down = Downsample(in_ch, in_ch, activation, negative_slope)
        self.proj = ConvBlock(in_ch, out_ch, activation, negative_slope)
        self.drb = ResidualConvStack(
            channels=out_ch,
            num_blocks=1,
            dilations=drb_dilations,
            activation=activation,
            negative_slope=negative_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drb(self.proj(self.down(x)))


class LowFrequencyBranch(nn.Module):
    """Encoder–decoder low-frequency restoration network.

    Inputs:
        low_freq_image: degraded low-frequency component, N x C x H x W
                        (typically the coarsest Laplacian level)
    Outputs:
        restored_low_freq: N x C x H x W  (restored low-frequency image)
        guidance_feat:     N x G x H x W  (feature map used as DGF guidance;
                           same spatial size as the input low-frequency image)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        encoder_channels: tuple[int, ...] = (32, 64, 128),
        stem_drb_dilations: tuple[int, ...] = (1, 2),
        stem_drb_count: int = 2,
        encoder_drb_dilations: tuple[int, ...] = (1, 2),
        decoder_drb_dilations: tuple[int, ...] = (1, 2),
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
        guidance_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channels = tuple(encoder_channels)
        enc_ch = self.encoder_channels
        self.stem_channels = base_channels
        g_ch = guidance_channels if guidance_channels is not None else enc_ch[-1]
        self.guidance_channels = g_ch

        # --- Feature extraction: 3x3 conv + LeakyReLU (PAPER FACT) ---
        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
        )

        # --- DRB stem (PAPER FACT) ---
        self.stem = ResidualConvStack(
            channels=base_channels,
            num_blocks=stem_drb_count,
            dilations=stem_drb_dilations,
            activation=activation,
            negative_slope=negative_slope,
            global_residual=True,
        )

        # --- Multi-scale encoder ---
        # Each stage halves spatial resolution.
        # skip channels collected: [base_channels, enc_ch[0], enc_ch[1], ...]
        self.encoder_stages = nn.ModuleList()
        prev = base_channels
        for ch in enc_ch:
            self.encoder_stages.append(
                EncoderStage(
                    prev,
                    ch,
                    drb_dilations=encoder_drb_dilations,
                    activation=activation,
                    negative_slope=negative_slope,
                )
            )
            prev = ch
        bottleneck_ch = enc_ch[-1]

        # --- Bottleneck ---
        self.bottleneck = ResidualConvStack(
            channels=bottleneck_ch,
            num_blocks=2,
            dilations=(1, 2, 4),
            activation=activation,
            negative_slope=negative_slope,
        )

        # --- Symmetric decoder ---
        # skips = [stem (base_ch, full res), enc0 (enc_ch[0], /2), enc1, ...]
        # Decoder walks these skips from coarse to fine.
        skip_channels = [base_channels] + list(enc_ch)
        # Decoder stage di walks skip_channels from the coarse end to full res.
        self.decoder_ups = nn.ModuleList()
        self.decoder_fuses = nn.ModuleList()
        self.decoder_drbs = nn.ModuleList()
        n_dec = len(skip_channels)  # one per skip, ending at stem resolution
        for di in range(n_dec):
            skip_idx = len(skip_channels) - 1 - di
            skip_ch = skip_channels[skip_idx]
            if di == 0:
                cur_ch = bottleneck_ch
            else:
                cur_ch = skip_channels[len(skip_channels) - 1 - (di - 1)]
            out_ch = skip_ch
            # Upsample from current width to the skip's channel width.
            self.decoder_ups.append(Upsample(cur_ch, out_ch, activation, negative_slope))
            self.decoder_fuses.append(
                ConvBlock(out_ch + skip_ch, out_ch, activation, negative_slope)
            )
            self.decoder_drbs.append(
                ResidualConvStack(
                    channels=out_ch,
                    num_blocks=1,
                    dilations=decoder_drb_dilations,
                    activation=activation,
                    negative_slope=negative_slope,
                )
            )

        # --- Symmetric reconstruction module ---
        self.recon_head = nn.Sequential(
            ConvBlock(base_channels, base_channels, activation, negative_slope),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1, bias=True),
        )
        self.guidance_head = nn.Conv2d(base_channels, g_ch, kernel_size=1, bias=True)

    def forward(self, low_freq_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = low_freq_image
        size0 = x0.shape[-2:]

        h = self.shallow(x0)
        h = self.stem(h)

        # Encoder: skip[0] = stem at full input resolution
        skips: list[torch.Tensor] = [h]
        for stage in self.encoder_stages:
            h = stage(h)
            skips.append(h)

        h = self.bottleneck(h)

        # Decoder: coarse -> fine
        for di, skip in enumerate(reversed(skips)):
            h = self.decoder_ups[di](h, size=skip.shape[-2:])
            h = torch.cat([h, skip], dim=1)
            h = self.decoder_fuses[di](h)
            h = self.decoder_drbs[di](h)

        # Final size safety (odd input sizes).
        if h.shape[-2:] != size0:
            h = F.interpolate(h, size=size0, mode="bilinear", align_corners=False)

        residual = self.recon_head(h)
        restored = low_freq_image + residual
        guidance = self.guidance_head(h)
        return restored, guidance
