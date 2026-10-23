"""Reusable building blocks for LPUDC.

PAPER FACT: The low-frequency branch uses a 3x3 convolution + LeakyReLU and a
Dilated Residual Block (DRB) to capture multi-scale contextual information.
The high-frequency branch uses DRBs and residual connections.

IMPLEMENTATION ASSUMPTION: Exact DRB internals (number of convs, dilation
schedule, activation placement) are NOT specified in the paper. We implement
a standard dilated residual block and document the choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str = "leaky_relu", slope: float = 0.01) -> nn.Module:
    """Create an activation module by name.

    PAPER FACT: Low-frequency branch uses LeakyReLU after the 3x3 conv.
    IMPLEMENTATION ASSUMPTION: Negative slope 0.01 (PyTorch default); the
    paper does not state the slope.
    """
    name = name.lower()
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=slope, inplace=False)
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=False)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class ConvNormAct(nn.Module):
    """Conv -> optional normalization -> activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
        norm: str = "none",
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=(norm == "none"),
        )
        if norm == "batch":
            self.norm: nn.Module = nn.BatchNorm2d(out_ch)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.act = make_activation(activation, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DilatedResidualBlock(nn.Module):
    """Dilated Residual Block (DRB).

    PAPER FACT: "It then employs a Dilated Residual Block (DRB), to capture
    multi-scale contextual information." (Sec. 3.2, low-frequency branch)
    "the network introduces residual blocks with dilated convolutions (DRB)
    to expand the receptive field" (Sec. 3.4, high-frequency branch).

    IMPLEMENTATION ASSUMPTION:
      - Two 3x3 dilated convolutions with LeakyReLU, plus a residual add.
      - Dilation rates default to (1, 2); configurable to (1, 2, 4).
      - Channel width is preserved (no bottleneck), which is the common
        "dilated residual" pattern in image restoration networks.
      - No BatchNorm (paper does not mention it; common in restorers to
        avoid shifting image statistics).
    """

    def __init__(
        self,
        channels: int,
        dilations: tuple[int, ...] = (1, 2),
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if len(dilations) < 1:
            raise ValueError("dilations must be non-empty")
        layers: list[nn.Module] = []
        in_ch = channels
        for d in dilations:
            layers.append(
                ConvNormAct(
                    in_ch,
                    channels,
                    kernel_size=3,
                    dilation=d,
                    activation=activation,
                    negative_slope=negative_slope,
                    norm="none",
                )
            )
            in_ch = channels
        self.body = nn.Sequential(*layers)
        # Final projection keeps the residual stream at `channels`.
        self.proj = (
            nn.Identity()
            if len(dilations) == 1 and False
            else nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        )
        self.residual_scale = residual_scale
        self.act = make_activation(activation, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = self.proj(out)
        return self.act(x + self.residual_scale * out)


class ConvBlock(nn.Module):
    """Simple conv-act-conv block used in encoder/decoder stages."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                activation=activation,
                negative_slope=negative_slope,
            ),
            ConvNormAct(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                activation=activation,
                negative_slope=negative_slope,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    """Stride-2 downsample: conv s=2 followed by ConvBlock."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)
        self.act = make_activation(activation, negative_slope)
        self.body = ConvBlock(out_ch, out_ch, activation, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.act(self.down(x)))


class Upsample(nn.Module):
    """Bilinear upsample + pixelshuffle-free 1x1/3x3 refine."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.body = ConvBlock(in_ch, out_ch, activation, negative_slope)

    def forward(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.body(x)


class ResidualConvStack(nn.Module):
    """Stack of residual blocks with an optional global residual."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        dilations: tuple[int, ...] = (1, 2),
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
        global_residual: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    channels,
                    dilations=dilations,
                    activation=activation,
                    negative_slope=negative_slope,
                )
                for _ in range(num_blocks)
            ]
        )
        self.global_residual = global_residual
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = self.out_proj(h)
        return x + h if self.global_residual else h
