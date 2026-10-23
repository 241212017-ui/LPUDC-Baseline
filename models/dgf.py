"""Deep Guided Filtering (DGF) module for LPUDC.

=============================================================================
Traditional Guided Filter (background)
=============================================================================
He et al.'s guided filter models the output in each local window omega_k as a
linear transform of the guidance image G:

    q_i = a_k * G_i + b_k,   for all i in omega_k                (Eq. 2)

Coefficients are obtained by minimizing

    sum_{i in omega_k} ( (a_k G_i + b_k - I_i)^2 + eps * a_k^2 )  (Eq. 3)

with closed form

    a_k = cov_omega(G, I) / (var_omega(G) + eps)
    b_k = mean_omega(I) - a_k * mean_omega(G)                     (Eq. 4)

and final output (averaging a_k, b_k over all windows containing i)

    q_i = a_bar_i * G_i + b_bar_i                                 (Eq. 5)

eps is typically 0.001. The filter smooths flat regions while preserving
edges that exist in the guidance image.

=============================================================================
Why traditional guided filtering is insufficient for UDC
=============================================================================
(PAPER FACT, Sec. 3.3)
- The linear assumption q = a G + b is "overly idealized for complex UDC
  degradation scenarios."
- It cannot effectively model nonlinear distortions such as screen
  interference, color shifts, and high-frequency artifacts.
- Weighted guided filter [Li et al.] only partially addresses this.

=============================================================================
What LPUDC changes (PAPER FACT, Sec. 3.3)
=============================================================================
1. Replace local averaging for mean_G / mean_I with a **learnable weighted
   mean**; weights are dynamically learned by a neural network.
2. Replace the fixed linear mapping of Eq. 5 with **convolutional blocks**,
   removing the constraints of the fixed formula so outputs can capture
   more expressive nonlinear relationships.
3. Role assignment:
     - G (guidance) = INPUT of the low-frequency restoration network
     - I (filter input) = OUTPUT of the low-frequency restoration network
4. Estimate transformation coefficients a_k, b_k between input and guidance
   (instead of directly predicting the filtered image). These coefficients
   can be solved using only G and I.
5. Coefficients are upsampled to the spatial resolution of the UPPER pyramid
   layer, then the filtered result is obtained with the standard guided
   filter output formulation (Eq. 5), refined by convolutions.

=============================================================================
How DGF connects the two branches
=============================================================================
The low-frequency branch produces (a) a restored low-frequency image used as
I, and (b) guidance features derived from the low-frequency network input
path used as G. DGF estimates a, b from (G, I), upsamples them to the next
pyramid level's resolution, and applies them to guide high-frequency
enhancement at that finer level. This transfers structural cues from low to
high resolution with low computational cost.

Cited module lineage: Wu et al., "Fast end-to-end trainable guided filter",
CVPR 2018 (paper reference [9]).
=============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _avg_pool_same(x: torch.Tensor, kernel: int, stride: int = 1) -> torch.Tensor:
    """Average pool with reflect padding so spatial size is preserved when
    kernel is odd."""
    if kernel <= 1:
        return x
    pad = kernel // 2
    if pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=kernel, stride=stride)


class LearnableWeightMean(nn.Module):
    """Learnable weighted local mean.

    PAPER FACT: "we replace the averaging operation with a weighted mean,
    where the weights are dynamically learned by a neural network."

    IMPLEMENTATION ASSUMPTION: weights predicted by a depthwise 3x3 conv
    followed by softmax over the kernel spatial dimensions (channel-wise).
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_net = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logits: N x C x H x W, interpreted as weights of a depthwise filter
        logits = self.weight_net(x)
        k = self.kernel_size
        n, c, h, w_ = x.shape
        x_pad = F.pad(x, (k // 2,) * 4, mode="reflect")
        logits_pad = F.pad(logits, (k // 2,) * 4, mode="reflect")
        # unfold -> N, C*k*k, H*W
        x_unf = F.unfold(x_pad, kernel_size=k)
        l_unf = F.unfold(logits_pad, kernel_size=k)
        # Reshape to N, C, k*k, H*W so softmax is per-channel over kernel taps.
        l = h * w_
        x_unf = x_unf.view(n, c, k * k, l)
        l_unf = l_unf.view(n, c, k * k, l)
        w = torch.softmax(l_unf, dim=2)  # local softmax weights
        out = (x_unf * w).sum(dim=2)     # N, C, H*W
        return out.view(n, c, h, w_)


class CoefficientEstimator(nn.Module):
    """Estimate guided-filter coefficients a, b from guidance G and input I.

    Implements the learnable analogue of Eq. (4):
        a = cov(G, I) / (var(G) + eps)
        b = mean(I) - a * mean(G)
    with learnable weighted means, then a light conv refinement.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        eps: float = 0.001,
        refine_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.mean_g = LearnableWeightMean(channels, kernel_size)
        self.mean_i = LearnableWeightMean(channels, kernel_size)
        self.mean_gg = LearnableWeightMean(channels, kernel_size)
        self.mean_ii = LearnableWeightMean(channels, kernel_size)
        self.mean_gi = LearnableWeightMean(channels, kernel_size)
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, refine_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv2d(refine_hidden, channels * 2, kernel_size=3, padding=1),
        )

    def forward(
        self, g: torch.Tensor, i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (a, b, mean_g, mean_i)."""
        mu_g = self.mean_g(g)
        mu_i = self.mean_i(i)
        mu_gg = self.mean_gg(g * g)
        mu_ii = self.mean_ii(i * i)
        mu_gi = self.mean_gi(g * i)

        var_g = (mu_gg - mu_g * mu_g).clamp_min(0.0)
        cov_gi = mu_gi - mu_g * mu_i
        a = cov_gi / (var_g + self.eps)
        b = mu_i - a * mu_g

        a_b = self.refine(torch.cat([a, b], dim=1))
        a_out, b_out = torch.split(a_b, a.shape[1], dim=1)
        # Residual refine around the closed-form estimate.
        a = a + a_out
        b = b + b_out
        return a, b, mu_g, mu_i


class GuidedOutput(nn.Module):
    """Apply guided filter output q = a*G + b, refined by convolutions.

    PAPER FACT: "the filtered result is obtained using the standard guided
    filter output formulation described in Eq. (5)" but "we replace this
    linear mapping with convolutional blocks".
    """

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        q = a * g + b
        return q + self.refine(q)


class DeepGuidedFilter(nn.Module):
    """Deep Guided Filtering module used between pyramid levels.

    Args:
        guidance_channels: channels of the guidance feature G
                           (from the low-frequency branch / lower level).
        input_channels:    channels of the feature I to be guided
                           (high-frequency features at the upper level).
        coefficient_channels: channel width used for a/b estimation.
                              If None, uses guidance_channels.
        kernel_size: neighbourhood size for learnable weighted means.
        eps: guided-filter regularization (paper: typically 0.001).
        upsample: if True, bilinearly upsample a/b to the size of `g` when
                  their spatial size differs (coefficient alignment step).
    """

    def __init__(
        self,
        guidance_channels: int,
        input_channels: int,
        coefficient_channels: int | None = None,
        kernel_size: int = 3,
        eps: float = 0.001,
        upsample: bool = True,
        refine_hidden: int = 32,
        output_channels: int | None = None,
    ) -> None:
        super().__init__()
        c = coefficient_channels or guidance_channels
        self.coeff_channels = c
        self.upsample = upsample
        self.eps = eps
        out_ch = output_channels if output_channels is not None else input_channels
        self.output_channels = out_ch

        # Project both streams into the coefficient channel space.
        self.g_proj = nn.Conv2d(guidance_channels, c, kernel_size=1, bias=True)
        self.i_proj = nn.Conv2d(input_channels, c, kernel_size=1, bias=True)
        self.estimator = CoefficientEstimator(
            c, kernel_size=kernel_size, eps=eps, refine_hidden=refine_hidden
        )
        # Guided feature at coefficient resolution.
        self.guide_proj = nn.Conv2d(guidance_channels, c, kernel_size=1, bias=True)
        self.output = GuidedOutput(c, hidden=refine_hidden)
        # Project guided feature to the requested output channel count.
        self.out_proj = nn.Conv2d(c, out_ch, kernel_size=1, bias=True)

    def forward(
        self,
        g: torch.Tensor,
        i_feat: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Args:
            g: guidance features (from low-frequency branch), any resolution.
            i_feat: high-frequency features to be guided (upper pyramid level).
            target_size: optional (H, W) to align outputs to. If None, uses
                         i_feat spatial size.

        Returns:
            Guided features with the same spatial size as `i_feat` (or
            `target_size`) and channel count `input_channels`.
        """
        if target_size is None:
            target_size = i_feat.shape[-2:]

        g_proj = self.g_proj(g)
        i_proj = self.i_proj(i_feat)

        # Estimate a, b at the guidance (low-frequency) resolution.
        # PAPER FACT: G = input of LF restoration net, I = output of LF
        # restoration net; both live at the low-frequency scale. Coefficients
        # are then upsampled to the upper pyramid level.
        est_size = g_proj.shape[-2:]
        if i_proj.shape[-2:] != est_size:
            i_for_est = F.interpolate(
                i_proj, size=est_size, mode="bilinear", align_corners=False
            )
        else:
            i_for_est = i_proj
        a, b, _, _ = self.estimator(g_proj, i_for_est)

        # PAPER FACT: coefficients are upsampled to align with the spatial
        # resolution of the upper pyramid layer.
        if a.shape[-2:] != target_size:
            a = F.interpolate(a, size=target_size, mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=target_size, mode="bilinear", align_corners=False)
        g_ref = F.interpolate(g_proj, size=target_size, mode="bilinear", align_corners=False)

        guided = self.output(a, b, g_ref)  # c x H x W
        guided = self.out_proj(guided)     # output_channels x H x W
        return guided


class DGFMultiScaleFusion(nn.Module):
    """Convenience wrapper: fuse LF guidance into HF features.

    Returns the guided HF features ready for the high-frequency enhancement
    network.
    """

    def __init__(
        self,
        guidance_channels: int,
        input_channels: int,
        coefficient_channels: int | None = None,
        kernel_size: int = 3,
        eps: float = 0.001,
    ) -> None:
        super().__init__()
        self.dgf = DeepGuidedFilter(
            guidance_channels=guidance_channels,
            input_channels=input_channels,
            coefficient_channels=coefficient_channels,
            kernel_size=kernel_size,
            eps=eps,
        )

    def forward(
        self, g: torch.Tensor, i_feat: torch.Tensor
    ) -> torch.Tensor:
        return self.dgf(g, i_feat, target_size=i_feat.shape[-2:])
