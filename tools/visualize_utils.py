"""Visualization utilities: image grids, pyramid dumps, comparison strips."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data.transforms import tensor_to_pil


def _safe_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def make_row(
    images: Sequence[torch.Tensor | Image.Image | np.ndarray],
    labels: Sequence[str] | None = None,
    pad: int = 4,
    label_h: int = 22,
    bg: int = 32,
) -> Image.Image:
    """Stack images horizontally with optional text labels."""
    pil_imgs: list[Image.Image] = []
    for im in images:
        if isinstance(im, torch.Tensor):
            pil_imgs.append(tensor_to_pil(im))
        elif isinstance(im, np.ndarray):
            arr = im
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            pil_imgs.append(Image.fromarray(arr[..., :3], mode="RGB"))
        else:
            pil_imgs.append(im.convert("RGB"))

    h = max(im.height for im in pil_imgs)
    w = sum(im.width for im in pil_imgs) + pad * (len(pil_imgs) - 1)
    total_h = h + (label_h if labels else 0)
    canvas = Image.new("RGB", (w, total_h), color=(bg, bg, bg))
    x = 0
    font = _safe_font(14)
    for i, im in enumerate(pil_imgs):
        y0 = label_h if labels else 0
        canvas.paste(im, (x, y0))
        if labels and i < len(labels):
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 2, 2), labels[i], fill=(255, 255, 255), font=font)
        x += im.width + pad
    return canvas


def save_comparison(
    degraded: torch.Tensor,
    restored: torch.Tensor,
    gt: torch.Tensor,
    path: str | Path,
    name: str | None = None,
) -> Path:
    """Save [DEGRADED] [RESTORED] [GROUND TRUTH] strip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = make_row(
        [degraded, restored, gt],
        labels=["DEGRADED", "RESTORED", "GROUND TRUTH"],
    )
    out = path if path.suffix else path / f"{name or 'comparison'}.png"
    if out.suffix == "":
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    row.save(out)
    return out


def normalize_for_vis(x: torch.Tensor, per_channel: bool = True) -> torch.Tensor:
    """Min-max normalize a feature map to [0, 1] for visualization."""
    x = x.detach().float()
    if per_channel:
        dim = (1, 2, 3) if x.dim() == 4 else (0, 1, 2)
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def save_pyramid_visualization(
    degraded: torch.Tensor,
    gt: torch.Tensor,
    pred_levels: Sequence[torch.Tensor],
    gt_levels: Sequence[torch.Tensor],
    out_dir: str | Path,
    name: str,
    max_levels: int = 6,
) -> list[Path]:
    """Save degraded / GT pyramid / predicted pyramid / restored images.

    This is critical for verifying the model learns frequency decomposition
    rather than behaving like a plain CNN.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    paths = save_comparison(degraded, pred_levels[0] if pred_levels else degraded, gt, out_dir / name)
    # Note: pred_levels[0] is finest residual — we save a proper comparison
    # of the final image separately by the caller. Here we dump levels.
    saved.append(paths)

    # Per-level side-by-side: GT level vs predicted level
    n = min(len(pred_levels), len(gt_levels), max_levels)
    for i in range(n):
        g = gt_levels[i]
        p = pred_levels[i]
        if p.shape != g.shape:
            p = torch.nn.functional.interpolate(
                p, size=g.shape[-2:], mode="bilinear", align_corners=False
            )
        # Visualize residual-like content with min-max norm if values are tiny.
        g_vis = normalize_for_vis(g) if g.abs().mean() < 0.2 else g.clamp(0, 1)
        p_vis = normalize_for_vis(p) if p.abs().mean() < 0.2 else p.clamp(0, 1)
        row = make_row(
            [g_vis, p_vis],
            labels=[f"GT level {i}", f"PRED level {i}"],
        )
        fp = out_dir / f"{name}_level{i}.png"
        row.save(fp)
        saved.append(fp)
    return saved


def save_guidance_visualization(
    guidance_maps: Sequence[torch.Tensor],
    out_dir: str | Path,
    name: str,
) -> list[Path]:
    """Save DGF intermediate guidance outputs (min-max normalized)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, g in enumerate(guidance_maps):
        vis = normalize_for_vis(g[0:1] if g.dim() == 4 else g)
        # Take first channel for a simple heatmap-like view, or RGB-like 3ch.
        if vis.shape[1] >= 3:
            vis = vis[:, :3]
        elif vis.shape[1] == 1:
            vis = vis.repeat(3, 1, 1)
        fp = out_dir / f"{name}_guidance_{i}.png"
        tensor_to_pil(vis).save(fp)
        saved.append(fp)
    return saved
