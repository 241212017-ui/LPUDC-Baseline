#!/usr/bin/env python
"""Visualize Laplacian pyramid decomposition of arbitrary images.

Useful to verify that the pyramid implementation actually separates
frequency bands (not a superficial naming trick).

Usage:
    python tools/visualize_pyramid.py --image path/to/img.png --levels 3
    python tools/visualize_pyramid.py --config configs/lpudc_poled.yaml \
        --split val --index 0 --checkpoint checkpoints/.../latest.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.transforms import pil_to_tensor, tensor_to_pil
from models.laplacian_pyramid import LaplacianPyramid
from tools.visualize_utils import make_row, normalize_for_vis
from utils import load_checkpoint, load_config, resolve_device


def visualize_decomposition(image_path: Path, levels: int, out_dir: Path) -> None:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    x = pil_to_tensor(img).unsqueeze(0)
    lp = LaplacianPyramid(levels)
    residuals, coarse, all_levels = lp(x)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    # Save each residual (min-max normalized for visibility) and the coarse.
    rows = []
    for i, res in enumerate(residuals):
        vis = normalize_for_vis(res)
        rows.append((f"h{i} (HF residual)", vis))
        tensor_to_pil(vis[0]).save(out_dir / f"{stem}_h{i}.png")
    rows.append(("coarse (LF)", coarse.clamp(0, 1)))
    tensor_to_pil(coarse.clamp(0, 1)[0]).save(out_dir / f"{stem}_coarse.png")

    # Reconstruct and save side-by-side
    recon = lp.reconstruct(residuals, coarse)
    strip = make_row(
        [x.clamp(0, 1)[0], recon.clamp(0, 1)[0]],
        labels=["INPUT", "RECONSTRUCTED"],
    )
    strip.save(out_dir / f"{stem}_recon_strip.png")

    # All levels grid
    grid_imgs = [normalize_for_vis(lv) if lv.abs().mean() < 0.2 else lv.clamp(0, 1) for lv in all_levels]
    grid = make_row([g[0] for g in grid_imgs], labels=[f"L{i}" for i in range(len(grid_imgs))])
    grid.save(out_dir / f"{stem}_pyramid_grid.png")
    err = (recon - x).abs().mean().item()
    print(f"Saved pyramid visualizations to {out_dir}")
    print(f"Reconstruction MAE vs input: {err:.6f} (should be small; LP is near-invertible)")


def visualize_model_pyramid(
    config_path: Path,
    checkpoint: Path | None,
    split: str,
    index: int,
    levels: int,
    out_dir: Path,
) -> None:
    from data.dataset import build_dataset

    cfg = load_config(config_path)
    dataset = build_dataset(cfg, split=split, train=False)
    sample = dataset[index]
    deg = sample["degraded"].unsqueeze(0)
    gt = sample["gt"].unsqueeze(0)

    device = resolve_device("auto")
    from models.lpudc import build_lpudc

    model_cfg = dict(cfg.get("model", {}))
    model_cfg["pyramid_levels"] = levels
    model_cfg["debug_shapes"] = True
    model = build_lpudc(model_cfg).to(device).eval()
    if checkpoint is not None:
        load_checkpoint(checkpoint, model, map_location=device)

    with torch.no_grad():
        out = model(deg.to(device), return_extras=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    name = str(sample.get("name", f"{split}_{index}"))
    strip = make_row(
        [
            deg[0],
            out["restored"][0].clamp(0, 1).cpu(),
            gt[0],
        ],
        labels=["DEGRADED", "RESTORED", "GT"],
    )
    strip.save(out_dir / f"{name}_comparison.png")

    # Pred vs degraded pyramid levels
    for i, (plv, dlv) in enumerate(zip(out["pred_levels"], out["degraded_levels"])):
        p_vis = normalize_for_vis(plv[0:1].cpu())
        d_vis = normalize_for_vis(dlv[0:1].cpu())
        row = make_row([d_vis[0], p_vis[0]], labels=[f"degraded L{i}", f"predicted L{i}"])
        row.save(out_dir / f"{name}_level{i}.png")
    print(f"Saved model pyramid visualization to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="visualizations/pyramid")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.image:
        visualize_decomposition(Path(args.image), args.levels, out_dir)
    elif args.config:
        visualize_model_pyramid(
            Path(args.config),
            Path(args.checkpoint) if args.checkpoint else None,
            args.split,
            args.index,
            args.levels,
            out_dir,
        )
    else:
        raise SystemExit("Provide --image or --config")


if __name__ == "__main__":
    main()
