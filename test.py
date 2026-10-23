#!/usr/bin/env python
"""LPUDC testing / evaluation entry point.

Computes PSNR, SSIM, LPIPS, DISTS on a dataset split (default: test).

Direction of improvement:
    PSNR  higher is better
    SSIM  higher is better
    LPIPS lower is better
    DISTS lower is better

Outputs:
    results/metrics.json
    results/metrics.csv
    results/visualizations/[DEGRADED][RESTORED][GROUND TRUTH] strips

Usage:
    python test.py --config configs/lpudc_poled.yaml \
        --checkpoint checkpoints/lpudc_poled/best_psnr.pth \
        --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data import build_dataloader, build_dataloader_real
from metrics import MetricSuite
from models.lpudc import build_lpudc
from tools.visualize_utils import save_comparison, save_pyramid_visualization
from utils import (
    ensure_dirs,
    load_checkpoint,
    load_config,
    print_device_info,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LPUDC on a dataset split")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--save-visualizations", action="store_true", default=None)
    p.add_argument("--no-visualizations", action="store_true")
    p.add_argument("--save-pyramid", action="store_true", help="Also dump pyramid levels")
    return p.parse_args()


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    metrics: MetricSuite,
    device: torch.device,
    out_dir: Path,
    split: str,
    save_vis: bool = True,
    max_vis: int = 30,
    save_pyramid: bool = False,
    max_images: int | None = None,
) -> dict[str, Any]:
    model.eval()
    vis_dir = out_dir / "visualizations" / split
    pyr_dir = out_dir / "visualizations" / f"{split}_pyramid"
    if save_vis:
        ensure_dirs(vis_dir)
    if save_pyramid:
        ensure_dirs(pyr_dir)

    per_image: list[dict[str, Any]] = []
    sums = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "DISTS": 0.0}
    counts = {"PSNR": 0, "SSIM": 0, "LPIPS": 0, "DISTS": 0}
    vis_saved = 0
    n_images = 0
    t0 = time.time()

    for batch in loader:
        deg = batch["degraded"].to(device)
        gt = batch["gt"].to(device)
        names = batch.get("name", None)
        out = model(deg, return_extras=True)
        restored = out["restored"].clamp(0.0, 1.0)

        vals = metrics(restored, gt)
        bs = deg.shape[0]
        for k, v in vals.items():
            if v is None:
                continue
            sums[k] += v * bs
            counts[k] += bs

        for i in range(bs):
            name = str(names[i]) if names is not None else f"{n_images:05d}"
            rec: dict[str, Any] = {"name": name}
            # Per-image metrics
            from metrics.psnr import psnr_per_image
            from metrics.ssim import ssim_per_image

            rec["PSNR"] = float(psnr_per_image(restored[i : i + 1], gt[i : i + 1]).item())
            rec["SSIM"] = float(ssim_per_image(restored[i : i + 1], gt[i : i + 1]).item())
            per_image.append(rec)

            if save_vis and vis_saved < max_vis:
                save_comparison(
                    deg[i].cpu(),
                    restored[i].cpu(),
                    gt[i].cpu(),
                    vis_dir / f"{vis_saved:04d}_{name}.png",
                )
                vis_saved += 1

            if save_pyramid and vis_saved <= max_vis:
                try:
                    save_pyramid_visualization(
                        deg[i].cpu(),
                        gt[i].cpu(),
                        [lv[i].cpu() for lv in out["pred_levels"]],
                        [lv[i].cpu() for lv in out["degraded_levels"]],
                        pyr_dir / f"{vis_saved:04d}_{name}",
                        name="pyramid",
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: pyramid dump failed for {name}: {exc}")

            n_images += 1

        if max_images is not None and n_images >= max_images:
            break

    elapsed = time.time() - t0
    summary: dict[str, Any] = {
        "split": split,
        "num_images": n_images,
        "seconds": elapsed,
        "seconds_per_image": elapsed / max(1, n_images),
        "metrics": {},
        "directions": MetricSuite.DIRECTIONS,
        "per_image": per_image,
        "note": (
            "LPIPS/DISTS may be null if the corresponding package/backbone "
            "is unavailable. Null is never replaced by a fabricated value."
        ),
    }
    for k in ("PSNR", "SSIM", "LPIPS", "DISTS"):
        if counts[k] > 0:
            summary["metrics"][k] = sums[k] / counts[k]
        else:
            summary["metrics"][k] = None
    return summary


def write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    ensure_dirs(out_dir)
    # JSON
    json_path = out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # CSV (aggregate + per-image)
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "num_images", "PSNR", "SSIM", "LPIPS", "DISTS", "seconds"])
        m = summary["metrics"]
        w.writerow(
            [
                summary["split"],
                summary["num_images"],
                m.get("PSNR"),
                m.get("SSIM"),
                m.get("LPIPS"),
                m.get("DISTS"),
                f"{summary['seconds']:.3f}",
            ]
        )
        w.writerow([])
        w.writerow(["name", "PSNR", "SSIM"])
        for rec in summary.get("per_image", []):
            w.writerow([rec.get("name"), rec.get("PSNR"), rec.get("SSIM")])
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    exp = cfg.get("experiment", {})
    set_seed(int(exp.get("seed", 42)))

    if args.device:
        cfg.setdefault("experiment", {})["device"] = args.device
    device = resolve_device(str(cfg.get("experiment", {}).get("device", "auto")))
    print_device_info(device)

    out_dir = Path(
        args.output_dir
        or cfg.get("experiment", {}).get("output_dir", "results/lpudc")
    )
    ensure_dirs(out_dir)

    # Loader
    dataset_loader = cfg.get("dataset", {}).get("loader", "standard")
    if dataset_loader == "poled_toled":
        loader = build_dataloader_real(
            cfg, split=args.split, train=False, batch_size=args.batch_size, num_workers=0
        )
    else:
        loader = build_dataloader(
            cfg, split=args.split, train=False, batch_size=args.batch_size, num_workers=0
        )
    print(f"{args.split} pairs: {len(loader.dataset)}")

    # Model + checkpoint
    model = build_lpudc(cfg.get("model", {})).to(device)
    payload = load_checkpoint(args.checkpoint, model, map_location=device)
    print(
        f"Loaded checkpoint {args.checkpoint} "
        f"(iteration={payload.get('iteration')}, "
        f"best={payload.get('best_metrics')})"
    )
    model.eval()

    # Metrics
    ecfg = cfg.get("evaluation", {})
    save_vis = bool(ecfg.get("save_visualizations", True))
    if args.no_visualizations:
        save_vis = False
    if args.save_visualizations:
        save_vis = True
    metrics = MetricSuite(
        device=device,
        lpips_net=str(ecfg.get("lpips_net", "alex")),
        enable_lpips=bool(ecfg.get("enable_lpips", True)),
        enable_dists=bool(ecfg.get("enable_dists", True)),
    )
    if metrics.lpips is not None and not metrics.lpips.available:
        print("WARNING: LPIPS unavailable -- will be reported as null")
    if metrics.dists is not None and not metrics.dists.available:
        print("WARNING: DISTS unavailable -- will be reported as null")

    summary = run_evaluation(
        model,
        loader,
        metrics,
        device,
        out_dir,
        split=args.split,
        save_vis=save_vis,
        max_vis=int(ecfg.get("max_vis_images", 30)),
        save_pyramid=bool(args.save_pyramid),
        max_images=args.max_images,
    )
    write_outputs(summary, out_dir)

    print("\n===== SUMMARY (higher better: PSNR, SSIM | lower better: LPIPS, DISTS) =====")
    m = summary["metrics"]
    for k, v in m.items():
        direction = MetricSuite.DIRECTIONS.get(k, "?")
        print(f"  {k:6s}: {v}   [{direction} is better]")
    print(f"  images: {summary['num_images']}  time/image: {summary['seconds_per_image']:.4f}s")
    print(
        "NOTE: Paper-reported reference values exist in docs/PAPER_SPECIFICATION.md "
        "but are NEVER used by this evaluation code."
    )


if __name__ == "__main__":
    main()
