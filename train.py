#!/usr/bin/env python
"""LPUDC training pipeline.

Features:
    - Configuration loading (YAML)
    - Dataset / DataLoader initialization
    - Model initialization
    - Adam optimizer + Cosine Annealing LR scheduler
    - Optional AMP / gradient scaling / gradient accumulation / grad checkpointing
    - Checkpoint saving (latest.pth, best_psnr.pth, best_ssim.pth)
    - Resume training
    - Validation with PSNR / SSIM / LPIPS / DISTS
    - Periodic image visualization
    - Loss / metric / LR logging (JSONL + CSV)

Paper-stated hyperparameters are read from the config; the reproduction
configs keep them exactly as reported. DEBUG configs are separate files.

Usage:
    python train.py --config configs/lpudc_poled.yaml
    python train.py --config configs/lpudc_poled_debug.yaml --amp
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Windows console compatibility: prefer UTF-8 output when possible.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data import build_dataloader, build_dataloader_real
from losses.lpudc_loss import build_loss
from metrics import MetricSuite
from models.lpudc import LPUDCConfig, build_lpudc
from tools.visualize_utils import save_comparison, save_guidance_visualization, save_pyramid_visualization
from utils import (
    AverageMeter,
    append_jsonl,
    count_parameters,
    ensure_dirs,
    format_seconds,
    load_checkpoint,
    load_config,
    print_device_info,
    resolve_device,
    save_checkpoint,
    save_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LPUDC")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--device", type=str, default=None, help="Override device (auto/cpu/cuda)")
    p.add_argument("--amp", action="store_true", help="Force-enable mixed precision")
    p.add_argument("--no-amp", action="store_true", help="Force-disable mixed precision")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--iterations", type=int, default=None, help="Override training iterations")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--max-images-visualize", type=int, default=None)
    return p.parse_args()


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = dict(cfg.get("model", {}))
    # Ensure loss/model share pyramid level count.
    model_cfg.setdefault("pyramid_levels", cfg.get("model", {}).get("pyramid_levels", 3))
    return build_lpudc(model_cfg)


def build_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    ocfg = cfg.get("optimization", {})
    name = str(ocfg.get("optimizer", "adam")).lower()
    lr = float(ocfg.get("lr", 2e-4))
    wd = float(ocfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(float(ocfg.get("beta1", 0.9)), float(ocfg.get("beta2", 0.999))),
            weight_decay=wd,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(float(ocfg.get("beta1", 0.9)), float(ocfg.get("beta2", 0.999))),
            weight_decay=wd,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    """Cosine annealing from lr to min_lr over total iterations (paper).

    PAPER FACT: Cosine Annealing scheme; LR reduced from 2e-4 to 1e-6.
    Training length is measured in iterations (300K), so T_max = iterations.
    """
    ocfg = cfg.get("optimization", {})
    tcfg = cfg.get("training", {})
    t_max = int(tcfg.get("iterations", 300000))
    eta_min = float(ocfg.get("min_lr", 1e-6))
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=eta_min
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    metrics: MetricSuite,
    device: torch.device,
    max_batches: int | None = None,
    save_dir: Path | None = None,
    max_vis: int = 4,
    tag: str = "val",
) -> dict[str, Any]:
    model.eval()
    meters = {k: AverageMeter() for k in ("PSNR", "SSIM", "LPIPS", "DISTS")}
    n_batches = 0
    vis_saved = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        deg = batch["degraded"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        out = model(deg, return_extras=False)
        restored = out["restored"]
        # Clamp to valid image range for metric computation.
        restored_c = restored.clamp(0.0, 1.0)
        vals = metrics(restored_c, gt)
        bs = deg.shape[0]
        for k, v in vals.items():
            if v is not None and k in meters:
                meters[k].update(v, bs)
        n_batches += 1
        if save_dir is not None and vis_saved < max_vis:
            for i in range(min(bs, max_vis - vis_saved)):
                name = batch.get("name", [f"{tag}_{bi}_{i}"])
                if isinstance(name, (list, tuple)):
                    stem = str(name[i])
                else:
                    stem = str(name)
                save_comparison(
                    deg[i].cpu(),
                    restored_c[i].cpu(),
                    gt[i].cpu(),
                    save_dir / f"{tag}_{vis_saved:04d}_{stem}.png",
                )
                vis_saved += 1

    result: dict[str, Any] = {"num_batches": n_batches}
    for k, m in meters.items():
        result[k] = m.avg if m.count > 0 else None
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides (clearly logged so configs remain the source of truth).
    if args.device:
        cfg.setdefault("experiment", {})["device"] = args.device
    if args.amp:
        cfg.setdefault("training", {})["amp"] = True
    if args.no_amp:
        cfg.setdefault("training", {})["amp"] = False
    if args.resume:
        cfg.setdefault("training", {})["resume"] = args.resume
    if args.iterations is not None:
        cfg.setdefault("training", {})["iterations"] = args.iterations
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed
    if args.max_images_visualize is not None:
        cfg.setdefault("evaluation", {})["max_vis_images"] = args.max_images_visualize

    exp = cfg.get("experiment", {})
    seed = int(exp.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(exp.get("device", "auto")))
    device_info = print_device_info(device)

    ckpt_dir = Path(exp.get("checkpoint_dir", "checkpoints/run"))
    log_dir = Path(exp.get("log_dir", "logs/run"))
    out_dir = Path(exp.get("output_dir", "results/run"))
    vis_dir = out_dir / "visualizations"
    ensure_dirs(ckpt_dir, log_dir, out_dir, vis_dir)

    # Save the exact config used for this run.
    save_config(cfg, ckpt_dir / "config.yaml")
    save_config(cfg, log_dir / "config.yaml")

    # ---- Data ----
    dataset_loader = cfg.get("dataset", {}).get("loader", "standard")
    if dataset_loader == "poled_toled":
        train_loader = build_dataloader_real(cfg, split="train", train=True)
        val_loader = build_dataloader_real(cfg, split="val", train=False)
    else:
        train_loader = build_dataloader(cfg, split="train", train=True)
        val_loader = build_dataloader(cfg, split="val", train=False)
    print(f"Train pairs: {len(train_loader.dataset)} | Val pairs: {len(val_loader.dataset)}")

    # ---- Model ----
    model = build_model(cfg).to(device)
    total_p, train_p = count_parameters(model)
    print(f"Model parameters: total={total_p:,} trainable={train_p:,} ({total_p/1e6:.2f}M)")

    # ---- Loss / Optim / Scheduler ----
    loss_fn = build_loss(
        {
            **cfg.get("loss", {}),
            "pyramid_levels": cfg.get("model", {}).get("pyramid_levels", 3),
        }
    ).to(device)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    tcfg = cfg.get("training", {})
    total_iters = int(tcfg.get("iterations", 300000))
    batch_size = int(tcfg.get("batch_size", 1))
    log_interval = int(tcfg.get("log_interval", 50))
    val_interval = int(tcfg.get("val_interval", 1000))
    save_interval = int(tcfg.get("save_interval", 5000))
    vis_interval = int(tcfg.get("vis_interval", 2000))
    grad_clip = tcfg.get("grad_clip", None)
    use_amp = bool(tcfg.get("amp", False))
    accum = max(1, int(tcfg.get("gradient_accumulation", 1)))
    use_grad_checkpoint = bool(tcfg.get("gradient_checkpointing", False))

    if use_grad_checkpoint:
        try:
            from torch.utils.checkpoint import checkpoint

            original_forward = model.forward

            def ckpt_forward(x, return_extras=True):
                return checkpoint(original_forward, x, return_extras, use_reentrant=False)

            model.forward = ckpt_forward  # type: ignore[method-assign]
            print("Gradient checkpointing: ENABLED (engineering option, not in paper)")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: gradient checkpointing failed to enable: {exc}")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    if use_amp:
        print(
            "AMP: ENABLED (engineering option -- the paper does NOT mention "
            "mixed-precision training; reproduction fidelity prefers amp=false)"
        )
    else:
        print("AMP: disabled (matches paper silence on mixed precision)")

    # ---- Metrics ----
    ecfg = cfg.get("evaluation", {})
    metrics = MetricSuite(
        device=device,
        lpips_net=str(ecfg.get("lpips_net", "alex")),
        enable_lpips=bool(ecfg.get("enable_lpips", True)) and device.type != "cpu",
        enable_dists=bool(ecfg.get("enable_dists", True)) and device.type != "cpu",
    )
    # On CPU, LPIPS/DISTS are slow; still allow if user forced them on.
    if device.type == "cpu":
        metrics = MetricSuite(
            device=device,
            lpips_net=str(ecfg.get("lpips_net", "alex")),
            enable_lpips=bool(ecfg.get("enable_lpips", False)),
            enable_dists=bool(ecfg.get("enable_dists", False)),
        )

    # ---- Resume ----
    start_iter = 0
    best_metrics: dict[str, Any] = {"best_psnr": -1.0, "best_ssim": -1.0}
    resume_path = tcfg.get("resume")
    if resume_path:
        payload = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)
        start_iter = int(payload.get("iteration", 0))
        best_metrics.update(payload.get("best_metrics", {}) or {})
        print(f"Resumed from {resume_path} at iteration {start_iter}")

    # ---- CSV header ----
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "iteration",
                    "train_total_loss",
                    "train_content_loss",
                    "train_pyramid_loss",
                    "lr",
                    "val_psnr",
                    "val_ssim",
                    "val_lpips",
                    "val_dists",
                    "elapsed_s",
                ]
            )

    # ---- Infinite iterator over train loader ----
    def infinite(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch

    train_iter = infinite(train_loader)

    model.train()
    loss_meter = AverageMeter()
    content_meter = AverageMeter()
    pyr_meter = AverageMeter()
    t_start = time.time()
    running = 0

    print(f"Starting training: {start_iter} -> {total_iters} iterations")
    print(f"Batch size: {batch_size} | Accum: {accum} | Device: {device}")

    for it in range(start_iter + 1, total_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_content = 0.0
        step_pyr = 0.0

        for micro in range(accum):
            mb = next(train_iter)
            deg = mb["degraded"].to(device, non_blocking=True)
            gt = mb["gt"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, enabled=use_amp and device.type == "cuda"
            ):
                out = model(deg, return_extras=False)
                losses = loss_fn(out["restored"], gt, out.get("pred_levels"))
                loss = losses["total_loss"] / accum

            scaler.scale(loss).backward()
            step_loss += float(losses["total_loss"].item())
            step_content += float(losses["content_loss"].item())
            step_pyr += float(losses["pyramid_loss"].item())

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        loss_meter.update(step_loss)
        content_meter.update(step_content)
        pyr_meter.update(step_pyr)
        running += 1

        if it % log_interval == 0 or it == 1:
            elapsed = time.time() - t_start
            rec = {
                "iteration": it,
                "train_total_loss": loss_meter.avg,
                "train_content_loss": content_meter.avg,
                "train_pyramid_loss": pyr_meter.avg,
                "lr": lr_now,
                "elapsed_s": elapsed,
            }
            append_jsonl(log_dir / "train_log.jsonl", rec)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        it,
                        f"{loss_meter.avg:.6f}",
                        f"{content_meter.avg:.6f}",
                        f"{pyr_meter.avg:.6f}",
                        f"{lr_now:.8f}",
                        "",
                        "",
                        "",
                        "",
                        f"{elapsed:.2f}",
                    ]
                )
            print(
                f"[{it}/{total_iters}] loss={loss_meter.avg:.4f} "
                f"(con={content_meter.avg:.4f}, pyr={pyr_meter.avg:.4f}) "
                f"lr={lr_now:.6e} elapsed={format_seconds(elapsed)}"
            )
            loss_meter.reset()
            content_meter.reset()
            pyr_meter.reset()
            running = 0

        # ---- Validation ----
        if it % val_interval == 0 or it == total_iters:
            val_res = validate(
                model,
                val_loader,
                metrics,
                device,
                max_batches=None,
                save_dir=vis_dir / f"iter_{it:07d}",
                max_vis=int(ecfg.get("max_vis_images", 4)),
                tag="val",
            )
            val_rec = {"iteration": it, **{f"val_{k}": v for k, v in val_res.items()}}
            append_jsonl(log_dir / "val_log.jsonl", val_rec)
            psnr_v = val_res.get("PSNR")
            ssim_v = val_res.get("SSIM")
            print(
                f"  VAL [{it}] PSNR={psnr_v} SSIM={ssim_v} "
                f"LPIPS={val_res.get('LPIPS')} DISTS={val_res.get('DISTS')}"
            )
            if psnr_v is not None:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            it,
                            "",
                            "",
                            "",
                            f"{lr_now:.8e}",
                            f"{psnr_v:.4f}",
                            f"{ssim_v:.4f}" if ssim_v is not None else "",
                            f"{val_res.get('LPIPS')}" if val_res.get("LPIPS") is not None else "",
                            f"{val_res.get('DISTS')}" if val_res.get("DISTS") is not None else "",
                            f"{time.time()-t_start:.2f}",
                        ]
                    )
            # Best checkpoints (metrics only -- never hardcoded paper numbers).
            if psnr_v is not None and psnr_v > float(best_metrics.get("best_psnr", -1e9)):
                best_metrics["best_psnr"] = psnr_v
                best_metrics["best_psnr_iteration"] = it
                save_checkpoint(
                    ckpt_dir / "best_psnr.pth",
                    model,
                    optimizer,
                    scheduler,
                    iteration=it,
                    best_metrics=best_metrics,
                    config=cfg,
                )
                print(f"  -> saved best_psnr.pth (PSNR={psnr_v:.4f})")
            if ssim_v is not None and ssim_v > float(best_metrics.get("best_ssim", -1e9)):
                best_metrics["best_ssim"] = ssim_v
                best_metrics["best_ssim_iteration"] = it
                save_checkpoint(
                    ckpt_dir / "best_ssim.pth",
                    model,
                    optimizer,
                    scheduler,
                    iteration=it,
                    best_metrics=best_metrics,
                    config=cfg,
                )
                print(f"  -> saved best_ssim.pth (SSIM={ssim_v:.4f})")
            model.train()

        # ---- Periodic visualization of training samples ----
        if it % vis_interval == 0 or it == 1:
            model.eval()
            with torch.no_grad():
                out = model(deg, return_extras=True)
            save_comparison(
                deg[0].cpu(),
                out["restored"][0].clamp(0, 1).cpu(),
                gt[0].cpu(),
                vis_dir / f"train_iter_{it:07d}.png",
            )
            try:
                save_pyramid_visualization(
                    deg[0].cpu(),
                    gt[0].cpu(),
                    [lv[0].cpu() for lv in out["pred_levels"]],
                    [lv[0].cpu() for lv in out["degraded_levels"]],
                    vis_dir / f"pyramid_iter_{it:07d}",
                    name="sample",
                )
                if out.get("guidance_maps"):
                    save_guidance_visualization(
                        [g[0:1].cpu() for g in out["guidance_maps"]],
                        vis_dir / f"guidance_iter_{it:07d}",
                        name="sample",
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: pyramid visualization failed: {exc}")
            model.train()

        # ---- Latest checkpoint ----
        if it % save_interval == 0 or it == total_iters:
            save_checkpoint(
                ckpt_dir / "latest.pth",
                model,
                optimizer,
                scheduler,
                iteration=it,
                best_metrics=best_metrics,
                config=cfg,
                extra={"device_info": device_info},
            )
            print(f"  -> saved latest.pth @ iter {it}")

    print("Training finished.")
    print(f"Best PSNR: {best_metrics.get('best_psnr')}")
    print(f"Best SSIM: {best_metrics.get('best_ssim')}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs: {log_dir}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
