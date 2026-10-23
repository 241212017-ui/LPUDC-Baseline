#!/usr/bin/env python
"""Train LPUDC on a custom paired dataset.

This script is a thin wrapper around the standard training pipeline with
the custom_dataset config. The architecture is UNCHANGED from the paper
baseline; only the dataset path differs.

Steps:
    1. Analyze the dataset (writes docs/DATASET_REPORT.md)
    2. Train LPUDC with configs/custom_dataset.yaml
    3. Evaluate PSNR/SSIM/LPIPS/DISTS on the test split
    4. Generate visual results

Usage:
    python scripts/train_custom.py --config configs/custom_dataset.yaml
    python scripts/train_custom.py --config configs/custom_dataset.yaml \
        --root /path/to/your/dataset --iterations 50000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import load_config, save_config


def main() -> None:
    p = argparse.ArgumentParser(description="Train LPUDC on a custom dataset")
    p.add_argument("--config", type=str, default="configs/custom_dataset.yaml")
    p.add_argument("--root", type=str, default=None, help="Override dataset.root")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.root:
        cfg.setdefault("dataset", {})["root"] = args.root
    if args.iterations is not None:
        cfg.setdefault("training", {})["iterations"] = args.iterations
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.device:
        cfg.setdefault("experiment", {})["device"] = args.device

    # Persist overrides to a run-local config so experiments are recorded.
    run_cfg_path = Path(cfg.get("experiment", {}).get("checkpoint_dir", "checkpoints/lpudc_custom"))
    run_cfg_path.mkdir(parents=True, exist_ok=True)
    active_cfg = run_cfg_path / "config.active.yaml"
    save_config(cfg, active_cfg)
    print(f"Active config: {active_cfg}")

    root = cfg.get("dataset", {}).get("root")
    print(f"Dataset root: {root}")

    # 1) Dataset analysis
    if not args.skip_analysis and not args.evaluate_only:
        print("\n===== DATASET ANALYSIS =====")
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "tools" / "dataset_analysis.py"),
            "--config",
            str(active_cfg),
        ]
        subprocess.run(cmd, cwd=str(_REPO_ROOT))
        report = Path("docs/DATASET_REPORT.md")
        if report.is_file():
            text = report.read_text(encoding="utf-8", errors="ignore")
            if "NOT USABLE" in text or "**STOP.**" in text:
                print("\nSTOP: dataset is not suitable paired restoration data.")
                print("See docs/DATASET_REPORT.md. LPUDC requires degraded->GT pairs.")
                sys.exit(2)

    ckpt_dir = Path(cfg.get("experiment", {}).get("checkpoint_dir", "checkpoints/lpudc_custom"))
    out_dir = Path(cfg.get("experiment", {}).get("output_dir", "results/lpudc_custom"))
    ckpt = args.checkpoint
    if ckpt is None:
        for cand in (ckpt_dir / "best_psnr.pth", ckpt_dir / "best_ssim.pth", ckpt_dir / "latest.pth"):
            if cand.is_file():
                ckpt = str(cand)
                break

    # 2) Train
    if not args.skip_train and not args.evaluate_only:
        print("\n===== TRAINING =====")
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "train.py"),
            "--config",
            str(active_cfg),
        ]
        if args.device:
            cmd += ["--device", args.device]
        subprocess.run(cmd, cwd=str(_REPO_ROOT))
        # Refresh ckpt pointer
        for cand in (ckpt_dir / "best_psnr.pth", ckpt_dir / "best_ssim.pth", ckpt_dir / "latest.pth"):
            if cand.is_file():
                ckpt = str(cand)
                break

    # 3) Evaluate
    if ckpt is None:
        print("No checkpoint found; skipping evaluation.")
        return
    print("\n===== EVALUATION (test split) =====")
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "test.py"),
        "--config",
        str(active_cfg),
        "--checkpoint",
        ckpt,
        "--split",
        "test",
        "--output-dir",
        str(out_dir),
        "--save-visualizations",
        "--save-pyramid",
    ]
    if args.device:
        cmd += ["--device", args.device]
    subprocess.run(cmd, cwd=str(_REPO_ROOT))

    # 4) Optional: reproduction-style report (paper numbers are reference only)
    print("\n===== REPRODUCTION-STYLE REPORT =====")
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "evaluate.py"),
        "--config",
        str(active_cfg),
        "--checkpoint",
        ckpt,
        "--split",
        "test",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=str(_REPO_ROOT))

    print("\nCustom-dataset experiment complete.")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Results:     {out_dir}")
    print(
        "NOTE: Baseline LPUDC architecture was NOT modified. Only after this "
        "baseline works should you design new modules (e.g. attention fusion)."
    )


if __name__ == "__main__":
    main()
