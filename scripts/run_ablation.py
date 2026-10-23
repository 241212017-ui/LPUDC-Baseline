#!/usr/bin/env python
"""Run LPUDC ablation experiments and build an automatic results table.

Paper ablations (Tables 3-8):
    A. No multi-scale interaction
    B. Multi-scale interaction without guided filtering (direct concat)
    C. Multi-scale interaction + guided filtering (full LPUDC)
    D. Pyramid levels 2 / 3 / 4
    E. Loss: L1 only vs L1 + pyramid loss

Configs live in configs/ablations/.

Usage:
    python scripts/run_ablation.py --configs configs/ablations/ablation_a_no_multiscale.yaml ...
    python scripts/run_ablation.py --all --dataset-root data/POLED --max-iters 200
    python scripts/run_ablation.py --table-only   # rebuild table from existing results

Missing metrics are written as `null` / `--`. NEVER fabricate numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import load_config

ABLATION_CONFIGS = [
    ("A_no_multiscale", "configs/ablations/ablation_a_no_multiscale.yaml"),
    ("B_concat_no_dgf", "configs/ablations/ablation_b_concat_no_dgf.yaml"),
    ("C_full_dgf", "configs/ablations/ablation_c_full_dgf.yaml"),
    ("D_levels_2", "configs/ablations/ablation_d_levels_2.yaml"),
    ("D_levels_3", "configs/ablations/ablation_d_levels_3.yaml"),
    ("D_levels_4", "configs/ablations/ablation_d_levels_4.yaml"),
    ("E_l1_only", "configs/ablations/ablation_e_l1_only.yaml"),
    ("E_l1_plus_pyr", "configs/ablations/ablation_e_l1_plus_pyr.yaml"),
]


def run_one(config_path: Path, extra_args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, str(_REPO_ROOT / "train.py"), "--config", str(config_path)] + extra_args
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    cfg = load_config(config_path)
    out_dir = Path(cfg.get("experiment", {}).get("output_dir", "results/run"))
    metrics_path = out_dir / "metrics.json"
    result: dict[str, Any] = {
        "config": str(config_path),
        "returncode": proc.returncode,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
        "metrics": None,
        "params_M": None,
    }
    # Params from model
    try:
        from models.lpudc import build_lpudc
        from utils import count_parameters

        model = build_lpudc(cfg.get("model", {}))
        total, _ = count_parameters(model)
        result["params_M"] = round(total / 1e6, 3)
    except Exception as exc:  # noqa: BLE001
        result["params_error"] = str(exc)
    return result


def collect_result(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = Path(cfg.get("experiment", {}).get("output_dir", "results/run"))
    metrics_path = out_dir / "metrics.json"
    metrics = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")).get("metrics")
        except Exception:
            metrics = None
    try:
        from models.lpudc import build_lpudc
        from utils import count_parameters

        model = build_lpudc(cfg.get("model", {}))
        total, _ = count_parameters(model)
        params_m = round(total / 1e6, 3)
    except Exception:
        params_m = None
    return {
        "name": config_path.stem,
        "config": str(config_path),
        "params_M": params_m,
        "metrics": metrics,
    }


def write_table(rows: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Experiment", "Params (M)", "PSNR", "SSIM", "LPIPS", "DISTS"])
        for r in rows:
            m = r.get("metrics") or {}
            w.writerow(
                [
                    r["name"],
                    r.get("params_M") if r.get("params_M") is not None else "",
                    m.get("PSNR") if m else "",
                    m.get("SSIM") if m else "",
                    m.get("LPIPS") if m else "",
                    m.get("DISTS") if m else "",
                ]
            )

    lines = [
        "# LPUDC Ablation Results",
        "",
        "Directions: **PSNR/SSIM higher is better**, **LPIPS/DISTS lower is better**.",
        "",
        "Missing entries are `--` (not measured / not fabricated).",
        "",
        "| Experiment | Params (M) | PSNR^ | SSIM^ | LPIPSv | DISTSv |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = r.get("metrics") or {}

        def fmt(key: str, nd: int = 4) -> str:
            v = m.get(key) if m else None
            return "--" if v is None else f"{v:.{nd}f}"

        lines.append(
            f"| {r['name']} | {r.get('params_M') if r.get('params_M') is not None else '--'} "
            f"| {fmt('PSNR')} | {fmt('SSIM')} | {fmt('LPIPS')} | {fmt('DISTS')} |"
        )
    lines.append("")
    lines.append("Paper reference (Tables 3-8) for orientation only -- not compared automatically")
    lines.append("unless you fill in measured values via the training pipeline.")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run LPUDC ablations")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--all", action="store_true", help="Run all predefined ablation configs")
    p.add_argument("--dataset-root", type=str, default=None, help="Override dataset.root in each config")
    p.add_argument("--max-iters", type=int, default=None, help="Override training.iterations")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--table-only", action="store_true", help="Only rebuild the results table")
    p.add_argument("--out-csv", type=str, default="results/ablations/ablation_table.csv")
    p.add_argument("--out-md", type=str, default="results/ablations/ablation_table.md")
    args = p.parse_args()

    if args.all:
        config_paths = [Path(rel) for _, rel in ABLATION_CONFIGS]
    elif args.configs:
        config_paths = [Path(c) for c in args.configs]
    else:
        # Default: table-only from all known configs
        config_paths = [Path(rel) for _, rel in ABLATION_CONFIGS]
        args.table_only = True

    rows: list[dict[str, Any]] = []
    if not args.table_only:
        for cp in config_paths:
            if not cp.is_file():
                print(f"WARNING: missing config {cp}, skipping")
                continue
            extra: list[str] = []
            if args.dataset_root:
                # Patch config into a temp copy
                cfg = load_config(cp)
                cfg.setdefault("dataset", {})["root"] = args.dataset_root
                tmp = Path("results/ablations") / f"_patched_{cp.name}"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                import yaml

                tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                cp = tmp
            if args.max_iters is not None:
                extra += ["--iterations", str(args.max_iters)]
            if args.device:
                extra += ["--device", args.device]
            res = run_one(cp, extra)
            # After training, evaluate test split to fill metrics.json in output_dir
            try:
                cfg = load_config(cp)
                out_dir = Path(cfg.get("experiment", {}).get("output_dir", "results/run"))
                ckpt = out_dir / "checkpoints" / "best_psnr.pth"
                # train.py saves checkpoints under experiment.checkpoint_dir
                ckpt_dir = Path(cfg.get("experiment", {}).get("checkpoint_dir", "checkpoints/run"))
                for cand in (ckpt_dir / "best_psnr.pth", ckpt_dir / "latest.pth"):
                    if cand.is_file():
                        eval_cmd = [
                            sys.executable,
                            str(_REPO_ROOT / "test.py"),
                            "--config",
                            str(cp),
                            "--checkpoint",
                            str(cand),
                            "--split",
                            "test",
                            "--output-dir",
                            str(out_dir),
                            "--no-visualizations",
                        ]
                        subprocess.run(eval_cmd, cwd=str(_REPO_ROOT))
                        break
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: post-eval failed: {exc}")
            rows.append(collect_result(Path(args.configs[0]) if False else Path(cp)))
    else:
        for cp in config_paths:
            if cp.is_file():
                rows.append(collect_result(cp))

    write_table(rows, Path(args.out_csv), Path(args.out_md))


if __name__ == "__main__":
    main()
