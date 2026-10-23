#!/usr/bin/env python
"""Dataset analyzer for paired image-restoration datasets.

Generates:
    docs/DATASET_REPORT.md
    results/dataset_analysis/viz_grid.png  (visualization grid)
    results/dataset_analysis/stats.json

Reports:
    - Number of train/val/test pairs
    - Image resolutions / aspect ratios / color channels
    - Missing pairs, corrupted images, duplicates (hash-based)
    - Pixel-value ranges, mean/std, brightness statistics
    - Difference statistics between degraded and GT

If the data is NOT paired image-to-image restoration data, STOP with an
explanation — LPUDC as implemented in the paper requires paired restoration
training data.

Usage:
    python tools/dataset_analysis.py --config configs/custom_dataset.yaml
    python tools/dataset_analysis.py --root data/custom --splits train val test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.dataset import IMAGE_EXTS, PairedImageDataset
from utils import load_config

IMG_EXTS = IMAGE_EXTS


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def analyze_split(
    root: Path,
    split: str,
    degraded_dirname: str = "degraded",
    gt_dirname: str = "gt",
    max_images_stats: int = 300,
    max_viz: int = 16,
) -> dict[str, Any]:
    split_root = root / split
    deg_dir = split_root / degraded_dirname
    gt_dir = split_root / gt_dirname
    report: dict[str, Any] = {
        "split": split,
        "exists": split_root.is_dir(),
        "num_degraded": 0,
        "num_gt": 0,
        "num_paired": 0,
        "missing_gt": [],
        "missing_degraded": [],
        "corrupted": [],
        "dimension_mismatch": [],
        "resolutions": [],
        "aspect_ratios": [],
        "channels": Counter(),
        "degraded_stats": {},
        "gt_stats": {},
        "diff_stats": {},
        "duplicate_degraded": [],
        "duplicate_gt": [],
        "viz_samples": [],
        "paired_ok": True,
        "stop_reason": None,
    }
    if not split_root.is_dir():
        report["paired_ok"] = False
        report["stop_reason"] = f"Split directory missing: {split_root}"
        return report
    if not deg_dir.is_dir() or not gt_dir.is_dir():
        report["paired_ok"] = False
        report["stop_reason"] = (
            f"Expected {{degraded,gt}} under {split_root}. "
            "LPUDC requires paired image-to-image restoration data."
        )
        return report

    deg_files = sorted(
        p for p in deg_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    gt_files = sorted(
        p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    deg_map = {p.stem: p for p in deg_files}
    gt_map = {p.stem: p for p in gt_files}
    common = sorted(set(deg_map) & set(gt_map))
    report["num_degraded"] = len(deg_map)
    report["num_gt"] = len(gt_map)
    report["num_paired"] = len(common)
    report["missing_gt"] = sorted(set(deg_map) - set(gt_map))[:50]
    report["missing_degraded"] = sorted(set(gt_map) - set(deg_map))[:50]

    if len(common) == 0:
        report["paired_ok"] = False
        report["stop_reason"] = (
            "No matching filename stems between degraded/ and gt/. "
            "LPUDC requires paired degraded→GT images. Do not force "
            "incompatible data into the model."
        )
        return report

    # Hash-based duplicates
    deg_hashes: dict[str, list[str]] = defaultdict(list)
    gt_hashes: dict[str, list[str]] = defaultdict(list)
    res_counter: Counter = Counter()
    ar_counter: Counter = Counter()
    ch_counter: Counter = Counter()

    deg_vals: list[np.ndarray] = []
    gt_vals: list[np.ndarray] = []
    diff_vals: list[np.ndarray] = []

    viz_rows: list[tuple[str, Image.Image, Image.Image]] = []

    n_stats = min(len(common), max_images_stats)
    for i, stem in enumerate(common):
        dpath, gpath = deg_map[stem], gt_map[stem]
        if i < n_stats:
            try:
                deg_hashes[file_md5(dpath)].append(stem)
                gt_hashes[file_md5(gpath)].append(stem)
            except Exception as exc:  # noqa: BLE001
                report["corrupted"].append({"stem": stem, "error": str(exc)})
                report["paired_ok"] = False
                continue
        try:
            with Image.open(dpath) as dimg, Image.open(gpath) as gimg:
                dimg = dimg.convert("RGB")
                gimg = gimg.convert("RGB")
                if dimg.size != gimg.size:
                    report["dimension_mismatch"].append(
                        {
                            "stem": stem,
                            "degraded": list(dimg.size),
                            "gt": list(gimg.size),
                        }
                    )
                res_counter[f"{dimg.size[0]}x{dimg.size[1]}"] += 1
                ar = dimg.size[0] / max(1, dimg.size[1])
                ar_counter[f"{ar:.3f}"] += 1
                ch_counter["RGB(3)"] += 1
                report["resolutions"].append(list(dimg.size))
                report["aspect_ratios"].append(ar)

                if i < n_stats:
                    darr = np.asarray(dimg, dtype=np.float32) / 255.0
                    garr = np.asarray(gimg, dtype=np.float32) / 255.0
                    deg_vals.append(darr)
                    gt_vals.append(garr)
                    diff_vals.append(garr - darr)
                if len(viz_rows) < max_viz:
                    viz_rows.append((stem, dimg.copy(), gimg.copy()))
        except Exception as exc:  # noqa: BLE001
            report["corrupted"].append({"stem": stem, "error": str(exc)})
            report["paired_ok"] = False

    report["duplicate_degraded"] = [
        stems for stems in deg_hashes.values() if len(stems) > 1
    ][:20]
    report["duplicate_gt"] = [
        stems for stems in gt_hashes.values() if len(stems) > 1
    ][:20]

    def _stats(arrs: list[np.ndarray]) -> dict[str, Any]:
        if not arrs:
            return {}
        stack = np.stack(arrs, axis=0)  # N,H,W,C
        return {
            "num_sampled": int(stack.shape[0]),
            "global_min": float(stack.min()),
            "global_max": float(stack.max()),
            "global_mean": float(stack.mean()),
            "global_std": float(stack.std()),
            "per_channel_mean": [float(stack[..., c].mean()) for c in range(stack.shape[-1])],
            "per_channel_std": [float(stack[..., c].std()) for c in range(stack.shape[-1])],
            "brightness_mean": float(stack.mean(axis=(1, 2, 3)).mean()),
            "brightness_std": float(stack.mean(axis=(1, 2, 3)).std()),
        }

    report["degraded_stats"] = _stats(deg_vals)
    report["gt_stats"] = _stats(gt_vals)
    if diff_vals:
        dstack = np.stack(diff_vals, axis=0)
        report["diff_stats"] = {
            "mean_abs": float(np.abs(dstack).mean()),
            "mean": float(dstack.mean()),
            "std": float(dstack.std()),
            "p95_abs": float(np.percentile(np.abs(dstack), 95)),
            "p99_abs": float(np.percentile(np.abs(dstack), 99)),
        }
    report["resolutions_top"] = res_counter.most_common(10)
    report["aspect_ratios_top"] = ar_counter.most_common(10)
    report["channels"] = dict(ch_counter)
    report["viz_samples"] = viz_rows  # kept in-memory for grid drawing
    return report


def draw_grid(
    report_by_split: dict[str, dict[str, Any]],
    out_path: Path,
    cols: int = 4,
) -> Path | None:
    rows: list[tuple[str, Image.Image, Image.Image]] = []
    for split, rep in report_by_split.items():
        for stem, dimg, gimg in rep.get("viz_samples", [])[:8]:
            rows.append((f"{split}:{stem}", dimg, gimg))
    if not rows:
        return None
    thumb_w, thumb_h = 256, 256
    label_h = 18
    n = len(rows)
    rows_n = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows_n * (thumb_h + label_h)), (24, 24, 24))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for i, (name, dimg, gimg) in enumerate(rows):
        r, c = divmod(i, cols)
        # side-by-side degraded|gt inside the cell
        cell = Image.new("RGB", (thumb_w, thumb_h), (0, 0, 0))
        dth = dimg.copy()
        gth = gimg.copy()
        dth.thumbnail((thumb_w // 2, thumb_h))
        gth.thumbnail((thumb_w // 2, thumb_h))
        cell.paste(dth, (0, (thumb_h - dth.height) // 2))
        cell.paste(gth, (thumb_w // 2, (thumb_h - gth.height) // 2))
        y = r * (thumb_h + label_h)
        canvas.paste(cell, (c * thumb_w, y))
        draw.text((c * thumb_w + 4, y + thumb_h + 2), name[:40], fill=(220, 220, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def write_markdown(
    report_by_split: dict[str, dict[str, Any]],
    root: Path,
    out_path: Path,
    grid_path: Path | None,
) -> Path:
    lines: list[str] = []
    lines.append("# Dataset Report (LPUDC)")
    lines.append("")
    lines.append(f"- **Root:** `{root}`")
    lines.append("")
    any_stop = False
    for split, rep in report_by_split.items():
        lines.append(f"## Split: `{split}`")
        lines.append("")
        if not rep.get("exists", True):
            lines.append(f"- **Status:** MISSING ({rep.get('stop_reason')})")
            lines.append("")
            any_stop = True
            continue
        if rep.get("paired_ok") is False:
            any_stop = True
            lines.append(f"- **Status:** ⚠️ NOT USABLE as paired restoration data")
            lines.append(f"- **Reason:** {rep.get('stop_reason')}")
            lines.append("")
            continue
        lines.append(f"- **Degraded images:** {rep['num_degraded']}")
        lines.append(f"- **GT images:** {rep['num_gt']}")
        lines.append(f"- **Paired:** {rep['num_paired']}")
        lines.append(f"- **Missing GT:** {len(rep.get('missing_gt') or [])}")
        lines.append(f"- **Missing degraded:** {len(rep.get('missing_degraded') or [])}")
        lines.append(f"- **Corrupted:** {len(rep.get('corrupted') or [])}")
        lines.append(f"- **Dimension mismatches:** {len(rep.get('dimension_mismatch') or [])}")
        lines.append(f"- **Duplicate degraded (hash):** {len(rep.get('duplicate_degraded') or [])}")
        lines.append(f"- **Duplicate GT (hash):** {len(rep.get('duplicate_gt') or [])}")
        lines.append(f"- **Resolutions (top):** {rep.get('resolutions_top')}")
        lines.append(f"- **Aspect ratios (top):** {rep.get('aspect_ratios_top')}")
        lines.append(f"- **Channels:** {rep.get('channels')}")
        lines.append("")
        lines.append("### Pixel statistics (sampled)")
        lines.append("")
        lines.append("| | min | max | mean | std | brightness mean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for label, key in (("Degraded", "degraded_stats"), ("GT", "gt_stats")):
            s = rep.get(key) or {}
            if s:
                lines.append(
                    f"| {label} | {s.get('global_min', 'n/a'):.4f} | "
                    f"{s.get('global_max', 'n/a'):.4f} | "
                    f"{s.get('global_mean', 'n/a'):.4f} | "
                    f"{s.get('global_std', 'n/a'):.4f} | "
                    f"{s.get('brightness_mean', 'n/a'):.4f} |"
                )
        lines.append("")
        d = rep.get("diff_stats") or {}
        if d:
            lines.append("### GT − Degraded difference statistics")
            lines.append("")
            lines.append(f"- mean: {d.get('mean')}")
            lines.append(f"- mean abs: {d.get('mean_abs')}")
            lines.append(f"- std: {d.get('std')}")
            lines.append(f"- p95 abs: {d.get('p95_abs')}")
            lines.append(f"- p99 abs: {d.get('p99_abs')}")
            lines.append("")
    lines.append("## Visualization grid")
    lines.append("")
    if grid_path is not None:
        rel = Path(grid_path).name
        lines.append(f"![grid](../results/dataset_analysis/{rel})")
        lines.append("")
        lines.append("Each cell: [DEGRADED | GROUND TRUTH]")
    else:
        lines.append("(no samples available to visualize)")
    lines.append("")
    lines.append("## Compatibility with LPUDC")
    lines.append("")
    if any_stop:
        lines.append(
            "**STOP.** At least one split is missing or is not paired "
            "degraded→GT image data. LPUDC as implemented in the paper "
            "requires a supervised paired image restoration setup. "
            "Do not force incompatible data into the model."
        )
    else:
        lines.append(
            "All requested splits contain paired degraded/GT images. "
            "You may proceed to training with `scripts/train_custom.py` "
            "or `train.py --config configs/custom_dataset.yaml`."
        )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze a paired restoration dataset")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out-md", type=str, default="docs/DATASET_REPORT.md")
    p.add_argument("--out-dir", type=str, default="results/dataset_analysis")
    p.add_argument("--max-stats", type=int, default=300)
    p.add_argument("--max-viz", type=int, default=16)
    args = p.parse_args()

    root: Path | None = None
    degraded_dirname, gt_dirname = "degraded", "gt"
    if args.config:
        cfg = load_config(args.config)
        dcfg = cfg.get("dataset", cfg)
        root = Path(dcfg.get("root") or dcfg.get("dataset_root") or ".")
        degraded_dirname = str(dcfg.get("degraded_dirname", "degraded"))
        gt_dirname = str(dcfg.get("gt_dirname", "gt"))
    if args.root:
        root = Path(args.root)
    if root is None:
        raise SystemExit("Provide --config or --root")

    report_by_split: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        print(f"Analyzing split={split} ...")
        rep = analyze_split(
            root,
            split,
            degraded_dirname=degraded_dirname,
            gt_dirname=gt_dirname,
            max_images_stats=args.max_stats,
            max_viz=args.max_viz,
        )
        # Remove non-serializable viz samples before JSON dump.
        viz = rep.pop("viz_samples", [])
        report_by_split[split] = rep
        rep["_viz"] = viz
        print(
            f"  paired={rep.get('num_paired')} ok={rep.get('paired_ok')} "
            f"reason={rep.get('stop_reason')}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON without viz arrays
    json_dump = {
        k: {kk: vv for kk, vv in v.items() if kk != "_viz"}
        for k, v in report_by_split.items()
    }
    (out_dir / "stats.json").write_text(
        json.dumps(json_dump, indent=2, default=str), encoding="utf-8"
    )
    grid_path = draw_grid(
        {k: {"viz_samples": v.get("_viz", [])} for k, v in report_by_split.items()},
        out_dir / "viz_grid.png",
    )
    md_path = write_markdown(report_by_split, root, Path(args.out_md), grid_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {out_dir / 'stats.json'}")
    if grid_path:
        print(f"Wrote {grid_path}")

    # Hard stop if incompatible
    bad = [s for s, r in report_by_split.items() if r.get("paired_ok") is False]
    if bad:
        print("\n" + "=" * 60)
        print("STOP: dataset is NOT suitable paired restoration data for splits:", bad)
        print("LPUDC requires degraded→GT paired images. Fix the dataset layout")
        print("before training. See docs/DATASET_REPORT.md for details.")
        print("=" * 60)
        sys.exit(2)


if __name__ == "__main__":
    main()
