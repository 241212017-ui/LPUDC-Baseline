#!/usr/bin/env python
"""evaluate.py - evaluate an LPUDC checkpoint and optionally write a
reproduction report comparing OUR measured values against paper-reported
reference numbers.

Research integrity:
    - Paper numbers are REFERENCE ONLY and never used to score the model.
    - Our metrics come exclusively from test.py's measurement pipeline.
    - Differences are reported honestly; no fabrication.

Usage:
    python evaluate.py --config configs/lpudc_poled.yaml \
        --checkpoint checkpoints/lpudc_poled/best_psnr.pth --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import load_config

# Paper-reported LPUDC results (REFERENCE ONLY - Ding et al., Neurocomputing
# 655 (2025) 131442, Tables 1-2). NEVER hardcoded into scoring code.
PAPER_RESULTS: dict[str, dict[str, Any]] = {
    "POLED": {
        "PSNR": 33.99,
        "SSIM": 0.9629,
        "LPIPS": 0.1678,
        "DISTS": 0.1399,
        "params_M": 4.0,
        "time_s": 0.07,
    },
    "TOLED": {
        "PSNR": 38.62,
        "SSIM": 0.9805,
        "LPIPS": 0.0893,
        "DISTS": 0.0858,
        "params_M": 4.0,
        "time_s": 0.07,
    },
}


def build_reproduction_report(
    our_summary: dict[str, Any],
    dataset_name: str,
    out_path: Path,
    params_m: float | None = None,
) -> Path:
    """Write docs/REPRODUCTION_REPORT.md (paper vs our measured values)."""
    paper = PAPER_RESULTS.get(dataset_name.upper())
    our = our_summary.get("metrics", {})
    lines: list[str] = []
    lines.append("# LPUDC Reproduction Report")
    lines.append("")
    lines.append(f"- **Dataset:** {dataset_name}")
    lines.append(f"- **Split:** {our_summary.get('split')}")
    lines.append(f"- **Images evaluated:** {our_summary.get('num_images')}")
    lines.append(f"- **Sec/image (ours):** {our_summary.get('seconds_per_image')}")
    lines.append("")
    lines.append("## Direction of metrics")
    lines.append("")
    lines.append("| Metric | Better when |")
    lines.append("|--------|-------------|")
    lines.append("| PSNR | higher |")
    lines.append("| SSIM | higher |")
    lines.append("| LPIPS | lower |")
    lines.append("| DISTS | lower |")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Metric | Paper (reference) | Our result | Absolute difference |")
    lines.append("|--------|-------------------:|-----------:|--------------------:|")
    if paper is None:
        lines.append(
            "| (all) | No paper entry for this dataset name | see metrics.json | n/a |"
        )
    else:
        for key in ("PSNR", "SSIM", "LPIPS", "DISTS"):
            p = paper.get(key)
            o = our.get(key)
            if p is None or o is None:
                diff = "n/a (missing)"
            else:
                diff = f"{o - p:+.4f}"
            lines.append(f"| {key} | {p} | {o} | {diff} |")
        our_time = our_summary.get("seconds_per_image")
        if our_time is not None and paper.get("time_s") is not None:
            tdiff = float(our_time) - float(paper["time_s"])
            lines.append(
                f"| Time (s/img) | {paper['time_s']} | {our_time:.4f} | {tdiff:+.4f} |"
            )
        else:
            lines.append(f"| Time (s/img) | {paper.get('time_s')} | {our_time} | n/a |")
        if params_m is not None:
            lines.append(
                f"| Params (M) | {paper.get('params_M')} | {params_m:.3f} | "
                f"{params_m - float(paper.get('params_M', 0)):+.3f} |"
            )
        else:
            lines.append(
                f"| Params (M) | {paper.get('params_M')} | (see model_profile) | n/a |"
            )
    lines.append("")
    lines.append("## Classification of values")
    lines.append("")
    lines.append("- **PAPER FACT:** the \"Paper (reference)\" column, from")
    lines.append("  Ding et al., Neurocomputing 655 (2025) 131442, Tables 1-2.")
    lines.append("- **OUR EXPERIMENTAL RESULT:** the \"Our result\" column, measured")
    lines.append("  by `test.py` on our checkpoint.")
    lines.append("- **IMPLEMENTATION ASSUMPTION:** architecture widths and unspecified")
    lines.append("  training details are documented in `docs/PAPER_SPECIFICATION.md`.")
    lines.append("")
    lines.append("## Possible reasons for difference")
    lines.append("")
    lines.append("1. **Missing implementation details.** The paper does not specify channel")
    lines.append("   widths, number of DRBs, encoder/decoder depths, inverse pixel-shuffle")
    lines.append("   factor, or the exact DGF network layout. Our choices are documented")
    lines.append("   assumptions and will not match the authors' private code exactly.")
    lines.append("2. **Laplacian pyramid kernels.** The paper says \"fixed-kernel weighted")
    lines.append("   average\" but does not give the kernel. We use the classical 5x5")
    lines.append("   binomial kernel.")
    lines.append("3. **Loss reduction.** The paper writes L1 as a sum over pixels; we use")
    lines.append("   mean L1 for scale stability.")
    lines.append("4. **Training schedule details.** Cosine T_max definition, weight decay,")
    lines.append("   initialization, and validation protocol are unspecified.")
    lines.append("5. **Dataset files / split layout.** POLED/TOLED release folder layouts")
    lines.append("   vary; our loader matches by file stem.")
    lines.append("6. **Hardware / precision.** Paper: single RTX 3090, no mention of AMP.")
    lines.append("   Our run hardware and precision settings may differ.")
    lines.append("7. **Randomness.** The paper does not report a seed.")
    lines.append("8. **Metric implementations.** LPIPS backbone and DISTS version are not")
    lines.append("   stated; slight metric-implementation differences are expected.")
    lines.append("9. **Training length.** Reproducing 300K full-resolution iterations")
    lines.append("   requires substantial GPU time; shorter runs will not match.")
    lines.append("")
    lines.append("## Integrity statement")
    lines.append("")
    lines.append("This report does **not** claim exact reproduction. Paper numbers are")
    lines.append("reference targets only and are never hardcoded into evaluation code.")
    lines.append("If our measured values differ, the difference is reported honestly above.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate LPUDC + optional reproduction report")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--report-path", type=str, default="docs/REPRODUCTION_REPORT.md")
    p.add_argument("--no-report", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(
        args.output_dir
        or cfg.get("experiment", {}).get("output_dir", "results/lpudc")
    )

    # Run the full measurement pipeline via test.py
    import subprocess

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "test.py"),
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
        "--output-dir",
        str(out_dir),
        "--save-visualizations",
        "--save-pyramid",
    ]
    if args.device:
        cmd += ["--device", args.device]
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        print(f"ERROR: metrics not found at {metrics_path}")
        sys.exit(1)

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    ds_name = str(cfg.get("dataset", {}).get("name", "unknown"))

    # Optional params measurement
    params_m = None
    try:
        from models.lpudc import build_lpudc
        from utils import count_parameters

        model = build_lpudc(cfg.get("model", {}))
        total, _ = count_parameters(model)
        params_m = total / 1e6
    except Exception:
        pass

    if not args.no_report:
        report_path = Path(args.report_path)
        if not report_path.is_absolute():
            report_path = _REPO_ROOT / report_path
        build_reproduction_report(summary, ds_name, report_path, params_m=params_m)
        print(f"Wrote {report_path}")

    print("\n===== MEASURED SUMMARY (NOT paper values) =====")
    m = summary.get("metrics", {})
    for k, v in m.items():
        direction = {
            "PSNR": "higher is better",
            "SSIM": "higher is better",
            "LPIPS": "lower is better",
            "DISTS": "lower is better",
        }.get(k, "")
        print(f"  {k:6s}: {v}   [{direction}]")
    if params_m is not None:
        print(f"  Params: {params_m:.3f}M  (paper reference: 4.0M)")


if __name__ == "__main__":
    main()
