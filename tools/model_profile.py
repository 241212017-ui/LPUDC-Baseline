#!/usr/bin/env python
"""Model complexity profiler for LPUDC.

Reports automatically:
    - Total parameters
    - Trainable parameters
    - FLOPs (if thop / fvcore available; otherwise a clear message)
    - Model size on disk (state_dict bytes)
    - Inference latency (measured, NEVER the paper's 0.07s unless reproduced)
    - Peak GPU memory (if CUDA available)

Usage:
    python tools/model_profile.py --config configs/lpudc_poled.yaml
    python tools/model_profile.py --config configs/lpudc_poled.yaml --image-size 1024 2048
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.lpudc import build_lpudc
from utils import count_parameters, load_config, print_device_info, resolve_device, set_seed


def try_flops(model: torch.nn.Module, example: torch.Tensor) -> float | None:
    """Attempt FLOPs estimation via thop or fvcore."""
    try:
        from thop import profile  # type: ignore

        model_cpu = model
        ex = example
        flops, _ = profile(model_cpu, inputs=(ex,), verbose=False)
        return float(flops)
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        analysis = FlopCountAnalysis(model, example)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return float(analysis.total())
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Profile LPUDC model complexity")
    p.add_argument("--config", type=str, default="configs/lpudc_poled.yaml")
    p.add_argument("--image-size", type=int, nargs=2, default=[1024, 2048],
                   metavar=("H", "W"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = resolve_device(args.device)
    info = print_device_info(device)

    model = build_lpudc(cfg.get("model", {})).to(device).eval()
    total, trainable = count_parameters(model)

    h, w = args.image_size
    example = torch.randn(args.batch_size, 3, h, w, device=device)

    # FLOPs
    flops = try_flops(model.cpu(), example.cpu()) if device.type == "cpu" else try_flops(model, example)
    model = model.to(device)

    # Model size (state_dict)
    state = model.state_dict()
    nbytes = sum(v.numel() * v.element_size() for v in state.values())
    size_mb = nbytes / (1024 ** 2)

    # Latency
    times: list[float] = []
    peak_mem = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(args.iters):
            t0 = time.perf_counter()
            _ = model(example)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    latency_mean = sum(times) / max(1, len(times))
    latency_std = (
        (sum((t - latency_mean) ** 2 for t in times) / max(1, len(times))) ** 0.5
    )

    report = {
        "config": args.config,
        "image_size": [h, w],
        "batch_size": args.batch_size,
        "device": str(device),
        "gpu": info.get("GPU"),
        "params_total": total,
        "params_trainable": trainable,
        "params_M": total / 1e6,
        "model_size_MB": size_mb,
        "flops": flops,
        "flops_G": (flops / 1e9) if flops else None,
        "latency_mean_s": latency_mean,
        "latency_std_s": latency_std,
        "latency_iters": args.iters,
        "peak_gpu_mem_GB": peak_mem,
        "note": (
            "Latency and peak memory are OUR measurements on THIS machine. "
            "The paper reports ~0.07s on an NVIDIA RTX 3090 for 1024x2048; "
            "do not treat our number as a reproduction of that claim."
        ),
        "paper_reference_not_used_for_scoring": {
            "params_M": 4.0,
            "time_s": 0.07,
        },
    }

    print("\n===== MODEL PROFILE =====")
    print(f"  Parameters (total)   : {total:,} ({total/1e6:.3f}M)")
    print(f"  Parameters (train.)  : {trainable:,} ({trainable/1e6:.3f}M)")
    print(f"  Model size           : {size_mb:.2f} MB (state_dict)")
    print(f"  FLOPs                : {report['flops_G']} G" if flops else "  FLOPs                : unavailable (install `thop` or `fvcore`)")
    print(f"  Latency (mean±std)   : {latency_mean*1000:.2f} ± {latency_std*1000:.2f} ms  [{args.iters} iters, batch={args.batch_size}, {h}x{w}]")
    if peak_mem is not None:
        print(f"  Peak GPU memory      : {peak_mem:.3f} GB")
    else:
        print("  Peak GPU memory      : N/A (CPU)")
    print("  NOTE: paper reports ~4.0M params / 0.07s on RTX 3090 — reference only.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
