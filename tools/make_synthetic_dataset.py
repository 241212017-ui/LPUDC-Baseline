#!/usr/bin/env python
"""Generate a tiny synthetic paired dataset for smoke / overfit tests.

Creates:
    data/synthetic_tiny/{train,val,test}/{degraded,gt}/*.png

Synthetic "degradation": brightness attenuation + color shift + mild blur,
mimicking the low-frequency UDC effects described in the paper (for testing
the pipeline only — NOT a real UDC dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def make_pair(rng: np.random.Generator, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = size
    # Smooth random GT image (low-frequency dominant + some texture)
    base = rng.random((h // 8 + 1, w // 8 + 1, 3)).astype(np.float32)
    gt_small = np.array(
        Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    texture = 0.15 * rng.random((h, w, 3)).astype(np.float32)
    gt = np.clip(gt_small + texture, 0.0, 1.0)

    # Synthetic UDC-like degradation: attenuate + color shift + blur
    gamma = float(rng.uniform(0.45, 0.7))  # brightness attenuation
    color_shift = rng.uniform(-0.08, 0.08, size=(3,)).astype(np.float32)
    degraded = np.clip(gt * gamma + color_shift, 0.0, 1.0)
    img = Image.fromarray((degraded * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))
    degraded = np.asarray(img, dtype=np.float32) / 255.0
    return degraded, gt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data/synthetic_tiny")
    p.add_argument("--num-train", type=int, default=8)
    p.add_argument("--num-val", type=int, default=2)
    p.add_argument("--num-test", type=int, default=2)
    p.add_argument("--size", type=int, nargs=2, default=[64, 64])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    h, w = args.size

    counts = {"train": args.num_train, "val": args.num_val, "test": args.num_test}
    for split, n in counts.items():
        for sub in ("degraded", "gt"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            deg, gt = make_pair(rng, (h, w))
            Image.fromarray((deg * 255).astype(np.uint8)).save(
                root / split / "degraded" / f"{split}_{i:03d}.png"
            )
            Image.fromarray((gt * 255).astype(np.uint8)).save(
                root / split / "gt" / f"{split}_{i:03d}.png"
            )
        print(f"{split}: {n} pairs -> {root / split}")


if __name__ == "__main__":
    main()
