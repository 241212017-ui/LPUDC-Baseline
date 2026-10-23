"""Shared utilities for LPUDC: config, device, seed, logging, checkpointing."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return cfg


def save_config(cfg: dict[str, Any], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cudnn can slow things down; enable only if requested.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str = "auto") -> torch.device:
    device_str = (device_str or "auto").lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def print_device_info(device: torch.device) -> dict[str, Any]:
    """Print Device / GPU / VRAM as required by the research brief."""
    info: dict[str, Any] = {
        "Device": str(device),
        "GPU": "N/A",
        "VRAM": "N/A",
        "CUDA_available": torch.cuda.is_available(),
        "PyTorch": torch.__version__,
        "Python": sys.version.split()[0],
    }
    if torch.cuda.is_available() and device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        info["GPU"] = props.name
        info["VRAM"] = f"{total_gb:.2f} GB total, {allocated_gb:.2f} GB allocated"
        info["CUDA_version"] = torch.version.cuda
    elif device.type == "cpu":
        info["GPU"] = "None (CPU mode)"
        info["VRAM"] = "N/A"
    print("=" * 60)
    for k, v in info.items():
        print(f"{k:16s}: {v}")
    print("=" * 60)
    return info


def ensure_dirs(*paths: str | os.PathLike) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    iteration: int = 0,
    best_metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "iteration": iteration,
        "best_metrics": best_metrics or {},
        "config": config or {},
        "torch_version": torch.__version__,
        "saved_at": time.time(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


class AverageMeter:
    """Tracks a running average of scalar values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def append_jsonl(path: str | os.PathLike, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_unique_params_approx(model: torch.nn.Module) -> float:
    """Parameter count in millions (informational)."""
    total, _ = count_parameters(model)
    return total / 1e6
