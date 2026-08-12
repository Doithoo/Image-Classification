"""Checkpoint management: save/load full run state for reproducibility.

A checkpoint is self-contained: it stores model weights, optimizer and scheduler
state, class mapping, preprocessing statistics, the full resolved config, the git
revision and RNG states. Inference reads everything it needs from the checkpoint,
eliminating train/inference configuration drift.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig, to_dict
from ..utils import git_revision


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int,
    best_metric: float,
    cfg: ExperimentConfig,
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a full checkpoint (weights + state + metadata)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": to_dict(cfg),
        "class_names": class_names,
        "git_revision": git_revision(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint dict (weights on CPU; caller moves them)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "config", "class_names"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint {p} is missing metadata: {sorted(missing)}")
    return payload
