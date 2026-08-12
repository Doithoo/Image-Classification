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

from ..config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, load_config, to_dict
from ..utils import git_revision


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    deployable_state_dict: dict[str, torch.Tensor] | None = None,
    ema: Any | None = None,
    scaler: Any | None = None,
    patience_left: int | None = None,
    epoch: int,
    best_metric: float,
    cfg: ExperimentConfig,
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a full checkpoint (weights + state + metadata)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    training_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    deployable_state = deployable_state_dict or training_state
    payload: dict[str, Any] = {
        # model_state_dict remains the inference-facing alias for old consumers.
        "model_state_dict": deployable_state,
        "training_model_state_dict": training_state,
        "deployable_model_state_dict": deployable_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "patience_left": patience_left,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": to_dict(cfg),
        "class_names": class_names,
        "git_revision": git_revision(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
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
    required = {"config", "class_names"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint {p} is missing metadata: {sorted(missing)}")
    state = deployable_model_state(payload)
    payload.setdefault("model_state_dict", state)
    payload.setdefault("training_model_state_dict", state)
    payload.setdefault("deployable_model_state_dict", state)
    payload.setdefault("ema_state_dict", None)
    payload.setdefault("scaler_state_dict", None)
    payload.setdefault("patience_left", None)
    return payload


def restore_config_from_checkpoint(payload: dict[str, Any]) -> ExperimentConfig:
    """Rebuild the resolved experiment config stored in a checkpoint."""
    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint is missing config metadata")
    cfg = load_config()
    classes = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig}
    for section, cls in classes.items():
        values = raw.get(section)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"checkpoint config section {section!r} must be a mapping")
        valid = cls.__dataclass_fields__
        setattr(cfg, section, cls(**{key: value for key, value in values.items() if key in valid}))
    for key in ("device", "output_dir", "run_name", "log_level"):
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg


def deployable_model_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Select inference weights, falling back to the legacy state-dict key."""
    state = payload.get("deployable_model_state_dict")
    if state is None:
        state = payload.get("model_state_dict")
    if state is None:
        raise ValueError("checkpoint is missing deployable model state")
    return state
