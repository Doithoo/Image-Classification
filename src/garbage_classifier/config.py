"""Configuration system: typed dataclasses loaded from YAML with sensible defaults.

A config file may override any subset of the defaults. The full resolved config is
saved into every experiment artifact, so runs are reproducible from the artifact
alone (see ``garbage_classifier.training.checkpoint``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

GARBAGE_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@dataclass
class DataConfig:
    """Dataset layout and preprocessing."""

    data_dir: str = "data"  # root that contains the class folders (raw split by class)
    manifest_dir: str = "data/manifests"  # where generated CSV manifests live
    classes: list[str] = field(default_factory=lambda: list(GARBAGE_CLASSES))
    image_size: int = 224
    resize_size: int = 256
    normalize_mean: list[float] = field(default_factory=lambda: [0.673, 0.639, 0.604])
    normalize_std: list[float] = field(default_factory=lambda: [0.208, 0.209, 0.231])
    split_ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 666
    num_workers: int = 4
    pin_memory: bool = True
    aug: str = "basic"  # basic | randaug
    mixup_alpha: float = 0.0  # >0 enables MixUp
    cutmix_alpha: float = 0.0  # >0 enables CutMix (only if mixup_alpha == 0)


@dataclass
class ModelConfig:
    """Model selection."""

    name: str = "resnet50"  # key of the model registry (timm: <timm-model>, legacy: <name>)
    source: str = "timm"  # timm | legacy | torchvision
    num_classes: int = 6
    pretrained: bool = True
    timm_backbone: str | None = None  # explicit timm name, overrides ``name`` if set


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"  # adamw | sgd | lion
    scheduler: str = "cosine"  # cosine | step | none
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    grad_clip: float = 1.0  # 0 disables
    amp: bool = True  # mixed precision (cuda/mps)
    early_stop_patience: int = 15  # epochs without val improvement before stopping
    best_metric: str = "macro_f1"  # metric used for best-checkpoint selection
    ema: bool = False  # exponential moving average of weights
    ema_decay: float = 0.999
    class_weight: str = "none"  # none | inverse | effective  (class-imbalance ablation)
    sampler: str = "none"  # none | weighted  (class-imbalance ablation)
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level config for a run (training / evaluation / inference)."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"  # auto | cpu | cuda | mps
    output_dir: str = "artifacts"  # root for run artifacts
    run_name: str | None = None  # default: <model>-<timestamp>
    log_level: str = "info"


def _from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys (forward compatible)."""
    valid = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config, overlaying optional dict overrides.

    ``overrides`` uses dotted keys, e.g. ``{"train.lr": 1e-4, "model.name": "swin_tiny"}``.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}

    cfg = ExperimentConfig()
    for section, values in raw.items():
        if section in ("data", "model", "train") and isinstance(values, dict):
            setattr(cfg, section, _from_dict(getattr(cfg, section).__class__, values))
        elif section in ("device", "output_dir", "run_name", "log_level"):
            setattr(cfg, section, values)
        else:
            raise ValueError(f"unknown config section: {section!r}")

    if overrides:
        _apply_overrides(cfg, overrides)
    return cfg


def _apply_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Serialize config to a plain dict (for YAML dumps and checkpoint metadata)."""
    return dataclasses.asdict(cfg)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
