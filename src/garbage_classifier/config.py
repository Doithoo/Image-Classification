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


@dataclass
class ModelConfig:
    """Model selection."""

    name: str = "resnet50"  # key of the model registry (timm model or legacy_* name)
    num_classes: int | None = None  # derived from the manifest unless explicitly asserted
    pretrained: bool = True


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
    mixup_alpha: float = 0.0  # >0 enables MixUp (soft-target augmentation)
    cutmix_alpha: float = 0.0  # >0 enables CutMix (ignored if mixup_alpha > 0)
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


def _from_dict(cls: type[Any], data: dict[str, Any], section: str) -> Any:
    """Build a dataclass from a dict, rejecting misspelled fields."""
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        key = sorted(unknown)[0]
        raise ValueError(f"unknown config key: {section}.{key}")
    return cls(**data)


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
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping, got {raw!r}")

    cfg = ExperimentConfig()
    for section, values in raw.items():
        if section in ("data", "model", "train") and isinstance(values, dict):
            setattr(cfg, section, _from_dict(getattr(cfg, section).__class__, values, section))
        elif section in ("device", "output_dir", "run_name", "log_level"):
            setattr(cfg, section, values)
        else:
            raise ValueError(f"unknown config section: {section!r}")

    if overrides:
        _apply_overrides(cfg, overrides)
    _validate_config(cfg)
    return cfg


def _apply_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) == 1 and parts[0] in {"device", "output_dir", "run_name", "log_level"}:
            setattr(cfg, parts[0], value)
            continue
        if len(parts) != 2 or parts[0] not in {"data", "model", "train"}:
            raise ValueError(f"unknown config key: {dotted_key}")
        obj = getattr(cfg, parts[0])
        if parts[1] not in {f.name for f in fields(obj)}:
            raise ValueError(f"unknown config key: {dotted_key}")
        setattr(obj, parts[1], value)


def _fail(key: str, value: Any, requirement: str) -> None:
    raise ValueError(f"invalid {key}={value!r}; expected {requirement}")


def _require_type(key: str, value: Any, expected: type[Any]) -> None:
    if type(value) is not expected:
        _fail(key, value, expected.__name__)


def _require_number(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(key, value, "a number")
    return float(value)


def _require_enum(key: str, value: Any, allowed: tuple[str, ...]) -> None:
    _require_type(key, value, str)
    if value not in allowed:
        _fail(key, value, f"one of: {', '.join(allowed)}")


def _require_int_range(key: str, value: Any, minimum: int, *, inclusive: bool = True) -> None:
    _require_type(key, value, int)
    if value < minimum if inclusive else value <= minimum:
        operator = ">=" if inclusive else ">"
        _fail(key, value, f"an integer {operator} {minimum}")


def _validate_triplet(key: str, value: Any, *, positive: bool) -> None:
    if not isinstance(value, list) or len(value) != 3:
        _fail(key, value, "a list of three numbers")
    numbers = [_require_number(key, item) for item in value]
    if any(item <= 0 if positive else not 0 <= item <= 1 for item in numbers):
        interval = "(0, 1]" if positive else "[0, 1]"
        _fail(key, value, f"three values in {interval}")
    if positive and any(item > 1 for item in numbers):
        _fail(key, value, "three values in (0, 1]")


def _validate_config(cfg: ExperimentConfig) -> None:
    """Validate the fully resolved configuration in one place."""
    for key, value in (
        ("data.data_dir", cfg.data.data_dir),
        ("data.manifest_dir", cfg.data.manifest_dir),
        ("model.name", cfg.model.name),
        ("output_dir", cfg.output_dir),
        ("train.best_metric", cfg.train.best_metric),
    ):
        _require_type(key, value, str)
        if not value:
            _fail(key, value, "a non-empty string")
    if cfg.run_name is not None:
        _require_type("run_name", cfg.run_name, str)

    for key, value in (
        ("data.seed", cfg.data.seed),
        ("train.seed", cfg.train.seed),
    ):
        _require_type(key, value, int)
    for key, value, minimum in (
        ("data.image_size", cfg.data.image_size, 1),
        ("data.resize_size", cfg.data.resize_size, 1),
        ("data.num_workers", cfg.data.num_workers, 0),
        ("train.epochs", cfg.train.epochs, 1),
        ("train.batch_size", cfg.train.batch_size, 1),
        ("train.warmup_epochs", cfg.train.warmup_epochs, 0),
        ("train.early_stop_patience", cfg.train.early_stop_patience, 0),
    ):
        _require_int_range(key, value, minimum)
    if cfg.model.num_classes is not None:
        _require_int_range("model.num_classes", cfg.model.num_classes, 1)
    if cfg.data.resize_size < cfg.data.image_size:
        _fail("data.resize_size", cfg.data.resize_size, f">= data.image_size ({cfg.data.image_size})")
    for key, value in (
        ("data.pin_memory", cfg.data.pin_memory),
        ("model.pretrained", cfg.model.pretrained),
        ("train.amp", cfg.train.amp),
        ("train.ema", cfg.train.ema),
    ):
        _require_type(key, value, bool)

    if not isinstance(cfg.data.classes, list) or not cfg.data.classes:
        _fail("data.classes", cfg.data.classes, "a non-empty list of unique class names")
    if any(type(name) is not str or not name for name in cfg.data.classes) or len(set(cfg.data.classes)) != len(
        cfg.data.classes
    ):
        _fail("data.classes", cfg.data.classes, "a non-empty list of unique class names")

    _validate_triplet("data.normalize_mean", cfg.data.normalize_mean, positive=False)
    _validate_triplet("data.normalize_std", cfg.data.normalize_std, positive=True)
    if not isinstance(cfg.data.split_ratios, list) or len(cfg.data.split_ratios) != 3:
        _fail("data.split_ratios", cfg.data.split_ratios, "three non-negative numbers summing to 1")
    ratios = [_require_number("data.split_ratios", item) for item in cfg.data.split_ratios]
    if any(item < 0 for item in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        _fail("data.split_ratios", cfg.data.split_ratios, "three non-negative numbers summing to 1")

    _require_enum("data.aug", cfg.data.aug, ("basic", "randaug"))
    _require_enum("train.optimizer", cfg.train.optimizer, ("adamw", "sgd", "lion"))
    _require_enum("train.scheduler", cfg.train.scheduler, ("cosine", "step", "none"))
    _require_enum("train.class_weight", cfg.train.class_weight, ("none", "inverse", "effective"))
    _require_enum("train.sampler", cfg.train.sampler, ("none", "weighted"))
    _require_enum("device", cfg.device, ("auto", "cpu", "cuda", "mps"))
    _require_enum("log_level", cfg.log_level, ("debug", "info", "warning", "error", "critical"))

    numeric_ranges = (
        ("train.lr", cfg.train.lr, 0.0, None, False, True),
        ("train.weight_decay", cfg.train.weight_decay, 0.0, None, True, True),
        ("train.momentum", cfg.train.momentum, 0.0, 1.0, True, False),
        ("train.label_smoothing", cfg.train.label_smoothing, 0.0, 1.0, True, False),
        ("train.mixup_alpha", cfg.train.mixup_alpha, 0.0, None, True, True),
        ("train.cutmix_alpha", cfg.train.cutmix_alpha, 0.0, None, True, True),
        ("train.grad_clip", cfg.train.grad_clip, 0.0, None, True, True),
        ("train.ema_decay", cfg.train.ema_decay, 0.0, 1.0, True, False),
    )
    for key, value, low, high, include_low, include_high in numeric_ranges:
        number = _require_number(key, value)
        if number < low or (number == low and not include_low):
            _fail(key, value, f">{'=' if include_low else ''} {low}")
        if high is not None and (number > high or (number == high and not include_high)):
            _fail(key, value, f"<{'=' if include_high else ''} {high}")


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Serialize config to a plain dict (for YAML dumps and checkpoint metadata)."""
    return dataclasses.asdict(cfg)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
