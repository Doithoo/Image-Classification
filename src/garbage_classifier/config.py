"""Typed, validated configuration for reproducible classification runs."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

GARBAGE_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

SUPPORTED_BEST_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "weighted_precision",
    "weighted_recall",
)


@dataclass(frozen=True)
class DataConfig:
    """Dataset layout and image preprocessing."""

    data_dir: Path = Path("data/raw")
    manifest_dir: Path = Path("data/manifests")
    image_size: int = 224
    resize_size: int = 256
    normalize_mean: list[float] = field(default_factory=lambda: [0.673, 0.639, 0.604])
    normalize_std: list[float] = field(default_factory=lambda: [0.208, 0.209, 0.231])
    split_ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 666
    num_workers: int = 0
    pin_memory: bool = True
    aug: str = "basic"
    interpolation: str = "bilinear"


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and explicit model-specific options."""

    name: str = "resnet18"
    num_classes: int | None = None
    pretrained: bool = False
    preprocessing: str = "fixed"  # fixed | model_default
    params: dict[str, Any] = field(default_factory=dict)
    factory: str | None = None  # optional trusted ``module:function`` factory


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters; advanced settings default to the simple path."""

    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    grad_clip: float = 0.0
    amp: bool = False
    early_stop_patience: int = 5  # 0 disables early stopping
    best_metric: str = "macro_f1"
    ema: bool = False
    ema_decay: float = 0.999
    class_weight: str = "none"
    sampler: str = "none"
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully resolved configuration shared by training, evaluation and inference."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"
    output_dir: Path = Path("artifacts")
    run_name: str | None = None
    log_level: str = "info"


class ConfigError(ValueError):
    """Raised when a configuration cannot be parsed or validated."""


def _as_path(key: str, value: Any) -> Path:
    if isinstance(value, str) and not value:
        _fail(key, value, "a non-empty path")
    try:
        return Path(value)
    except TypeError as exc:
        raise ConfigError(f"invalid {key}={value!r}; expected a path") from exc


def _from_dict(cls: type[Any], data: Mapping[str, Any], section: str) -> Any:
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        key = sorted(unknown)[0]
        raise ConfigError(f"unknown config key: {section}.{key}")
    values = dict(data)
    if cls is DataConfig:
        for key in ("data_dir", "manifest_dir"):
            if key in values:
                values[key] = _as_path(f"data.{key}", values[key])
    if cls is ModelConfig and "params" in values and not isinstance(values["params"], Mapping):
        raise ConfigError("model.params must be a mapping")
    return cls(**values)


def config_from_dict(values: Mapping[str, Any], *, allow_legacy: bool = False) -> ExperimentConfig:
    """Build a validated config from a serialized mapping, including old checkpoints."""
    raw = {key: value for key, value in values.items()}
    if allow_legacy and isinstance(raw.get("data"), Mapping):
        data = dict(raw["data"])
        data.pop("classes", None)
        raw["data"] = data
    cfg = ExperimentConfig()
    for section, value in raw.items():
        if section == "data":
            if not isinstance(value, Mapping):
                raise ConfigError("config section 'data' must be a mapping")
            cfg = replace(cfg, data=_from_dict(DataConfig, value, section))
        elif section == "model":
            if not isinstance(value, Mapping):
                raise ConfigError("config section 'model' must be a mapping")
            cfg = replace(cfg, model=_from_dict(ModelConfig, value, section))
        elif section == "train":
            if not isinstance(value, Mapping):
                raise ConfigError("config section 'train' must be a mapping")
            cfg = replace(cfg, train=_from_dict(TrainConfig, value, section))
        elif section == "device":
            cfg = replace(cfg, device=value)
        elif section == "output_dir":
            cfg = replace(cfg, output_dir=_as_path("output_dir", value))
        elif section == "run_name":
            cfg = replace(cfg, run_name=value)
        elif section == "log_level":
            cfg = replace(cfg, log_level=value)
        else:
            raise ConfigError(f"unknown config section: {section!r}")
    _validate_config(cfg)
    return cfg


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Load defaults, an optional YAML file, then dotted-key overrides."""
    raw: Mapping[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"config file not found: {config_path}")
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"config root must be a mapping, got {loaded!r}")
        raw = loaded

    cfg = config_from_dict(raw)

    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    _validate_config(cfg)
    return cfg


def load_config_with_sources(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[ExperimentConfig, dict[str, str]]:
    """Resolve config and identify whether each leaf came from default, YAML or CLI."""
    cfg = load_config(path, overrides)
    sources = {key: "default" for key in _leaf_paths(to_dict(ExperimentConfig()))}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(loaded, Mapping):
            for key in _leaf_paths(loaded):
                sources[key] = "yaml"
    for key in overrides or {}:
        sources[key] = "cli"
    return cfg, dict(sorted(sources.items()))


def _apply_overrides(cfg: ExperimentConfig, overrides: Mapping[str, Any]) -> ExperimentConfig:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) == 1:
            if parts[0] == "device":
                cfg = replace(cfg, device=value)
            elif parts[0] == "output_dir":
                cfg = replace(cfg, output_dir=_as_path("output_dir", value))
            elif parts[0] == "run_name":
                cfg = replace(cfg, run_name=value)
            elif parts[0] == "log_level":
                cfg = replace(cfg, log_level=value)
            else:
                raise ConfigError(f"unknown config key: {dotted_key}")
            continue
        if len(parts) == 2 and parts[0] == "data":
            if parts[1] not in {field.name for field in fields(DataConfig)}:
                raise ConfigError(f"unknown config key: {dotted_key}")
            if parts[1] in {"data_dir", "manifest_dir"}:
                value = _as_path(f"data.{parts[1]}", value)
            cfg = replace(cfg, data=replace(cfg.data, **{parts[1]: value}))
            continue
        if len(parts) == 2 and parts[0] == "model":
            if parts[1] not in {field.name for field in fields(ModelConfig)}:
                raise ConfigError(f"unknown config key: {dotted_key}")
            if parts[1] == "params" and not isinstance(value, Mapping):
                raise ConfigError("model.params must be a mapping")
            cfg = replace(cfg, model=replace(cfg.model, **{parts[1]: value}))
            continue
        if len(parts) == 2 and parts[0] == "train":
            if parts[1] not in {field.name for field in fields(TrainConfig)}:
                raise ConfigError(f"unknown config key: {dotted_key}")
            cfg = replace(cfg, train=replace(cfg.train, **{parts[1]: value}))
            continue
        if len(parts) == 3 and parts[:2] == ["model", "params"] and parts[2]:
            params = dict(cfg.model.params)
            params[parts[2]] = value
            cfg = replace(cfg, model=replace(cfg.model, params=params))
            continue
        raise ConfigError(f"unknown config key: {dotted_key}")
    return cfg


def _leaf_paths(values: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return tuple(paths)


def _fail(key: str, value: Any, requirement: str) -> None:
    raise ConfigError(f"invalid {key}={value!r}; expected {requirement}")


def _require_type(key: str, value: Any, expected: type[Any]) -> None:
    if type(value) is not expected:
        _fail(key, value, expected.__name__)


def _require_number(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        _fail(key, value, "a finite number")
    return float(value)


def _require_enum(key: str, value: Any, allowed: Sequence[str]) -> None:
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
    string_fields: tuple[tuple[str, Any], ...] = (
        ("model.name", cfg.model.name),
        ("train.best_metric", cfg.train.best_metric),
    )
    for field_name, field_value in string_fields:
        _require_type(field_name, field_value, str)
        if not field_value:
            _fail(field_name, field_value, "a non-empty string")
    path_fields: tuple[tuple[str, Any], ...] = (
        ("data.data_dir", cfg.data.data_dir),
        ("data.manifest_dir", cfg.data.manifest_dir),
        ("output_dir", cfg.output_dir),
    )
    for field_name, field_value in path_fields:
        if not isinstance(field_value, Path) or not str(field_value):
            _fail(field_name, field_value, "a non-empty path")
    if cfg.run_name is not None:
        if type(cfg.run_name) is not str or not cfg.run_name.strip():
            _fail("run_name", cfg.run_name, "a non-empty string or null")
        run_path = Path(cfg.run_name)
        if run_path.is_absolute() or run_path.name != cfg.run_name or cfg.run_name in {".", ".."}:
            _fail("run_name", cfg.run_name, "one directory name without path separators")
    if cfg.model.factory is not None and (type(cfg.model.factory) is not str or ":" not in cfg.model.factory):
        _fail("model.factory", cfg.model.factory, "a module:function string or null")
    if not isinstance(cfg.model.params, dict):
        _fail("model.params", cfg.model.params, "a mapping")

    integer_fields: tuple[tuple[str, Any], ...] = (("data.seed", cfg.data.seed), ("train.seed", cfg.train.seed))
    for field_name, field_value in integer_fields:
        _require_type(field_name, field_value, int)
    ranged_integer_fields: tuple[tuple[str, Any, int], ...] = (
        ("data.image_size", cfg.data.image_size, 1),
        ("data.resize_size", cfg.data.resize_size, 1),
        ("data.num_workers", cfg.data.num_workers, 0),
        ("train.epochs", cfg.train.epochs, 1),
        ("train.batch_size", cfg.train.batch_size, 1),
        ("train.warmup_epochs", cfg.train.warmup_epochs, 0),
        ("train.early_stop_patience", cfg.train.early_stop_patience, 0),
    )
    for field_name, field_value, minimum in ranged_integer_fields:
        _require_int_range(field_name, field_value, minimum)
    if cfg.model.num_classes is not None:
        _require_int_range("model.num_classes", cfg.model.num_classes, 1)
    if cfg.data.resize_size < cfg.data.image_size:
        _fail("data.resize_size", cfg.data.resize_size, f">= data.image_size ({cfg.data.image_size})")
    if cfg.train.warmup_epochs >= cfg.train.epochs and cfg.train.warmup_epochs:
        _fail("train.warmup_epochs", cfg.train.warmup_epochs, f"< train.epochs ({cfg.train.epochs})")
    boolean_fields: tuple[tuple[str, Any], ...] = (
        ("data.pin_memory", cfg.data.pin_memory),
        ("model.pretrained", cfg.model.pretrained),
        ("train.amp", cfg.train.amp),
        ("train.ema", cfg.train.ema),
    )
    for field_name, field_value in boolean_fields:
        _require_type(field_name, field_value, bool)

    _validate_triplet("data.normalize_mean", cfg.data.normalize_mean, positive=False)
    _validate_triplet("data.normalize_std", cfg.data.normalize_std, positive=True)
    if not isinstance(cfg.data.split_ratios, list) or len(cfg.data.split_ratios) != 3:
        _fail("data.split_ratios", cfg.data.split_ratios, "three non-negative numbers summing to 1")
    ratios = [_require_number("data.split_ratios", item) for item in cfg.data.split_ratios]
    if any(item < 0 for item in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        _fail("data.split_ratios", cfg.data.split_ratios, "three non-negative numbers summing to 1")

    _require_enum("data.aug", cfg.data.aug, ("basic", "randaug"))
    _require_enum("data.interpolation", cfg.data.interpolation, ("bilinear", "bicubic"))
    _require_enum("model.preprocessing", cfg.model.preprocessing, ("fixed", "model_default"))
    if cfg.model.factory is not None and cfg.model.preprocessing == "model_default":
        _fail("model.preprocessing", cfg.model.preprocessing, "fixed when model.factory is set")
    _require_enum("train.optimizer", cfg.train.optimizer, ("adamw", "sgd", "lion"))
    _require_enum("train.scheduler", cfg.train.scheduler, ("cosine", "step", "none"))
    _require_enum("train.class_weight", cfg.train.class_weight, ("none", "inverse", "effective"))
    _require_enum("train.sampler", cfg.train.sampler, ("none", "weighted"))
    _require_enum("train.best_metric", cfg.train.best_metric, SUPPORTED_BEST_METRICS)
    _require_enum("device", cfg.device, ("auto", "cpu", "cuda", "mps"))
    _require_enum("log_level", cfg.log_level, ("debug", "info", "warning", "error", "critical"))
    if cfg.train.mixup_alpha > 0 and cfg.train.cutmix_alpha > 0:
        _fail(
            "train.mixup_alpha/train.cutmix_alpha", (cfg.train.mixup_alpha, cfg.train.cutmix_alpha), "only one enabled"
        )

    numeric_ranges: tuple[tuple[str, Any, float, float | None, bool, bool], ...] = (
        ("train.lr", cfg.train.lr, 0.0, None, False, True),
        ("train.weight_decay", cfg.train.weight_decay, 0.0, None, True, True),
        ("train.momentum", cfg.train.momentum, 0.0, 1.0, True, False),
        ("train.label_smoothing", cfg.train.label_smoothing, 0.0, 1.0, True, False),
        ("train.mixup_alpha", cfg.train.mixup_alpha, 0.0, None, True, True),
        ("train.cutmix_alpha", cfg.train.cutmix_alpha, 0.0, None, True, True),
        ("train.grad_clip", cfg.train.grad_clip, 0.0, None, True, True),
        ("train.ema_decay", cfg.train.ema_decay, 0.0, 1.0, True, False),
    )
    for field_name, field_value, low, high, include_low, include_high in numeric_ranges:
        number = _require_number(field_name, field_value)
        if number < low or (number == low and not include_low):
            _fail(field_name, field_value, f">{'=' if include_low else ''} {low}")
        if high is not None and (number > high or (number == high and not include_high)):
            _fail(field_name, field_value, f"<{'=' if include_high else ''} {high}")


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Serialize config using only YAML/JSON-friendly primitive values."""
    return _serialize(dataclasses.asdict(cfg))


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(item) for item in value]
    return value


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    from .utils import write_text_atomic

    write_text_atomic(Path(path), yaml.safe_dump(to_dict(cfg), sort_keys=False))
