"""Unit tests for the configuration system."""

import pytest
import yaml

from garbage_classifier.config import ModelConfig, dump_config, load_config, to_dict


def test_defaults():
    cfg = load_config()
    assert cfg.model.name == "resnet50"
    assert cfg.train.epochs == 60
    assert cfg.data.classes == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def test_yaml_overrides(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"model": {"name": "convnext_tiny", "pretrained": False}, "train": {"epochs": 5}}))
    cfg = load_config(p)
    assert cfg.model.name == "convnext_tiny"
    assert cfg.model.pretrained is False
    assert cfg.train.epochs == 5
    assert cfg.data.image_size == 224  # untouched default


def test_dotted_overrides():
    cfg = load_config(overrides={"train.lr": 1e-4, "model.name": "swin_tiny"})
    assert cfg.train.lr == 1e-4
    assert cfg.model.name == "swin_tiny"


def test_unknown_section_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("nonsense: {a: 1}\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_unknown_nested_field_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("train: {learning_rate: 0.01}\n")

    with pytest.raises(ValueError, match=r"train\.learning_rate"):
        load_config(p)


@pytest.mark.parametrize("key", ["train.learning_rate", "train.lr.extra", "unknown.value"])
def test_unknown_dotted_override_rejected(key):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        load_config(overrides={key: 1})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("train.epochs", True),
        ("train.lr", "0.001"),
        ("data.pin_memory", 1),
        ("model.pretrained", "false"),
        ("data.classes", "paper"),
    ],
)
def test_invalid_types_rejected(key, value):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        load_config(overrides={key: value})


@pytest.mark.parametrize(
    ("key", "value", "allowed"),
    [
        ("train.optimizer", "rmsprop", "adamw, sgd, lion"),
        ("train.scheduler", "linear", "cosine, step, none"),
        ("data.aug", "autoaugment", "basic, randaug"),
        ("train.class_weight", "balanced", "none, inverse, effective"),
        ("train.sampler", "random", "none, weighted"),
        ("device", "tpu", "auto, cpu, cuda, mps"),
        ("log_level", "trace", "debug, info, warning, error, critical"),
    ],
)
def test_invalid_enums_name_value_and_allowed_options(key, value, allowed):
    with pytest.raises(ValueError) as exc_info:
        load_config(overrides={key: value})

    message = str(exc_info.value)
    assert key in message
    assert repr(value) in message
    assert allowed in message


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("train.epochs", 0),
        ("train.batch_size", 0),
        ("train.lr", 0.0),
        ("train.weight_decay", -0.1),
        ("train.momentum", 1.1),
        ("train.warmup_epochs", -1),
        ("train.label_smoothing", 1.0),
        ("train.mixup_alpha", -0.1),
        ("train.cutmix_alpha", -0.1),
        ("train.grad_clip", -0.1),
        ("train.early_stop_patience", -1),
        ("train.ema_decay", 1.0),
        ("data.image_size", 0),
        ("data.resize_size", 0),
        ("data.num_workers", -1),
    ],
)
def test_out_of_range_values_rejected(key, value):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        load_config(overrides={key: value})


def test_resize_must_cover_crop_size():
    with pytest.raises(ValueError, match="data.resize_size"):
        load_config(overrides={"data.image_size": 256, "data.resize_size": 224})


@pytest.mark.parametrize(
    "ratios",
    [[0.8, 0.2], [0.8, -0.1, 0.3], [0.8, 0.1, 0.2], [0.8, 0.1, "0.1"]],
)
def test_invalid_split_ratios_rejected(ratios):
    with pytest.raises(ValueError, match="data.split_ratios"):
        load_config(overrides={"data.split_ratios": ratios})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("data.normalize_mean", [0.5, 0.5]),
        ("data.normalize_mean", [0.5, -0.1, 0.5]),
        ("data.normalize_mean", [0.5, 1.1, 0.5]),
        ("data.normalize_std", [0.2, 0.2]),
        ("data.normalize_std", [0.2, 0.0, 0.2]),
    ],
)
def test_invalid_normalization_statistics_rejected(key, value):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        load_config(overrides={key: value})


@pytest.mark.parametrize("classes", [[], ["paper", "paper"], ["paper", 1]])
def test_invalid_classes_rejected(classes):
    with pytest.raises(ValueError, match="data.classes"):
        load_config(overrides={"data.classes": classes})


def test_model_config_no_longer_exposes_timm_backbone():
    assert "timm_backbone" not in ModelConfig.__dataclass_fields__

    with pytest.raises(ValueError, match="model.timm_backbone"):
        load_config(overrides={"model.timm_backbone": "resnet18"})


def test_dump_roundtrip(tmp_path):
    cfg = load_config(overrides={"train.epochs": 3, "run_name": "x"})
    out = tmp_path / "out.yaml"
    dump_config(cfg, out)
    loaded = load_config(out)
    assert to_dict(loaded) == to_dict(cfg)
