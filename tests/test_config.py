"""Configuration contracts, provenance reporting and validation behavior."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from garbage_classifier.config import (
    SUPPORTED_BEST_METRICS,
    ConfigError,
    DataConfig,
    ModelConfig,
    TrainConfig,
    dump_config,
    load_config,
    load_config_with_sources,
    to_dict,
)
from garbage_classifier.models.registry import resolve_preprocessing


def test_minimal_defaults_are_immutable_and_do_not_claim_dataset_classes():
    cfg = load_config()
    assert cfg.model.name == "resnet18"
    assert cfg.model.pretrained is False
    assert cfg.train.epochs == 20
    assert cfg.data.data_dir == Path("data/raw")
    assert "classes" not in DataConfig.__dataclass_fields__
    with pytest.raises(FrozenInstanceError):
        cfg.device = "cpu"


def test_run_name_is_one_safe_directory_component():
    for invalid in ("../escape", "nested/run", ".", "..", "/absolute"):
        with pytest.raises(ConfigError, match="run_name"):
            load_config(overrides={"run_name": invalid})


def test_model_default_preprocessing_comes_from_each_provider_spec():
    efficientnet = load_config(overrides={"model.name": "efficientnet_b3", "model.preprocessing": "model_default"})
    resolved = resolve_preprocessing(efficientnet.data, efficientnet.model)
    assert (resolved.image_size, resolved.resize_size, resolved.interpolation) == (288, 329, "bicubic")

    vit = load_config(overrides={"model.name": "vit_base_patch16_224", "model.preprocessing": "model_default"})
    resolved_vit = resolve_preprocessing(vit.data, vit.model)
    assert resolved_vit.normalize_mean == [0.5, 0.5, 0.5]

    torchvision = load_config(overrides={"model.name": "tv_resnet50", "model.preprocessing": "model_default"})
    resolved_tv = resolve_preprocessing(torchvision.data, torchvision.model)
    assert (resolved_tv.resize_size, resolved_tv.interpolation) == (232, "bilinear")


def test_yaml_dotted_and_model_params_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"model": {"name": "convnext_tiny", "params": {"drop_path_rate": 0.1}}}))

    cfg = load_config(path, {"model.params.drop_path_rate": 0.2, "data.data_dir": "images"})

    assert cfg.model.name == "convnext_tiny"
    assert cfg.model.params == {"drop_path_rate": 0.2}
    assert cfg.data.data_dir == Path("images")


def test_show_config_sources_data_model_and_cli_provenance(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  epochs: 3\n")

    cfg, sources = load_config_with_sources(path, {"train.lr": 0.01})

    assert cfg.train.epochs == 3
    assert cfg.train.lr == 0.01
    assert sources["train.epochs"] == "yaml"
    assert sources["train.lr"] == "cli"
    assert sources["model.name"] == "default"


@pytest.mark.parametrize("key", ["train.learning_rate", "train.lr.extra", "unknown.value", "data.classes"])
def test_unknown_keys_are_rejected(key):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(overrides={key: 1})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("train.epochs", True),
        ("train.lr", "0.001"),
        ("data.pin_memory", 1),
        ("model.pretrained", "false"),
        ("model.params", "not-a-mapping"),
        ("data.interpolation", "nearest"),
        ("model.preprocessing", "unknown"),
        ("model.factory", "not-a-factory"),
        ("data.data_dir", None),
        ("output_dir", []),
    ],
)
def test_invalid_types_and_choices_are_rejected(key, value):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(overrides={key: value})


@pytest.mark.parametrize("metric", SUPPORTED_BEST_METRICS)
def test_every_declared_best_metric_is_accepted(metric):
    assert load_config(overrides={"train.best_metric": metric}).train.best_metric == metric


def test_invalid_metric_and_incompatible_techniques_are_rejected():
    with pytest.raises(ConfigError, match="train.best_metric"):
        load_config(overrides={"train.best_metric": "not-a-metric"})
    with pytest.raises(ConfigError, match="mixup_alpha"):
        load_config(overrides={"train.mixup_alpha": 0.2, "train.cutmix_alpha": 0.2})
    with pytest.raises(ConfigError, match="warmup_epochs"):
        load_config(overrides={"train.epochs": 2, "train.warmup_epochs": 2})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("train.epochs", 0),
        ("train.batch_size", 0),
        ("train.lr", 0.0),
        ("train.early_stop_patience", -1),
        ("data.image_size", 0),
        ("data.resize_size", 0),
        ("data.num_workers", -1),
        ("data.normalize_mean", [0.5, 0.5]),
        ("data.normalize_std", [0.2, 0.0, 0.2]),
        ("data.split_ratios", [0.8, 0.1, 0.2]),
    ],
)
def test_ranges_are_validated(key, value):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(overrides={key: value})


def test_config_round_trip_including_new_contract_fields(tmp_path):
    cfg = load_config(
        overrides={
            "model.preprocessing": "model_default",
            "model.params": {"drop_rate": 0.1},
            "data.interpolation": "bicubic",
            "run_name": "x",
        }
    )
    output = tmp_path / "config.yaml"
    dump_config(cfg, output)

    assert to_dict(load_config(output)) == to_dict(cfg)
    assert set(ModelConfig.__dataclass_fields__) >= {"preprocessing", "params", "factory"}
    assert "interpolation" in DataConfig.__dataclass_fields__
    assert TrainConfig.__dataclass_fields__["early_stop_patience"].default == 5
