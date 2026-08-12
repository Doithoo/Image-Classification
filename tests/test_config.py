"""Unit tests for the configuration system."""

import pytest
import yaml

from garbage_classifier.config import dump_config, load_config, to_dict


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


def test_dump_roundtrip(tmp_path):
    cfg = load_config(overrides={"train.epochs": 3, "run_name": "x"})
    out = tmp_path / "out.yaml"
    dump_config(cfg, out)
    loaded = load_config(out)
    assert to_dict(loaded) == to_dict(cfg)
