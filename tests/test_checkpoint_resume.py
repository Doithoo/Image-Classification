"""Safe checkpoint schema, atomic publication and resume-identity tests."""

import random

import numpy as np
import pytest
import torch

from garbage_classifier.config import dump_config, load_config
from garbage_classifier.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    build_resume_identity,
    deployable_model_state,
    load_checkpoint,
    restore_config_from_checkpoint,
    save_checkpoint,
    validate_inference_model_source,
    validate_resume_identity,
)
from garbage_classifier.training.ema import EMA
from garbage_classifier.training.trainer import Trainer


class UnsafeCheckpointObject:
    pass


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key])


def test_schema_v2_round_trips_training_state_using_tensor_only_loader(tmp_path):
    cfg = load_config(overrides={"train.amp": False, "train.ema": True, "model.num_classes": 2})
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update(model)
    fast_state = {key: value.clone() for key, value in model.state_dict().items()}
    deployable_state = {key: value.clone() for key, value in ema.shadow.items()}
    path = tmp_path / "roundtrip.pt"

    save_checkpoint(
        path,
        model=model,
        deployable_state_dict=deployable_state,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        patience_left=4,
        epoch=3,
        best_metric=0.75,
        cfg=cfg,
        class_names=["a", "b"],
        manifest_identity="manifest-123",
    )
    payload = load_checkpoint(path)

    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["manifest_identity"] == "manifest-123"
    assert payload["preprocessing"]["image_size"] == cfg.data.image_size
    _assert_state_equal(payload["training_model_state_dict"], fast_state)
    _assert_state_equal(payload["deployable_model_state_dict"], deployable_state)
    assert not list(tmp_path.glob(".roundtrip.pt.*.tmp"))


def test_resume_validates_identity_and_restores_rng(tmp_path):
    cfg = load_config(
        overrides={
            "train.amp": False,
            "train.ema": True,
            "train.scheduler": "step",
            "train.epochs": 5,
            "model.num_classes": 2,
        }
    )
    source = Trainer(torch.nn.Linear(2, 2), cfg, torch.device("cpu"), ["a", "b"], tmp_path / "source", "manifest-123")
    source.epoch = 2
    source.best_metric = 0.6
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    source._save("resume.pt", {"accuracy": 0.6})
    expected_random = (random.random(), np.random.random(), torch.rand(1))

    resumed = Trainer(torch.nn.Linear(2, 2), cfg, torch.device("cpu"), ["a", "b"], tmp_path / "resumed", "manifest-123")
    resumed._resume(str(tmp_path / "source" / "resume.pt"))

    assert (resumed.epoch, resumed.best_metric) == (2, 0.6)
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(1), expected_random[2])

    payload = load_checkpoint(tmp_path / "source" / "resume.pt")
    with pytest.raises(CheckpointCompatibilityError, match="manifest_identity"):
        validate_resume_identity(payload, build_resume_identity(cfg, ["a", "b"], "other-manifest"))
    with pytest.raises(CheckpointCompatibilityError, match="class_names"):
        validate_resume_identity(payload, build_resume_identity(cfg, ["b", "a"], "manifest-123"))


def test_external_factory_requires_an_explicit_reviewed_config(tmp_path):
    cfg = load_config(
        overrides={
            "model.name": "custom",
            "model.factory": "reviewed.module:build",
            "model.num_classes": 2,
        }
    )
    path = tmp_path / "external.pt"
    save_checkpoint(
        path,
        model=torch.nn.Linear(2, 2),
        epoch=0,
        best_metric=0.0,
        cfg=cfg,
        class_names=["a", "b"],
        manifest_identity="manifest-123",
    )
    payload = load_checkpoint(path)

    with pytest.raises(CheckpointCompatibilityError, match="reviewed training config"):
        validate_inference_model_source(payload)

    config_path = tmp_path / "config.yaml"
    dump_config(cfg, config_path)
    assert validate_inference_model_source(payload, config_path).model.factory == "reviewed.module:build"

    changed = load_config(overrides={"model.name": "other", "model.factory": "reviewed.module:build"})
    dump_config(changed, config_path)
    with pytest.raises(CheckpointCompatibilityError, match="changes checkpoint model field"):
        validate_inference_model_source(payload, config_path)


def test_checkpoint_loader_rejects_unsafe_pickle_payload(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"config": {}, "class_names": ["a"], "unsafe": UnsafeCheckpointObject()}, path)
    with pytest.raises(CheckpointCompatibilityError, match="cannot safely load"):
        load_checkpoint(path)


def test_checkpoint_loader_rejects_internally_inconsistent_contracts(tmp_path):
    cfg = load_config(overrides={"model.num_classes": 2})
    path = tmp_path / "inconsistent.pt"
    save_checkpoint(
        path,
        model=torch.nn.Linear(2, 2),
        epoch=0,
        best_metric=0.0,
        cfg=cfg,
        class_names=["a", "b"],
        manifest_identity="manifest-123",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["model"]["name"] = "different"
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match="model contract"):
        load_checkpoint(path)


def test_legacy_tensor_checkpoint_is_prediction_compatible_but_not_resumable(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {}, "class_names": ["a", "b"]}, path)

    payload = load_checkpoint(path)

    assert payload["legacy_checkpoint"] is True
    assert deployable_model_state(payload) is payload["model_state_dict"]
    assert restore_config_from_checkpoint(payload).model.name == "resnet18"
    with pytest.raises(CheckpointCompatibilityError, match="legacy"):
        validate_resume_identity(payload, build_resume_identity(load_config(), ["a", "b"], "identity"))


def test_restore_config_accepts_old_unused_classes_metadata():
    payload = {
        "config": {"data": {"classes": ["old"], "image_size": 37, "resize_size": 41}},
        "model_state_dict": {"weight": torch.tensor([1.0])},
    }
    cfg = restore_config_from_checkpoint(payload)
    assert cfg.data.image_size == 37
    assert cfg.data.resize_size == 41
