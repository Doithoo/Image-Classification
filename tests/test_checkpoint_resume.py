import random

import numpy as np
import pytest
import torch

from garbage_classifier.config import load_config
from garbage_classifier.training.checkpoint import (
    deployable_model_state,
    load_checkpoint,
    restore_config_from_checkpoint,
    save_checkpoint,
)
from garbage_classifier.training.ema import EMA
from garbage_classifier.training.trainer import Trainer


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key])


def test_checkpoint_round_trips_training_and_deployable_state(tmp_path):
    cfg = load_config(overrides={"train.amp": False, "train.ema": True})
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
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
        scaler=scaler,
        patience_left=4,
        epoch=3,
        best_metric=0.75,
        cfg=cfg,
        class_names=["a", "b"],
    )
    payload = load_checkpoint(path)

    _assert_state_equal(payload["training_model_state_dict"], fast_state)
    _assert_state_equal(payload["deployable_model_state_dict"], deployable_state)
    _assert_state_equal(payload["model_state_dict"], deployable_state)
    _assert_state_equal(payload["ema_state_dict"]["shadow"], deployable_state)
    assert payload["scaler_state_dict"] == scaler.state_dict()
    assert payload["patience_left"] == 4
    assert set(payload["rng_state"]) == {"python", "numpy", "torch", "cuda"}


def test_trainer_resume_restores_full_state_and_rng(tmp_path):
    cfg = load_config(
        overrides={
            "train.amp": False,
            "train.ema": True,
            "train.scheduler": "step",
            "train.epochs": 5,
        }
    )
    source = Trainer(torch.nn.Linear(2, 2), cfg, torch.device("cpu"), ["a", "b"], tmp_path / "source")
    with torch.no_grad():
        source.model.weight.add_(1.5)
    source.optimizer.param_groups[0]["lr"] = 0.0123
    source.optimizer.step()
    source.scheduler.step()
    expected_lr = source.optimizer.param_groups[0]["lr"]
    source.ema.update(source.model)
    source.epoch = 2
    source.best_metric = 0.6
    source.patience_left = 3
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    source._save("resume.pt", {"accuracy": 0.6})
    expected_fast = {key: value.clone() for key, value in source.model.state_dict().items()}
    expected_ema = {key: value.clone() for key, value in source.ema.shadow.items()}
    expected_random = (random.random(), np.random.random(), torch.rand(1))

    resumed = Trainer(torch.nn.Linear(2, 2), cfg, torch.device("cpu"), ["a", "b"], tmp_path / "resumed")
    resumed._resume(str(tmp_path / "source" / "resume.pt"))

    _assert_state_equal(resumed.model.state_dict(), expected_fast)
    _assert_state_equal(resumed.ema.shadow, expected_ema)
    assert resumed.optimizer.param_groups[0]["lr"] == expected_lr
    assert resumed.scheduler.state_dict() == source.scheduler.state_dict()
    assert resumed.scaler.state_dict() == source.scaler.state_dict()
    assert (resumed.epoch, resumed.best_metric, resumed.patience_left) == (2, 0.6, 3)
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(1), expected_random[2])


def test_old_checkpoint_remains_loadable(tmp_path):
    model = torch.nn.Linear(2, 2)
    old = {
        "model_state_dict": model.state_dict(),
        "config": {},
        "class_names": ["a", "b"],
        "epoch": 1,
    }
    path = tmp_path / "old.pt"
    torch.save(old, path)

    payload = load_checkpoint(path)

    _assert_state_equal(payload["training_model_state_dict"], old["model_state_dict"])
    _assert_state_equal(payload["deployable_model_state_dict"], old["model_state_dict"])


def test_restore_config_and_deployable_state_are_shared_checkpoint_helpers():
    state = {"weight": torch.tensor([2.0])}
    payload = {
        "config": {
            "data": {"image_size": 37, "resize_size": 41},
            "model": {"name": "mobilenetv3_small_100"},
        },
        "deployable_model_state_dict": state,
        "model_state_dict": {"weight": torch.tensor([1.0])},
    }

    cfg = restore_config_from_checkpoint(payload)

    assert cfg.data.image_size == 37
    assert cfg.data.resize_size == 41
    assert cfg.model.name == "mobilenetv3_small_100"
    assert deployable_model_state(payload) is state


def test_restore_config_reports_missing_metadata_clearly():
    with pytest.raises(ValueError, match="checkpoint is missing config metadata"):
        restore_config_from_checkpoint({})


def test_deployable_state_supports_legacy_model_state_dict():
    state = {"weight": torch.tensor([1.0])}
    assert deployable_model_state({"model_state_dict": state}) is state
