import random

import numpy as np
import torch

from garbage_classifier.config import load_config
from garbage_classifier.training.checkpoint import load_checkpoint, save_checkpoint
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
