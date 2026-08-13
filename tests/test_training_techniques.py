"""Tests for the learning modules: MixUp/CutMix, EMA, LR warmup."""

import pytest
import torch

from garbage_classifier.config import load_config
from garbage_classifier.training.ema import EMA
from garbage_classifier.training.mixup import MixupCutmix, one_hot_mixup_target, soft_cross_entropy
from garbage_classifier.training.trainer import Trainer


def test_one_hot_targets_and_smoothing():
    labels = torch.tensor([0, 2])
    t = one_hot_mixup_target(labels, num_classes=3, label_smoothing=0.0)
    assert t.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    # smoothing pulls mass away from the true class: 0.9 + 0.1/3 on true, 0.1/3 elsewhere
    ts = one_hot_mixup_target(labels, num_classes=3, label_smoothing=0.1)
    assert abs(ts[0, 0].item() - (0.9 + 0.1 / 3)) < 1e-6
    assert abs(ts[0, 1].item() - 0.1 / 3) < 1e-6


def test_mixup_disabled_passthrough():
    m = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=0.0, num_classes=3)
    assert not m.enabled
    images = torch.randn(4, 3, 8, 8)
    labels = torch.randint(0, 3, (4,))
    out_images, out_targets = m(images, labels)
    # disabled: images unchanged, targets are one-hot of the same labels
    assert torch.equal(out_images, images)
    assert torch.allclose(out_targets.argmax(dim=1), labels)


def test_mixup_mixes_images_and_targets():
    torch.manual_seed(0)
    m = MixupCutmix(mixup_alpha=1.0, cutmix_alpha=0.0, num_classes=3)
    images = torch.randn(8, 3, 16, 16)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out_images, out_targets = m(images, labels)
    # each output image is a convex combination of two inputs -> norm strictly smaller
    assert out_images.norm() < images.norm()
    # soft targets sum to 1 per sample and lie strictly between classes
    assert torch.allclose(out_targets.sum(dim=1), torch.ones(8))
    assert out_targets.min() >= 0.0 and out_targets.max() <= 1.0


def test_cutmix_preserves_local_info():
    torch.manual_seed(1)
    m = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, num_classes=3)
    images = torch.randn(8, 3, 16, 16)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out_images, out_targets = m(images, labels)
    assert out_images.shape == images.shape
    assert torch.allclose(out_targets.sum(dim=1), torch.ones(8))


def test_cutmix_copies_source_pixels_and_uses_actual_patch_area():
    torch.manual_seed(7)
    m = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=1.0, num_classes=4)
    images = torch.stack([torch.full((1, 9, 11), float(i)) for i in range(4)])
    labels = torch.arange(4)

    mixed, targets = m(images, labels)

    for index in range(len(images)):
        values = mixed[index].unique()
        assert all(value.item() in range(4) for value in values)
        assert len(values) <= 2
        own_fraction = (mixed[index] == float(index)).float().mean()
        assert torch.isclose(targets[index, index], own_fraction)


def test_soft_cross_entropy_matches_hard_for_onehot():
    torch.manual_seed(2)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    hard = torch.nn.functional.cross_entropy(logits, labels)
    soft = soft_cross_entropy(logits, one_hot_mixup_target(labels, 3), class_weights=None)
    assert torch.allclose(hard, soft, atol=1e-5)


def test_ema_shadow_tracks_slowly():
    torch.manual_seed(3)
    model = torch.nn.Linear(4, 2)
    ema = EMA(model, decay=0.9)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        model.weight.add_(1.0)  # big jump in fast weights
    ema.update(model)
    # shadow moved only 10% of the way toward the new weights
    delta = (ema.shadow["weight"] - before["weight"]).abs().max().item()
    assert 0.05 < delta < 0.2


def test_ema_apply_restores():
    torch.manual_seed(4)
    model = torch.nn.Linear(4, 2)
    ema = EMA(model, decay=0.9)
    ema.apply_to(model)
    for name, p in model.state_dict().items():
        if name in ema.shadow:
            assert torch.equal(p, ema.shadow[name])


def test_warmup_then_cosine_scheduler():
    """Warmup raises LR slowly, then cosine decays it."""
    cfg = load_config(overrides={"train.warmup_epochs": 3, "train.epochs": 10, "train.scheduler": "cosine"})
    model = torch.nn.Linear(8, 2)
    trainer = Trainer(model, cfg, torch.device("cpu"), ["a", "b"], "/tmp/__warmup_test__")
    assert trainer.scheduler is not None
    lrs = []
    for _ in range(10):
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        trainer.optimizer.step()
        trainer.scheduler.step()
    # warmup phase: strictly increasing
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3]
    # decay phase: strictly decreasing
    assert lrs[4] > lrs[5] > lrs[8]
    # never exceeds the base lr
    assert max(lrs) <= cfg.train.lr + 1e-9


def test_loader_smaller_than_batch_still_updates_model(tmp_path):
    cfg = load_config(
        overrides={
            "train.batch_size": 8,
            "train.amp": False,
            "train.mixup_alpha": 0.0,
            "train.cutmix_alpha": 0.0,
        }
    )
    model = torch.nn.Linear(2, 2)
    trainer = Trainer(model, cfg, torch.device("cpu"), ["a", "b"], tmp_path)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        batch_size=8,
        drop_last=False,
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}

    result = trainer._run_epoch(loader, train=True)

    assert result["loss"] > 0
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_empty_loader_has_clear_error(tmp_path):
    cfg = load_config(overrides={"train.amp": False})
    trainer = Trainer(torch.nn.Linear(2, 2), cfg, torch.device("cpu"), ["a", "b"], tmp_path)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)), batch_size=4
    )

    with pytest.raises(ValueError, match="training loader is empty"):
        trainer._run_epoch(loader, train=True)
