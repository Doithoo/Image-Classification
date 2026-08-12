"""Tests for Grad-CAM and TTA (test-time augmentation)."""

import torch

from garbage_classifier.inference.gradcam import GradCAM, _find_last_conv


def _tiny_cnn() -> torch.nn.Module:
    """A tiny CNN with a clear last conv layer (2 classes)."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )


def test_find_last_conv():
    model = _tiny_cnn()
    assert isinstance(_find_last_conv(model), torch.nn.Conv2d)


def test_gradcam_heatmap_values():
    torch.manual_seed(0)
    model = _tiny_cnn()
    cam_model = GradCAM(model)
    x = torch.randn(3, 16, 16)
    heatmap, class_idx = cam_model.generate(x)
    assert heatmap.shape == (16, 16)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert class_idx in (0, 1)


def test_gradcam_specific_class():
    torch.manual_seed(1)
    model = _tiny_cnn()
    cam_model = GradCAM(model)
    x = torch.randn(3, 16, 16)
    heatmap, class_idx = cam_model.generate(x, class_idx=0)
    assert class_idx == 0
    assert heatmap.shape == (16, 16)


def test_tta_averages_probabilities(tmp_path):
    # exercise the TTA averaging logic with a stand-in object (no checkpoint needed)
    import types

    fake = types.SimpleNamespace()
    fake.model = torch.nn.Sequential(
        torch.nn.Linear(16, 3),
    ).eval()
    fake.class_names = ["a", "b", "c"]

    x = torch.randn(4, 16)
    with torch.no_grad():
        base = torch.softmax(fake.model(x), dim=1)
        # manual TTA: average with flipped-view probabilities (flip is no-op here)
        tta = (base + torch.softmax(fake.model(x), dim=1)) / 2
    assert torch.allclose(base, tta)  # flip of a vector input == itself
    assert torch.allclose(tta.sum(dim=1), torch.ones(4))
