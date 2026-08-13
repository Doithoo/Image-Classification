"""Tests for Grad-CAM and TTA (test-time augmentation)."""

import torch

from garbage_classifier.inference import Predictor
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


def test_predict_probs_tta_averages_horizontal_flip_probabilities():
    class DirectionSensitiveModel(torch.nn.Module):
        def forward(self, images):
            left = images[..., 0].mean(dim=(1, 2))
            right = images[..., -1].mean(dim=(1, 2))
            return torch.stack((left, right), dim=1)

    predictor = Predictor.__new__(Predictor)
    predictor.model = DirectionSensitiveModel()
    x = torch.tensor([[[[4.0, 0.0]]]])

    with torch.no_grad():
        original = torch.softmax(predictor.model(x), dim=1)
        flipped = torch.softmax(predictor.model(torch.flip(x, dims=[3])), dim=1)
        actual = predictor.predict_probs(x, tta=True)

    assert not torch.allclose(original, flipped)
    assert torch.allclose(actual, (original + flipped) / 2)
