"""Grad-CAM for registered CNN models with explicit hook lifecycle management."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _find_last_conv(model: nn.Module) -> nn.Module:
    last: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("no Conv2d layer found; this model needs a model-specific explanation method")
    return last


class GradCAM:
    """Generate a class activation map without backward-module-hook warnings."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model.eval()
        self.target_layer = target_layer or _find_last_conv(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self._activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient: torch.Tensor) -> None:
        self._gradients = gradient.detach()

    def close(self) -> None:
        """Release the forward hook when explanation is complete."""
        self._handle.remove()

    def heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> tuple[torch.Tensor, int]:
        if self._activations is None or self._gradients is None:
            raise RuntimeError("run generate() before requesting the heatmap")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), class_idx

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, int]:
        tensor = input_tensor.to(device).unsqueeze(0) if input_tensor.dim() == 3 else input_tensor.to(device)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()
        return self.heatmap(tensor, class_idx)


def overlay_heatmap(image: Any, heatmap: torch.Tensor, alpha: float = 0.5):
    """Blend a normalized activation map over an RGB source image."""
    import numpy as np
    from PIL import Image

    cam = heatmap.detach().cpu().numpy()
    colour = np.stack([np.clip(cam * 2, 0, 1), np.clip(cam * 2 - 0.5, 0, 1), np.clip(1 - cam * 2, 0, 1)], axis=-1)
    source = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    blended = alpha * np.asarray(source) / 255.0 + (1 - alpha) * colour
    return Image.fromarray((blended * 255).astype(np.uint8))
