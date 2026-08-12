"""Grad-CAM: class-activation heatmaps for CNN interpretability.

Why this exists (learning note):
  - Grad-CAM (Selvaraju et al. 2017) answers "which pixels made the model decide
    class C?" — the classic way to debug *why* a model misclassifies an image.
  - Idea: take the feature map of the LAST convolutional layer (it keeps spatial
    information), weight each channel by how much it contributed to the class
    score, and average the channels into a heatmap:
        1. forward pass  -> capture activations A (C x H x W) at the target layer
        2. backward pass -> gradients d score/d A  from the class of interest
        3. channel weight  α_c = mean over spatial dims of grad(A_c)
        4. heatmap        = ReLU( Σ_c α_c · A_c )   (ReLU keeps only positive evidence)
        5. normalize, resize to input, overlay in red-blue
  - The last conv layer is chosen because it has the best resolution vs semantic
    trade-off: earlier layers are too low-level, the classifier head has no
    spatial structure.

Implementation notes:
  - We find the LAST Conv2d module by walking the module tree, so this works for
    any architecture (ResNet, MobileNet, Swin, ...) without hardcoding layer names.
  - PyTorch hooks: ``register_forward_hook`` stores activations; a full backward
    is triggered from the selected class logit to obtain the gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _find_last_conv(model: nn.Module) -> nn.Module:
    """Return the last Conv2d submodule (any architecture)."""
    last: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("no Conv2d layer found in model")
    return last


class GradCAM:
    """Generate a class-activation heatmap for a given input."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        self.target_layer = _find_last_conv(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        # forward hook: keep the conv output; backward hook: keep its gradient
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _backward_hook(
        self, module: nn.Module, grad_in: tuple[torch.Tensor, ...], grad_out: tuple[torch.Tensor, ...]
    ) -> None:
        self._gradients = grad_out[0].detach()

    @torch.no_grad()
    def heatmap(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[torch.Tensor, int]:
        """Return (heatmap [0,1] at input spatial size, predicted class idx)."""
        if self._activations is None or self._gradients is None:
            raise RuntimeError("run a backward pass first: use generate()")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # α_c: per-channel mean gradient
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))  # ReLU(Σ α·A)
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        # normalize to [0, 1] per image
        cam = cam.squeeze(0).squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def generate(
        self, input_tensor: torch.Tensor, class_idx: int | None = None, device: str = "cpu"
    ) -> tuple[torch.Tensor, int]:
        """Run forward+backward for the class of interest and return the heatmap."""
        tensor = input_tensor.to(device).unsqueeze(0) if input_tensor.dim() == 3 else input_tensor.to(device)
        self.model.zero_grad()
        logits = self.model(tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        # backprop from the class score to obtain d score / d activations
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot)
        return self.heatmap(tensor, class_idx)


def overlay_heatmap(image, heatmap: torch.Tensor, alpha: float = 0.5):
    """Blend the Grad-CAM heatmap (red-blue) onto the original image."""
    import numpy as np
    from PIL import Image

    cam = heatmap.cpu().numpy()
    # color map: blue (low) -> green -> red (high)
    cmap = np.stack([cam, cam, cam], axis=-1)
    cmap[..., 0] = np.clip(cam * 2.0, 0, 1)  # red channel rises first
    cmap[..., 1] = np.clip(cam * 2.0 - 0.5, 0, 1)
    cmap[..., 2] = np.clip(1.0 - cam * 2.0, 0, 1)  # blue fades

    image = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    blended = alpha * np.array(image) / 255.0 + (1 - alpha) * cmap
    return Image.fromarray((blended * 255).astype(np.uint8))
