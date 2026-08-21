"""Checkpoint-driven single-image and batch inference."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from ..data.transforms import build_inference_transform
from ..models.registry import create_model
from ..training.checkpoint import (
    deployable_model_state,
    load_checkpoint,
    validate_inference_model_source,
)
from ..utils import pick_device


class Predictor:
    """A model reconstructed only from a self-contained checkpoint contract."""

    def __init__(
        self, checkpoint_path: str | Path, device: str = "auto", config_path: str | Path | None = None
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        payload = load_checkpoint(self.checkpoint_path)
        self.class_names: list[str] = payload["class_names"]
        checkpoint_cfg = validate_inference_model_source(payload, config_path)
        self.cfg = replace(checkpoint_cfg, device=device)
        self.device = pick_device(device)
        self.model = create_model(
            self.cfg.model.name,
            num_classes=len(self.class_names),
            pretrained=False,
            factory=self.cfg.model.factory,
            params=self.cfg.model.params,
        )
        self.model.load_state_dict(deployable_model_state(payload))
        self.model.to(self.device).eval()
        self.transform = build_inference_transform(self.cfg.data)

    def predict(self, image: Image.Image, top_k: int = 1, tta: bool = False) -> list[tuple[str, float]]:
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = self.predict_probs(image_tensor, tta=tta)[0]
        top = torch.topk(probabilities, k=min(top_k, len(self.class_names)))
        return [
            (self.class_names[index], float(probability))
            for index, probability in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ]

    def predict_path(self, path: str | Path, top_k: int = 1, tta: bool = False) -> list[tuple[str, float]]:
        with Image.open(path) as image:
            return self.predict(image, top_k=top_k, tta=tta)

    def predict_probs(self, images: torch.Tensor, tta: bool = False) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(images), dim=1)
            if tta:
                probabilities = (probabilities + torch.softmax(self.model(torch.flip(images, dims=[3])), dim=1)) / 2.0
        return probabilities
