"""Inference: single-image and batch prediction driven entirely by a checkpoint.

The checkpoint is self-contained (config + class names + weights), so prediction
never requires manually repeating class names, normalization statistics or model
names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..config import DataConfig, ModelConfig, load_config
from ..data.transforms import build_inference_transform
from ..models.registry import create_model
from ..training.checkpoint import load_checkpoint
from ..utils import pick_device


class Predictor:
    def __init__(
        self, checkpoint_path: str | Path, device: str = "auto", config_path: str | Path | None = None
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        payload = load_checkpoint(self.checkpoint_path)
        self.class_names: list[str] = payload["class_names"]

        if config_path is not None:
            self.cfg = load_config(config_path)
        else:
            # restore full config from checkpoint metadata (no drift possible)
            raw = payload["config"]
            self.cfg = load_config()
            for section in ("data", "model", "train"):
                if section in raw:
                    setattr(self.cfg, section, _restore_section(section, raw[section]))
            self.cfg.device = device

        self.device = pick_device(device)
        self.model = create_model(
            self.cfg.model.name,
            num_classes=len(self.class_names),
            pretrained=False,
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = build_inference_transform(self.cfg.data)

    def predict(self, image: Image.Image, top_k: int = 1, tta: bool = False) -> list[tuple[str, float]]:
        """Return [(class_name, probability)] sorted descending, limited to top_k.

        TTA (test-time augmentation): average the softmax over the original and
        horizontally-flipped views. Averaging probabilities reduces variance from
        augmentation-sensitive decisions — a cheap accuracy boost at inference.
        """
        img = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.predict_probs(img, tta=tta)[0]
        top = torch.topk(probs, k=min(top_k, len(self.class_names)))
        return [(self.class_names[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist(), strict=True)]

    def predict_path(self, path: str | Path, top_k: int = 1, tta: bool = False) -> list[tuple[str, float]]:
        return self.predict(Image.open(path), top_k=top_k, tta=tta)

    def predict_probs(self, images: torch.Tensor, tta: bool = False) -> torch.Tensor:
        """Return per-sample class probabilities for a batched, preprocessed tensor.

        Used by ``evaluate --tta`` so both the CLI and the single-image path share
        the same TTA logic.
        """
        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(images), dim=1)
            if tta:
                probs = probs + torch.softmax(self.model(torch.flip(images, dims=[3])), dim=1)
                probs = probs / 2.0
        return probs


def _restore_section(section: str, data: dict[str, Any]) -> Any:
    from ..config import TrainConfig

    cls = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig}[section]
    valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in valid})
