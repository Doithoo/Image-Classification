"""Transforms for train / valid / test, built from a DataConfig."""

from __future__ import annotations

import torchvision.transforms as T
from torchvision import transforms

from ..config import DataConfig


def _normalize(cfg: DataConfig) -> T.Normalize:
    return transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)


def build_train_transform(cfg: DataConfig) -> T.Compose:
    ops: list[T.Transform] = [
        transforms.Resize(cfg.resize_size),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.aug == "randaug":
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [transforms.ToTensor(), _normalize(cfg)]
    return transforms.Compose(ops)


def build_eval_transform(cfg: DataConfig) -> T.Compose:
    return transforms.Compose(
        [
            transforms.Resize(cfg.resize_size),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            _normalize(cfg),
        ]
    )


def build_inference_transform(cfg: DataConfig) -> T.Compose:
    """Same as eval transform; used by predict for consistency with evaluation."""
    return build_eval_transform(cfg)
