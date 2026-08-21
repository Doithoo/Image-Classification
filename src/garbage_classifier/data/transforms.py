"""Transforms for train / valid / test, built from a DataConfig.

Learning note — why train and validation use DIFFERENT transforms:
  - Training needs *diversity*: random crops, flips and augmentations present a
    slightly different image every epoch, which is what makes the model robust
    instead of memorizing exact pixels.
  - Validation/test need *determinism*: the same fixed preprocessing for every
    image, so the reported metric is comparable across experiments.
"""

from __future__ import annotations

import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..config import DataConfig


def _interpolation(cfg: DataConfig) -> InterpolationMode:
    return InterpolationMode.BILINEAR if cfg.interpolation == "bilinear" else InterpolationMode.BICUBIC


def _normalize(cfg: DataConfig) -> T.Normalize:
    # Pixels are in [0,1] after ToTensor; Normalize shifts them to approx.
    # standard normal (mean 0, std 1), which helps the optimizer converge.
    return transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)


def build_train_transform(cfg: DataConfig) -> T.Compose:
    ops: list[T.Transform] = [
        # scale the 512x384 image so the short side is 256 (keeps aspect ratio)
        transforms.Resize(cfg.resize_size, interpolation=_interpolation(cfg)),
        # crop a random 224x224 region with scale 0.6-1.0 of the image:
        # random location + random zoom = "see the object in many ways"
        transforms.RandomResizedCrop(
            cfg.image_size,
            scale=(0.6, 1.0),
            interpolation=_interpolation(cfg),
        ),
        # 50% chance of mirroring — objects are still the same class when flipped
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.aug == "randaug":
        # learned-style policy: 2 random augmentation ops (contrast, rotate, ...)
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [
        # HWC uint8 [0,255]  ->  CHW float32 [0,1]
        transforms.ToTensor(),
        _normalize(cfg),
    ]
    return transforms.Compose(ops)


def build_eval_transform(cfg: DataConfig) -> T.Compose:
    return transforms.Compose(
        [
            transforms.Resize(cfg.resize_size, interpolation=_interpolation(cfg)),
            # deterministic crop: always take the center 224x224
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            _normalize(cfg),
        ]
    )


def build_inference_transform(cfg: DataConfig) -> T.Compose:
    """Same as eval transform; used by predict for consistency with evaluation."""
    return build_eval_transform(cfg)
