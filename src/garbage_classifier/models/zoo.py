"""Maintained timm and torchvision classification model specifications."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as tv_models

from .registry import ModelSpec, register


def _timm_factory(timm_name: str):
    def factory(num_classes: int, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        import timm

        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

    return factory


def _timm_spec(name: str, *, supports_gradcam: bool = True) -> ModelSpec:
    """Read timm's static pretrained config without constructing a model or fetching weights."""
    import timm

    pretrained = timm.get_pretrained_cfg(name)
    if pretrained is None:
        raise ValueError(f"timm has no pretrained configuration for {name!r}")
    input_size = int(pretrained.input_size[-1])
    crop_pct = float(pretrained.crop_pct or 1.0)
    resize_size = int(input_size / crop_pct)
    mean = tuple(float(value) for value in pretrained.mean)
    std = tuple(float(value) for value in pretrained.std)
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"model {name!r} does not expose three-channel normalization")
    return ModelSpec(
        name=name,
        provider="timm",
        upstream_name=name,
        input_size=input_size,
        resize_size=resize_size,
        normalize_mean=(mean[0], mean[1], mean[2]),
        normalize_std=(std[0], std[1], std[2]),
        interpolation=str(pretrained.interpolation),
        supports_gradcam=supports_gradcam,
    )


def _torchvision_factory(tv_name: str, head_attr: str):
    def factory(num_classes: int, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        weights = "DEFAULT" if pretrained else None
        model = getattr(tv_models, tv_name)(weights=weights, **kwargs)
        if head_attr == "fc":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif head_attr == "classifier":
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    return factory


_TIMM_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "convnext_tiny",
    "convnext_small",
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnetv2_s",
    "regnetx_002",
    "regnetx_004",
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
)

for _name in _TIMM_MODELS:
    register(
        _name,
        _timm_spec(
            _name,
            supports_gradcam=_name not in {"swin_tiny_patch4_window7_224", "vit_base_patch16_224"},
        ),
    )(_timm_factory(_name))

register(
    "tv_resnet50",
    ModelSpec(
        name="tv_resnet50",
        provider="torchvision",
        upstream_name="resnet50",
        input_size=224,
        resize_size=232,
        interpolation="bilinear",
    ),
)(_torchvision_factory("resnet50", head_attr="fc"))
