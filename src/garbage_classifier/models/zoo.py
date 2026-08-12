"""Model zoo: maintained timm backbones plus a torchvision baseline.

Learning note — why timm?
  - timm (PyTorch Image Models) hosts hundreds of *maintained* CNN/Transformer
    architectures WITH pretrained weights and a uniform API. Using it means we
    get battle-tested code and ImageNet-pretrained weights for free.
  - torchvision provides the classic reference implementations (ResNet etc.) —
    we keep one (`tv_resnet50`) as a parity baseline against the timm version.

Every entry here registers into the registry under a key; experiments select
models by key via `model.name` in config.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as tv_models

from .registry import register


def _timm_factory(timm_name: str):
    """Return a factory that builds the timm model with a fresh classifier head.

    ``timm.create_model(name, pretrained=..., num_classes=6)`` does two things:
      1. builds the backbone with ImageNet-pretrained weights (if requested)
      2. REPLACES its final classifier with a new Linear layer for 6 classes —
         the "classification head" that gets trained from scratch on our data.
    """

    def factory(num_classes: int, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        import timm  # imported lazily so timm is only required when used

        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    return factory


def _torchvision_factory(tv_name: str, head_attr: str | None = None):
    """Return a factory wrapping a torchvision model, replacing its classifier head.

    torchvision models name their final layer differently:
      - ResNet uses ``model.fc`` (a single Linear)
      - VGG/AlexNet use ``model.classifier`` (a Sequential; we replace its last layer)
    ``head_attr`` tells us which one to swap, keeping `in_features` intact.
    """

    def factory(num_classes: int, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        weights = "DEFAULT" if pretrained else None  # "DEFAULT" = best pretrained weights
        model = getattr(tv_models, tv_name)(weights=weights)
        if head_attr == "fc":
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif head_attr == "classifier":
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    return factory


# ---- timm models (maintained, pretrained) ----
# 每行注册一个模型：名字 -> 能造出该模型的工厂函数。
# 新增模型只需在这里加一行（前提是 timm 支持它，见教程 3）。
for _name in [
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
]:
    register(_name)(_timm_factory(_name))

# ---- torchvision baseline (strict parity with the historical README results) ----
register("tv_resnet50")(_torchvision_factory("resnet50", head_attr="fc"))

# legacy hand-written implementations are registered in garbage_classifier.models.legacy
from . import legacy  # noqa: E402,F401
