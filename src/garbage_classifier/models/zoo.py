"""Model zoo: maintained timm backbones plus a torchvision baseline.

Per the refactoring plan, timm/torchvision maintained models are the default
choice; the legacy hand-written implementations are migrated behind the same
registry interface in a later phase.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as tv_models

from .registry import register


def _timm_factory(timm_name: str):
    """Build a factory that wraps a timm model with a fresh classifier head."""

    def factory(num_classes: int, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        import timm  # imported lazily so timm is only required when used

        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        return model

    return factory


def _torchvision_factory(tv_name: str, head_attr: str | None = None):
    """Build a factory wrapping a torchvision model, replacing its classifier head."""

    def factory(num_classes: int, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        weights = "DEFAULT" if pretrained else None
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
    "shufflenet_v2_x0_5",
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
]:
    register(_name)(_timm_factory(_name))

# ---- torchvision baseline (strict parity with the historical README results) ----
register("tv_resnet50")(_torchvision_factory("resnet50", head_attr="fc"))

# legacy hand-written implementations are registered in garbage_classifier.models.legacy
from . import legacy  # noqa: E402,F401
