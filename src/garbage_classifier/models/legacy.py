"""Legacy hand-written model implementations, registered behind the registry.

These are the 16 model families from the original ``Code/model/`` directory,
vendored without external dependencies (ptflops/torchinfo imports are guarded;
they were only used in ``__main__`` complexity-printing blocks).

Registration uses lazy factories so importing this module never touches the
legacy code — heavy imports happen only when a ``legacy_*`` model is actually
created. Legacy names are prefixed with ``legacy_`` to keep them distinct from
the maintained timm/torchvision backends.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from .registry import register

# name -> (module, factory, fixed_kwargs or None)
# ``None`` fixed kwargs means the factory takes (num_classes, pretrained=False, **kw)
_LEGACY: dict[str, tuple[str, str, dict[str, Any] | None]] = {
    "legacy_alexnet": ("alexnet", "AlexNet", None),
    "legacy_vgg11": ("vgg", "vgg", {"model_name": "vgg11"}),
    "legacy_vgg13": ("vgg", "vgg", {"model_name": "vgg13"}),
    "legacy_vgg16": ("vgg", "vgg", {"model_name": "vgg16"}),
    "legacy_vgg19": ("vgg", "vgg", {"model_name": "vgg19"}),
    "legacy_resnet18": ("resnet", "resnet18", None),
    "legacy_resnet34": ("resnet", "resnet34", None),
    "legacy_resnet50": ("resnet", "resnet50", None),
    "legacy_resnet101": ("resnet", "resnet101", None),
    "legacy_resnet152": ("resnet", "resnet152", None),
    "legacy_googlenet": ("googlenet", "GoogLeNet", {"aux_logits": False}),
    "legacy_densenet121": ("densenet", "densenet121", None),
    "legacy_densenet169": ("densenet", "densenet169", None),
    "legacy_densenet201": ("densenet", "densenet201", None),
    "legacy_densenet161": ("densenet", "densenet161", None),
    "legacy_mobilenet_v2": ("mobilenetv2", "MobileNetV2", None),
    "legacy_mobilenet_v3_small": ("mobilenetv3", "mobilenet_v3_small", None),
    "legacy_mobilenet_v3_large": ("mobilenetv3", "mobilenet_v3_large", None),
    "legacy_efficientnet_b0": ("efficientnet", "efficientnet_b0", None),
    "legacy_efficientnet_b1": ("efficientnet", "efficientnet_b1", None),
    "legacy_efficientnet_b2": ("efficientnet", "efficientnet_b2", None),
    "legacy_efficientnet_b3": ("efficientnet", "efficientnet_b3", None),
    "legacy_efficientnet_b4": ("efficientnet", "efficientnet_b4", None),
    "legacy_efficientnet_b5": ("efficientnet", "efficientnet_b5", None),
    "legacy_efficientnet_b6": ("efficientnet", "efficientnet_b6", None),
    "legacy_efficientnet_b7": ("efficientnet", "efficientnet_b7", None),
    "legacy_efficientnetv2_s": ("efficientnet_v2", "efficientnetv2_s", None),
    "legacy_efficientnetv2_m": ("efficientnet_v2", "efficientnetv2_m", None),
    "legacy_efficientnetv2_l": ("efficientnet_v2", "efficientnetv2_l", None),
    "legacy_regnet": ("regnet", "regnet", None),
    "legacy_shufflenet_v2_x0_5": ("shufflenet", "shufflenet_v2_x0_5", None),
    "legacy_shufflenet_v2_x1_0": ("shufflenet", "shufflenet_v2_x1_0", None),
    "legacy_convnext_tiny": ("convnext", "convnext_tiny", None),
    "legacy_convnext_small": ("convnext", "convnext_small", None),
    "legacy_convnext_base": ("convnext", "convnext_base", None),
    "legacy_convnext_large": ("convnext", "convnext_large", None),
    "legacy_vit_base_patch16_224": ("vit", "vit_base_patch16_224", None),
    "legacy_swin_tiny": ("swin", "swin_tiny_patch4_window7_224", None),
}


def _make_factory(module: str, fn_name: str, fixed: dict[str, Any] | None) -> Callable[..., Any]:
    def factory(num_classes: int, pretrained: bool = False, **kwargs: Any):
        # pretrained is accepted for interface parity; legacy models have no
        # maintained pretrained weights, so it is ignored (documented).
        mod = importlib.import_module(f".legacy_models.{module}", __package__)
        fn = getattr(mod, fn_name)
        kw = dict(fixed or {})
        kw.update(kwargs)
        return fn(num_classes=num_classes, **kw)

    return factory


def register_legacy_models() -> None:
    for name, (module, fn_name, fixed) in _LEGACY.items():
        register(name)(_make_factory(module, fn_name, fixed))


register_legacy_models()
