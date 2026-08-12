"""Model registry — a single place to create models by name.

Learning note — the factory/registry pattern:
  - A *factory* is a function that *builds* a model: ``build(num_classes, pretrained)``.
  - The *registry* is a dict mapping a string name to a factory.
  - Training code only ever says ``create_model("resnet50", num_classes=6)`` — it
    never imports the model class directly. This is what makes experiments
    **config-driven**: change ``model.name`` in YAML, the factory changes, the
    rest of the pipeline stays untouched.

Why not just ``import resnet50`` at the call site?
  - Every model variant would need an if/else chain ("if name == 'resnet50': ..."),
    and adding a model means editing that chain everywhere. With a registry,
    adding a model means adding one line at registration time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

# name -> factory. Factories have the signature:
#   factory(num_classes: int, pretrained: bool = False, **kwargs) -> nn.Module
Factory = Callable[..., nn.Module]
_REGISTRY: dict[str, Factory] = {}


def register(name: str) -> Callable[[Factory], Factory]:
    """Decorator: register a model factory under ``name``.

    Usage::

        @register("my_model")
        def build(num_classes, pretrained=False, **kwargs):
            return MyModel(num_classes=num_classes)
    """

    def decorator(fn: Factory) -> Factory:
        if name in _REGISTRY:
            raise ValueError(f"model already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def create_model(name: str, num_classes: int, pretrained: bool = False, **kwargs: Any) -> nn.Module:
    """Build a model from the registry. Raises KeyError for unknown names."""
    if name not in _REGISTRY:
        known = sorted(_REGISTRY)
        raise KeyError(f"unknown model {name!r}; available: {known}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)


def available_models() -> list[str]:
    """All registry keys (useful for `garbage bench` and debugging)."""
    return sorted(_REGISTRY)


def get_num_parameters(model: nn.Module) -> int:
    """Total trainable + frozen parameter count (params * 4 bytes ~= memory)."""
    return sum(p.numel() for p in model.parameters())
