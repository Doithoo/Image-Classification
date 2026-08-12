"""Model registry: a single place to create models by name.

The registry decouples experiments from model code: a config only references a
registry key, never an import. Both timm-backed models and the legacy hand-written
implementations register here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

Factory = Callable[..., nn.Module]
_REGISTRY: dict[str, Factory] = {}


def register(name: str) -> Callable[[Factory], Factory]:
    """Decorator: register a model factory under ``name``."""

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
    return sorted(_REGISTRY)


def get_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
