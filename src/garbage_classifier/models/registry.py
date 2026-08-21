"""Model registry and inspectable construction specifications."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch.nn as nn

from ..config import DataConfig, ModelConfig

Factory = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    """Stable metadata for a registered model without constructing or downloading it."""

    name: str
    provider: str
    upstream_name: str
    input_size: int = 224
    resize_size: int = 256
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"
    supports_gradcam: bool = True
    target_layer: str | None = None


_REGISTRY: dict[str, tuple[ModelSpec, Factory]] = {}


def register(name: str, spec: ModelSpec | None = None) -> Callable[[Factory], Factory]:
    """Register a model factory and immutable specification under ``name``."""
    model_spec = spec or ModelSpec(name=name, provider="custom", upstream_name=name)
    if model_spec.name != name:
        raise ValueError(f"model spec name {model_spec.name!r} does not match registry key {name!r}")

    def decorator(factory: Factory) -> Factory:
        if name in _REGISTRY:
            raise ValueError(f"model already registered: {name}")
        _REGISTRY[name] = (model_spec, factory)
        return factory

    return decorator


def model_spec(name: str) -> ModelSpec:
    """Return static metadata for a registered model without building it."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available_models()}")
    return _REGISTRY[name][0]


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def available_model_specs() -> list[ModelSpec]:
    return [model_spec(name) for name in available_models()]


def create_model(
    name: str,
    num_classes: int,
    pretrained: bool = False,
    *,
    factory: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build a registered model or an explicitly named trusted extension factory."""
    factory_kwargs = dict(params or {})
    if factory is not None:
        module_name, separator, attribute = factory.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("model.factory must be a module:function string")
        try:
            function = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot import model factory {factory!r}: {exc}") from exc
        if not callable(function):
            raise ValueError(f"model factory {factory!r} is not callable")
        model = function(num_classes=num_classes, pretrained=pretrained, **factory_kwargs)
    else:
        if name not in _REGISTRY:
            raise KeyError(f"unknown model {name!r}; available: {available_models()}")
        model = _REGISTRY[name][1](num_classes=num_classes, pretrained=pretrained, **factory_kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    return model


def resolve_preprocessing(data: DataConfig, model: ModelConfig) -> DataConfig:
    """Resolve the explicit preprocessing policy before a run is recorded."""
    if model.preprocessing == "fixed":
        return data
    if model.factory is not None:
        raise ValueError("model_default preprocessing requires a registered model; external factories must use fixed")
    spec = model_spec(model.name)
    return replace(
        data,
        image_size=spec.input_size,
        resize_size=spec.resize_size,
        normalize_mean=list(spec.normalize_mean),
        normalize_std=list(spec.normalize_std),
        interpolation=spec.interpolation,
    )


def get_num_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
