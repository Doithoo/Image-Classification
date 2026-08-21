"""Models package: registry, inspectable specifications and built-in zoo."""

from . import zoo  # noqa: F401
from .registry import (
    ModelSpec,
    available_model_specs,
    available_models,
    create_model,
    get_num_parameters,
    model_spec,
    register,
    resolve_preprocessing,
)

__all__ = [
    "ModelSpec",
    "available_model_specs",
    "available_models",
    "create_model",
    "get_num_parameters",
    "model_spec",
    "register",
    "resolve_preprocessing",
]
