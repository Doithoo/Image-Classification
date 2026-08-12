"""Models package: registry and model zoo."""

from . import zoo  # noqa: F401  (populates the registry)
from .registry import available_models, create_model, get_num_parameters, register

__all__ = ["register", "create_model", "available_models", "get_num_parameters"]
