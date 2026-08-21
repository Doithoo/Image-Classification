"""Versioned, atomically saved and tensor-only checkpoints."""

from __future__ import annotations

import os
import pickle
import random
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig, config_from_dict, load_config, to_dict
from ..utils import git_revision

CHECKPOINT_SCHEMA_VERSION = 2


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot safely restore the requested operation."""


@dataclass(frozen=True)
class ResumeIdentity:
    """Fields that must remain stable when continuing an optimization trajectory."""

    model: dict[str, Any]
    class_names: tuple[str, ...]
    manifest_identity: str
    preprocessing: dict[str, Any]


def preprocessing_contract(cfg: ExperimentConfig) -> dict[str, Any]:
    """Return exact input preprocessing that must match checkpoint weights."""
    return {
        "image_size": cfg.data.image_size,
        "resize_size": cfg.data.resize_size,
        "normalize_mean": list(cfg.data.normalize_mean),
        "normalize_std": list(cfg.data.normalize_std),
        "interpolation": cfg.data.interpolation,
        "color_space": "RGB",
        "input_range": [0.0, 1.0],
        "augmentation": cfg.data.aug,
        "preprocessing_policy": cfg.model.preprocessing,
    }


def model_contract(cfg: ExperimentConfig) -> dict[str, Any]:
    """Return model construction fields that are relevant to checkpoint compatibility."""
    return {
        "name": cfg.model.name,
        "num_classes": cfg.model.num_classes,
        "params": dict(cfg.model.params),
        "factory": cfg.model.factory,
    }


def build_resume_identity(
    cfg: ExperimentConfig,
    class_names: list[str],
    manifest_identity: str,
) -> ResumeIdentity:
    return ResumeIdentity(model_contract(cfg), tuple(class_names), manifest_identity, preprocessing_contract(cfg))


def _pack_python_rng_state(state: tuple[Any, ...]) -> dict[str, Any]:
    version, values, gaussian = state
    return {"version": int(version), "state": torch.tensor(values, dtype=torch.int64), "gaussian": gaussian}


def _unpack_python_rng_state(state: Mapping[str, Any]) -> tuple[Any, ...]:
    values = state.get("state")
    if not isinstance(values, torch.Tensor):
        raise CheckpointCompatibilityError("checkpoint Python RNG state is invalid")
    return int(state["version"]), tuple(int(value) for value in values.tolist()), state.get("gaussian")


def _pack_numpy_rng_state(state: Any) -> dict[str, Any]:
    algorithm, values, position, has_gauss, cached_gaussian = state
    return {
        "algorithm": str(algorithm),
        "state": torch.from_numpy(np.asarray(values, dtype=np.uint32).copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _unpack_numpy_rng_state(state: Mapping[str, Any]) -> tuple[Any, ...]:
    values = state.get("state")
    if not isinstance(values, torch.Tensor):
        raise CheckpointCompatibilityError("checkpoint NumPy RNG state is invalid")
    return (
        str(state["algorithm"]),
        values.detach().cpu().numpy().astype(np.uint32, copy=False),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": _pack_python_rng_state(random.getstate()),
        "numpy": _pack_numpy_rng_state(np.random.get_state()),
        "torch": torch.get_rng_state(),
        "cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore an RNG state written by checkpoint schema v2."""
    if "python" in state:
        python_state = state["python"]
        if not isinstance(python_state, Mapping):
            raise CheckpointCompatibilityError("checkpoint Python RNG state is invalid")
        random.setstate(_unpack_python_rng_state(python_state))
    if "numpy" in state:
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, Mapping):
            raise CheckpointCompatibilityError("checkpoint NumPy RNG state is invalid")
        np.random.set_state(_unpack_numpy_rng_state(numpy_state))
    if "torch" in state:
        torch_state = state["torch"]
        if not isinstance(torch_state, torch.Tensor):
            raise CheckpointCompatibilityError("checkpoint Torch RNG state is invalid")
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and state.get("cuda"):
        cuda_state = state["cuda"]
        if not isinstance(cuda_state, list) or not all(isinstance(value, torch.Tensor) for value in cuda_state):
            raise CheckpointCompatibilityError("checkpoint CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    deployable_state_dict: dict[str, torch.Tensor] | None = None,
    ema: Any | None = None,
    scaler: Any | None = None,
    patience_left: int | None = None,
    epoch: int,
    best_metric: float,
    cfg: ExperimentConfig,
    class_names: list[str],
    manifest_identity: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist full state using a schema accepted by ``torch.load(weights_only=True)``."""
    if not manifest_identity or manifest_identity == "unverified":
        raise ValueError("a verified manifest_identity is required for new checkpoints")
    if not class_names or len(set(class_names)) != len(class_names) or any(not name for name in class_names):
        raise ValueError("class_names must be non-empty and unique")
    if cfg.model.num_classes != len(class_names):
        raise ValueError("resolved model.num_classes must match class_names before checkpointing")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    training_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    deployable_state = deployable_state_dict or training_state
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": deployable_state,
        "training_model_state_dict": training_state,
        "deployable_model_state_dict": deployable_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "patience_left": patience_left,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": to_dict(cfg),
        "model": model_contract(cfg),
        "preprocessing": preprocessing_contract(cfg),
        "class_names": list(class_names),
        "manifest_identity": manifest_identity,
        "git_revision": git_revision(),
        "rng_state": _rng_state(),
        "extra": extra or {},
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_tensor_only(path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, OSError, RuntimeError, EOFError) as exc:
        raise CheckpointCompatibilityError(
            f"cannot safely load checkpoint {path}: {exc}. Re-export it with checkpoint schema v2."
        ) from exc
    if not isinstance(loaded, dict):
        raise CheckpointCompatibilityError(f"checkpoint {path} must contain a mapping")
    return loaded


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Safely load and validate a v2 checkpoint or a tensor-only legacy checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = _load_tensor_only(checkpoint_path)
    schema_version = payload.get("schema_version")
    if schema_version is None:
        required = {"config", "class_names"}
        missing = required - set(payload)
        if missing:
            raise CheckpointCompatibilityError(
                f"legacy checkpoint {checkpoint_path} is missing metadata: {sorted(missing)}"
            )
        payload["legacy_checkpoint"] = True
        state = deployable_model_state(payload)
        payload.setdefault("training_model_state_dict", state)
        payload.setdefault("deployable_model_state_dict", state)
        payload.setdefault("ema_state_dict", None)
        payload.setdefault("scaler_state_dict", None)
        payload.setdefault("patience_left", None)
        return payload
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint {checkpoint_path} has unsupported schema_version {schema_version!r}"
        )
    required = {
        "config",
        "model",
        "preprocessing",
        "class_names",
        "manifest_identity",
        "rng_state",
        "training_model_state_dict",
        "deployable_model_state_dict",
    }
    missing = required - set(payload)
    if missing:
        raise CheckpointCompatibilityError(f"checkpoint {checkpoint_path} is missing fields: {sorted(missing)}")
    if not isinstance(payload["class_names"], list) or not all(
        isinstance(value, str) and value for value in payload["class_names"]
    ):
        raise CheckpointCompatibilityError("checkpoint class_names must be a non-empty list of non-empty strings")
    if not payload["class_names"] or len(set(payload["class_names"])) != len(payload["class_names"]):
        raise CheckpointCompatibilityError("checkpoint class_names must be non-empty and unique")
    if not isinstance(payload["manifest_identity"], str) or not payload["manifest_identity"]:
        raise CheckpointCompatibilityError("checkpoint manifest_identity must be a non-empty string")
    for key in ("model", "preprocessing", "config", "rng_state"):
        if not isinstance(payload[key], Mapping):
            raise CheckpointCompatibilityError(f"checkpoint {key} must be a mapping")
    try:
        stored_cfg = config_from_dict(payload["config"], allow_legacy=True)
    except ValueError as exc:
        raise CheckpointCompatibilityError(f"checkpoint config is invalid: {exc}") from exc
    if dict(payload["model"]) != model_contract(stored_cfg):
        raise CheckpointCompatibilityError("checkpoint model contract does not match its resolved config")
    if dict(payload["preprocessing"]) != preprocessing_contract(stored_cfg):
        raise CheckpointCompatibilityError("checkpoint preprocessing contract does not match its resolved config")
    if stored_cfg.model.num_classes != len(payload["class_names"]):
        raise CheckpointCompatibilityError("checkpoint class count does not match its resolved model config")
    state = deployable_model_state(payload)
    training_state = payload["training_model_state_dict"]
    if not isinstance(training_state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in training_state.items()
    ):
        raise CheckpointCompatibilityError("checkpoint training_model_state_dict must be a tensor mapping")
    payload.setdefault("model_state_dict", state)
    payload.setdefault("training_model_state_dict", state)
    payload.setdefault("deployable_model_state_dict", state)
    payload.setdefault("ema_state_dict", None)
    payload.setdefault("scaler_state_dict", None)
    payload.setdefault("patience_left", None)
    return payload


def validate_resume_identity(checkpoint: Mapping[str, Any], expected: ResumeIdentity) -> None:
    """Reject resumes that would silently change data, model classes or preprocessing."""
    if checkpoint.get("legacy_checkpoint"):
        raise CheckpointCompatibilityError("legacy checkpoints cannot resume schema-v2 experiments")
    mismatches: list[str] = []
    if dict(checkpoint.get("model", {})) != expected.model:
        mismatches.append("model")
    names = checkpoint.get("class_names")
    if (tuple(names) if isinstance(names, list | tuple) else ()) != expected.class_names:
        mismatches.append("class_names")
    if checkpoint.get("manifest_identity") != expected.manifest_identity:
        mismatches.append("manifest_identity")
    if dict(checkpoint.get("preprocessing", {})) != expected.preprocessing:
        mismatches.append("preprocessing")
    if mismatches:
        raise CheckpointCompatibilityError("resume identity mismatch: " + ", ".join(mismatches))


def restore_config_from_checkpoint(payload: Mapping[str, Any]) -> ExperimentConfig:
    """Restore stored config while accepting the former unused ``data.classes`` key."""
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise CheckpointCompatibilityError("checkpoint is missing config metadata")
    try:
        return config_from_dict(raw, allow_legacy=True)
    except ValueError as exc:
        raise CheckpointCompatibilityError(f"checkpoint config is invalid: {exc}") from exc


def validate_inference_model_source(
    payload: Mapping[str, Any],
    trusted_config_path: str | Path | None = None,
) -> ExperimentConfig:
    """Require an explicit matching config before importing an external model factory."""
    checkpoint_cfg = restore_config_from_checkpoint(payload)
    if trusted_config_path is None:
        if checkpoint_cfg.model.factory is not None:
            raise CheckpointCompatibilityError(
                "checkpoint uses an external model factory; pass the reviewed training config explicitly"
            )
        return checkpoint_cfg

    requested = load_config(trusted_config_path)
    expected = model_contract(checkpoint_cfg)
    actual = model_contract(requested)
    for key in ("name", "params", "factory"):
        if actual[key] != expected[key]:
            raise CheckpointCompatibilityError(
                f"trusted config {trusted_config_path} changes checkpoint model field {key!r}"
            )
    return checkpoint_cfg


def deployable_model_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Select inference weights, with a fallback for minimal historical checkpoints."""
    state = payload.get("deployable_model_state_dict") or payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise CheckpointCompatibilityError("checkpoint is missing a tensor model state")
    return state
