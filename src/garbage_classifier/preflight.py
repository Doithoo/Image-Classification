"""Non-mutating checks that explain why a training request can or cannot start."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data.manifest import DatasetMetadata
from .models.registry import model_spec
from .utils import pick_device


@dataclass(frozen=True)
class PreflightIssue:
    field: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...]
    notices: tuple[str, ...]

    def raise_for_issues(self) -> None:
        if self.issues:
            details = "\n".join(f"- {issue.field}: {issue.message}" for issue in self.issues)
            raise PreflightError(f"training preflight failed:\n{details}")


class PreflightError(ValueError):
    """Raised when a training request cannot safely start."""


def validate_training_request(
    cfg: ExperimentConfig,
    metadata: DatasetMetadata,
    run_dir: Path,
) -> PreflightReport:
    """Aggregate independent errors before downloads, model construction or writes."""
    issues: list[PreflightIssue] = []
    notices: list[str] = []
    for split in ("train", "valid", "test"):
        if not (cfg.data.manifest_dir / f"{split}.csv").is_file():
            issues.append(PreflightIssue("data.manifest_dir", f"missing {split}.csv"))
    if metadata.split_counts["train"] == 0:
        issues.append(PreflightIssue("data.manifest_dir", "train split is empty"))
    if metadata.split_counts["valid"] == 0:
        issues.append(PreflightIssue("data.manifest_dir", "valid split is empty"))
    if cfg.model.num_classes is not None and cfg.model.num_classes != len(metadata.classes):
        issues.append(
            PreflightIssue(
                "model.num_classes",
                f"configured {cfg.model.num_classes}, prepared data requires {len(metadata.classes)}",
            )
        )
    if cfg.device == "cuda" and not torch.cuda.is_available():
        issues.append(PreflightIssue("device", "CUDA was requested but is unavailable"))
    if cfg.device == "mps" and not torch.backends.mps.is_available():
        issues.append(PreflightIssue("device", "MPS was requested but is unavailable"))
    if not is_writable_destination(run_dir):
        issues.append(PreflightIssue("output_dir", f"cannot write below {run_dir.parent}"))
    try:
        model_spec(cfg.model.name)
    except KeyError:
        if cfg.model.factory is None:
            issues.append(PreflightIssue("model.name", f"unknown registered model {cfg.model.name!r}"))
    if cfg.model.pretrained:
        notices.append("pretrained weights may require network access unless already cached by the model provider")
    device = pick_device(cfg.device)
    if cfg.train.amp and device.type == "mps":
        notices.append(
            "AMP on MPS uses the selected PyTorch autocast implementation; confirm numerical behavior per run"
        )
    if cfg.train.amp and device.type == "cpu":
        notices.append("AMP is disabled on CPU")
    if cfg.train.early_stop_patience == 0:
        notices.append("early stopping is disabled because train.early_stop_patience=0")
    return PreflightReport(tuple(issues), tuple(notices))


def is_writable_destination(path: Path) -> bool:
    """Check an existing parent without creating the requested experiment directory."""
    candidate = path.parent
    while not candidate.exists():
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)
