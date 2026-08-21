"""Machine-readable environment and provenance metadata for completed runs."""

from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml

from ..utils import git_revision, write_text_atomic


def build_run_metadata(
    device: torch.device,
    *,
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
    manifest_identity: str | None = None,
    dataset_schema_version: int | None = None,
) -> dict[str, Any]:
    accelerator = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "platform": platform.platform(),
        "device": str(device),
        "accelerator": accelerator,
        "git_revision": git_revision(),
    }
    if manifest_identity is not None:
        metadata["manifest_identity"] = manifest_identity
    if dataset_schema_version is not None:
        metadata["dataset_schema_version"] = dataset_schema_version
    return metadata


def write_run_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    write_text_atomic(Path(path), yaml.safe_dump(metadata, sort_keys=False))
