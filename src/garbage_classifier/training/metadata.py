from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ..utils import git_revision


def build_run_metadata(
    device: torch.device,
    *,
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
) -> dict[str, Any]:
    accelerator = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()
    return {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "platform": platform.platform(),
        "device": str(device),
        "accelerator": accelerator,
        "git_revision": git_revision(),
    }


def write_run_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(metadata, sort_keys=False))
