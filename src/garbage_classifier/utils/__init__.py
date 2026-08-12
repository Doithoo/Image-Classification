"""Small shared utilities: seeding, device selection, git info, logging."""

from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed every RNG used by the stack (python, numpy, torch, cuda/mps)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS uses torch.manual_seed; no separate API needed.


def pick_device(preference: str = "auto") -> torch.device:
    """Resolve device: auto prefers cuda, then mps, then cpu."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_revision(repo_root: str | Path | None = None) -> str:
    """Return the short git revision of the repository, or 'unknown'.

    Probes the current working directory first (works for both editable and
    installed packages), falling back to the given ``repo_root``.
    """
    candidates = [Path.cwd()] + ([Path(repo_root)] if repo_root else [])
    for root in candidates:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            continue
    return "unknown"


def setup_logging(level: str = "info") -> logging.Logger:
    logger = logging.getLogger("garbage_classifier")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


class CsvLogger:
    """Append key/value rows to a CSV file; used for per-epoch metrics."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            self.path.write_text(",".join(fieldnames) + "\n")

    def write(self, row: dict[str, float | int | str]) -> None:
        with self.path.open("a") as f:
            f.write(",".join(str(row.get(name, "")) for name in self.fieldnames) + "\n")
