"""Small shared utilities: reproducibility, atomic publication and logging."""

from __future__ import annotations

import csv
import io
import logging
import os
import random
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed every RNG used by the stack (python, numpy, torch and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preference: str = "auto") -> torch.device:
    """Resolve device: auto prefers CUDA, then MPS, then CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_text_atomic(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write one text file atomically, never exposing a partial result."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_directory(stage: Path, destination: Path, *, overwrite: bool = False) -> None:
    """Atomically publish a fully-built directory, optionally replacing an old one."""
    if not stage.is_dir():
        raise ValueError(f"staging directory does not exist: {stage}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"output destination is not a directory: {destination}")
    if destination.exists():
        if any(destination.iterdir()) and not overwrite:
            raise FileExistsError(f"output directory already exists and is non-empty: {destination}")
        if any(destination.iterdir()):
            backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
            os.replace(destination, backup)
        else:
            destination.rmdir()
    try:
        os.replace(stage, destination)
    except OSError:
        if backup is not None and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def git_revision(repo_root: str | Path | None = None) -> str:
    """Return the short git revision of the repository, or ``unknown``."""
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
        except OSError:
            continue
    return "unknown"


def file_sha256(path: str | Path) -> str:
    """Return the full SHA-256 digest of a file without loading it into memory."""
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Atomically append a row to a small metrics CSV with proper CSV escaping."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = fieldnames
        if not self.path.exists():
            self._write_rows([])

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        write_text_atomic(self.path, output.getvalue())

    def reconcile_epoch(self, completed_epoch: int) -> None:
        """Trim rows beyond a resumed checkpoint and reject missing or duplicate history."""
        rows: list[dict[str, str]] = []
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        kept = [row for row in rows if int(row["epoch"]) <= completed_epoch]
        epochs = [int(row["epoch"]) for row in kept]
        if epochs != list(range(1, completed_epoch + 1)):
            raise ValueError(
                f"metrics history {self.path} does not match resumed checkpoint epoch {completed_epoch}: {epochs}"
            )
        if kept != rows:
            self._write_rows(kept)

    def write(self, row: Mapping[str, float | int | str]) -> None:
        rows: list[dict[str, str]] = []
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        rows.append({name: str(row.get(name, "")) for name in self.fieldnames})
        self._write_rows(rows)
