"""Comparison of compatible completed runs using validation evidence."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..utils import write_text_atomic


@dataclass(frozen=True)
class RunComparisonRow:
    run_name: str
    run_dir: Path
    epoch: int
    metric: str
    metric_value: float
    manifest_identity: str
    device: str


def _read_best_metric(run_dir: Path, metric: str) -> tuple[int, float]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        raise ValueError(f"missing metrics.csv in {run_dir}")
    column = {"balanced_accuracy": "balanced_acc", "macro_f1": "macro_f1", "accuracy": "accuracy"}.get(metric, metric)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in (rows[0] if rows else {}):
        raise ValueError(f"{path}: metric {metric!r} is not available")
    best = max(rows, key=lambda row: float(row[column]))
    return int(best["epoch"]), float(best[column])


def compare_runs(run_dirs: list[str | Path], metric: str = "macro_f1") -> list[RunComparisonRow]:
    """Rank completed runs, rejecting comparisons across different prepared data."""
    if len(run_dirs) < 2:
        raise ValueError("compare-runs needs at least two run directories")
    rows: list[RunComparisonRow] = []
    identities: set[str] = set()
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        metadata_path = run_dir / "run.yaml"
        if not metadata_path.is_file():
            raise ValueError(f"missing run.yaml in {run_dir}")
        raw_metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        identity = raw_metadata.get("manifest_identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{metadata_path}: manifest_identity is required")
        epoch, value = _read_best_metric(run_dir, metric)
        identities.add(identity)
        rows.append(
            RunComparisonRow(
                run_name=run_dir.name,
                run_dir=run_dir,
                epoch=epoch,
                metric=metric,
                metric_value=value,
                manifest_identity=identity,
                device=str(raw_metadata.get("device", "unknown")),
            )
        )
    if len(identities) != 1:
        raise ValueError("cannot compare runs with different manifest_identity values")
    return sorted(rows, key=lambda row: row.metric_value, reverse=True)


def write_comparison(path: str | Path, rows: list[RunComparisonRow]) -> Path:
    """Publish a CSV comparison atomically without silently replacing evidence."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"comparison output already exists: {destination}")
    output = io.StringIO(newline="")
    fieldnames = ["rank", "run_name", "run_dir", "epoch", "metric", "metric_value", "device", "manifest_identity"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for rank, row in enumerate(rows, start=1):
        writer.writerow(
            {
                "rank": rank,
                "run_name": row.run_name,
                "run_dir": row.run_dir,
                "epoch": row.epoch,
                "metric": row.metric,
                "metric_value": row.metric_value,
                "device": row.device,
                "manifest_identity": row.manifest_identity,
            }
        )
    write_text_atomic(destination, output.getvalue())
    return destination
