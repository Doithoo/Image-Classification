"""Prepare-data command logic: build portable manifests from class folders."""

from __future__ import annotations

from pathlib import Path

from .manifest import build_manifest


def prepare_data(
    data_dir: str | Path,
    manifest_dir: str | Path,
    split_ratios: list[float],
    seed: int,
    strict: bool = False,
) -> dict[str, Path]:
    """Generate train/valid/test CSV manifests; returns {split: manifest_path}.

    Content-identical images stay in one split. ``strict=True`` refuses any
    duplicate, while cross-class duplicates always fail as annotation conflicts.
    """
    manifests = build_manifest(
        data_dir,
        manifest_dir,
        split_ratios=split_ratios,
        seed=seed,
        validate=True,
        strict=strict,
    )
    print(f"manifests written to {manifest_dir}:")
    for split, path in manifests.items():
        print(f"  {split:6s} {path}")
    print(f"summary: {Path(manifest_dir) / 'summary.txt'}")
    return manifests
