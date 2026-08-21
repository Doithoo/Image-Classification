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
    overwrite: bool = False,
) -> dict[str, Path]:
    """Generate verified manifests and atomically publish them as one dataset contract."""
    manifests = build_manifest(
        data_dir,
        manifest_dir,
        split_ratios=split_ratios,
        seed=seed,
        validate=True,
        strict=strict,
        overwrite=overwrite,
    )
    print(f"manifests written to {manifest_dir}:")
    for split, path in manifests.items():
        print(f"  {split:6s} {path}")
    print(f"metadata: {Path(manifest_dir) / 'dataset.yaml'}")
    return manifests
