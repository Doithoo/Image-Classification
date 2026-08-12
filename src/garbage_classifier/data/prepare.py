"""Prepare-data command logic: build portable manifests from class folders."""

from __future__ import annotations

from pathlib import Path

from .manifest import build_manifest, find_duplicates


def prepare_data(
    data_dir: str | Path,
    manifest_dir: str | Path,
    split_ratios: list[float],
    seed: int,
    strict: bool = False,
) -> dict[str, Path]:
    """Generate train/valid/test CSV manifests; returns {split: manifest_path}.

    Optionally detects content-duplicate images (same bytes, different names)
    before splitting — ``strict=True`` refuses to continue when duplicates exist.
    """
    dups = find_duplicates(data_dir)
    if dups:
        n = sum(len(g) - 1 for g in dups)
        print(f"warning: {n} duplicate images found (same content, different names); e.g. {dups[0][:2]}")
        if strict:
            print("strict mode: aborting")
            raise SystemExit(1)

    manifests = build_manifest(
        data_dir,
        manifest_dir,
        split_ratios=split_ratios,
        seed=seed,
        validate=True,
    )
    print(f"manifests written to {manifest_dir}:")
    for split, path in manifests.items():
        print(f"  {split:6s} {path}")
    print(f"summary: {Path(manifest_dir) / 'summary.txt'}")
    return manifests
