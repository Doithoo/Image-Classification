#!/usr/bin/env python3
"""Download and audit-patch the garbage-classification dataset (v1.0).

The upstream dataset (2,527 images, 6 classes, 512x384 JPEG) is hosted as a
GitHub Release asset and is not stored in git. This script downloads, verifies
the SHA-256 checksum, extracts it, and removes hash-verified files listed in the
recorded audit patch before ``garbage prepare-data`` is run.

Usage:
    python scripts/download_data.py                 # -> data/raw/
    python scripts/download_data.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

DATASET_URL = "https://github.com/Doithoo/Image-Classification/releases/download/v1.0/garbage-classification.tar.gz"
EXPECTED_SHA256 = "be0b99fc61360cf267f8be4e0c10d1d2dc23f141b6fbeac20122468bb81ea1b6"
DATASET_PATCH_VERSION = "v1.0-audit.1"
DATASET_AUDIT_REMOVALS = {
    # Human-reviewed annotation errors in the checksum-verified v1.0 archive.
    "metal/metal91.jpg": "81546d1362d75fc60718e5852593c58f66f813e8c1e9d1eae6871fedb02e2868",
    "plastic/plastic152.jpg": "c41b99aec8a3257c8668a2144ebbaf8042018f0a85d88e22bf4e413b43693dd8",
    # Exact duplicate of glass/glass389.jpg; retain the reviewed glass label.
    "plastic/plastic332.jpg": "e971f4f8f50e960e454ea724cec922fc2988bbf4027aa48b553e139bb5890968",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_dataset_audit(data_dir: Path) -> list[str]:
    """Remove reviewed v1 files, but only when every present hash matches."""
    present: list[tuple[str, Path]] = []
    for relative_path, expected_hash in DATASET_AUDIT_REMOVALS.items():
        path = data_dir / relative_path
        if not path.exists():
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{relative_path} checksum mismatch while applying {DATASET_PATCH_VERSION}: "
                f"expected {expected_hash}, got {actual_hash}; no files were removed"
            )
        present.append((relative_path, path))

    for _, path in present:
        path.unlink()
    return [relative_path for relative_path, _ in present]


def _move_extracted_entries(extracted_dir: Path, destination: Path) -> None:
    """Move verified archive entries into a newly-created destination."""
    destination.mkdir(parents=True, exist_ok=True)
    for entry in list(extracted_dir.iterdir()):
        if entry.name.startswith("._"):
            entry.unlink()
            continue
        shutil.move(str(entry), destination / entry.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="destination directory (class folders land here)")
    parser.add_argument("--url", default=DATASET_URL, help="dataset archive URL")
    parser.add_argument("--sha256", default=EXPECTED_SHA256, help="expected archive checksum")
    args = parser.parse_args()

    out = Path(args.data_dir)
    if out.is_dir() and any(out.iterdir()):
        removed = apply_dataset_audit(out)
        print(f"dataset already present at {out}; skipping download")
        print(f"dataset patch {DATASET_PATCH_VERSION}: removed {len(removed)} audited files")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    archive = out.parent / "garbage-classification.tar.gz"

    print(f"downloading {args.url} ...")
    urllib.request.urlretrieve(args.url, archive)

    actual = _sha256(archive)
    if actual != args.sha256:
        archive.unlink(missing_ok=True)
        raise SystemExit(f"checksum mismatch: expected {args.sha256}, got {actual}")

    print(f"checksum OK ({actual}) — extracting ...")
    tmp = out.parent / ".garbage-extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.unpack_archive(archive, tmp)
    _move_extracted_entries(tmp, out)
    tmp.rmdir()
    archive.unlink()

    removed = apply_dataset_audit(out)
    print(f"dataset patch {DATASET_PATCH_VERSION}: removed {len(removed)} audited files")

    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"done: {n} files extracted to {out}")
    print("next: garbage prepare-data --set data.data_dir data/raw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
