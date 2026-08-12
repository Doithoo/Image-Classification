#!/usr/bin/env python3
"""Download the garbage-classification dataset (v1.0).

The dataset (2,527 images, 6 classes, 512x384 JPEG) is hosted as a GitHub
Release asset and is not stored in git. This script downloads, verifies the
SHA-256 checksum and extracts it so that ``garbage prepare-data`` can be run
against it.

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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="destination directory (class folders land here)")
    parser.add_argument("--url", default=DATASET_URL, help="dataset archive URL")
    parser.add_argument("--sha256", default=EXPECTED_SHA256, help="expected archive checksum")
    args = parser.parse_args()

    out = Path(args.data_dir)
    if out.is_dir() and any(out.iterdir()):
        print(f"dataset already present at {out}; skipping download")
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
    # macOS may create AppleDouble (._*) junk on extraction; drop it
    for entry in list(tmp.iterdir()):
        if entry.name.startswith("._"):
            entry.unlink()
            continue
        shutil.move(str(entry), out / entry.name)
    tmp.rmdir()
    archive.unlink()

    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"done: {n} files extracted to {out}")
    print("next: garbage prepare-data --set data.data_dir data/raw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
