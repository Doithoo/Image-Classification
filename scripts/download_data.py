#!/usr/bin/env python3
"""Download the garbage-classification dataset.

The dataset (2,527 images, 6 classes) is not stored in git. It will be published
as a GitHub Release asset / HuggingFace dataset. Until then, point ``--data-dir``
at an existing local copy (e.g. the legacy ``Garbage_classification/`` folder).

Usage:
    python scripts/download_data.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

# TODO(data-release): replace with the real URL once the dataset is published
DATASET_URL = "https://github.com/<owner>/<repo>/releases/download/v1.0/garbage-classification.tar.gz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="destination directory")
    parser.add_argument("--url", default=DATASET_URL, help="dataset archive URL")
    args = parser.parse_args()

    out = Path(args.data_dir)
    if (out / "cardboard").exists():
        print(f"dataset already present at {out}; skipping download")
        return 0

    print(f"downloading {args.url} ...")
    archive = out.with_suffix(".tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(args.url, archive)
    shutil.unpack_archive(archive, out)
    print(f"dataset extracted to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
