#!/usr/bin/env python3
"""Read one manifest batch and print the tensors that enter the model.

Usage:
    python examples/02_dataset_batch.py --manifest data/manifests/train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from garbage_classifier.config import load_config
from garbage_classifier.data import ImageClassificationDataset, collate_fn
from garbage_classifier.data.transforms import build_train_transform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/train.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    cfg = load_config()
    dataset = ImageClassificationDataset(args.manifest, transform=build_train_transform(cfg.data))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    images, labels = next(iter(loader))
    print(f"dataset samples: {len(dataset)}")
    print(f"images: {tuple(images.shape)} dtype={images.dtype}")
    print(f"labels: {tuple(labels.shape)} values={labels.tolist()}")
    print(f"pixel range after normalize: [{images.min().item():.2f}, {images.max().item():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
