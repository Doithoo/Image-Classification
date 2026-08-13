#!/usr/bin/env python3
"""Show the smallest image -> tensor -> logits -> probability flow.

Usage:
    python examples/01_tensor_flow.py
    python examples/01_tensor_flow.py --image data/raw/paper/paper1.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from garbage_classifier.config import GARBAGE_CLASSES, load_config
from garbage_classifier.data.transforms import build_inference_transform
from garbage_classifier.models import create_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="optional image; otherwise use a random tensor")
    parser.add_argument("--model", default="resnet18", help="registry model name")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.image is None:
        images = torch.rand(1, 3, cfg.data.image_size, cfg.data.image_size)
    else:
        if not args.image.exists():
            raise FileNotFoundError(args.image)
        image = build_inference_transform(cfg.data)(Image.open(args.image).convert("RGB"))
        images = image.unsqueeze(0)

    model = create_model(args.model, num_classes=len(GARBAGE_CLASSES), pretrained=False).eval()
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
    top_index = int(probabilities[0].argmax())
    print(f"images: {tuple(images.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"probabilities sum: {probabilities[0].sum().item():.3f}")
    print(f"prediction: {GARBAGE_CLASSES[top_index]} ({probabilities[0, top_index].item():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
