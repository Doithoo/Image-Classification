#!/usr/bin/env python3
"""Load a checkpoint and predict one image.

Usage:
    python examples/04_predict.py --checkpoint artifacts/<run>/best.pt --image image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from garbage_classifier.inference import Predictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("top-k must be positive")
    predictor = Predictor(args.checkpoint)
    for name, probability in predictor.predict_path(args.image, top_k=args.top_k, tta=args.tta):
        print(f"{name}: {probability:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
