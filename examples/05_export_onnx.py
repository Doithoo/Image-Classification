#!/usr/bin/env python3
"""Export a checkpoint for a Python-free ONNX Runtime process.

Usage:
    python examples/05_export_onnx.py --checkpoint artifacts/<run>/best.pt --output model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from garbage_classifier.inference.export import export_onnx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("model.onnx"))
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    output = export_onnx(args.checkpoint, args.output, verify=not args.no_verify)
    print(f"saved: {output}")
    print(f"metadata: {output.with_suffix('.onnx.meta.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
