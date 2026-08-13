#!/usr/bin/env python3
"""Create a beginner-friendly dataset preview before training.

Usage:
    python scripts/preview_dataset.py --data-dir data/raw
    python scripts/preview_dataset.py --data-dir data/raw --output artifacts/dataset-preview.png

The output is a contact sheet with one row per class and a ``class_counts.csv``
file next to it. It is intentionally independent of the training pipeline so a
bad folder layout or surprising class distribution can be found early.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_images(data_dir: str | Path) -> dict[str, list[Path]]:
    """Return sorted image paths grouped by class folder."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data dir not found: {root}")
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    images = {
        class_dir.name: sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.startswith("._")
        )
        for class_dir in classes
    }
    if not any(images.values()):
        raise ValueError(f"no supported images found under: {root}")
    return images


def _thumbnail(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def create_preview(
    data_dir: str | Path,
    output_path: str | Path,
    samples_per_class: int = 5,
    tile_size: int = 160,
) -> tuple[Path, Path]:
    """Write a contact sheet and class-count CSV; return both paths."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    if tile_size < 32:
        raise ValueError("tile_size must be at least 32")

    grouped = collect_images(data_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    classes = list(grouped)
    label_width = max(150, max(len(name) for name in classes) * 12 + 24)
    columns = samples_per_class
    row_height = tile_size + 36
    sheet = Image.new("RGB", (label_width + columns * tile_size, len(classes) * row_height), "#f4f4f4")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row, class_name in enumerate(classes):
        y = row * row_height
        draw.text((12, y + 8), f"{class_name}\n({len(grouped[class_name])} images)", fill="black", font=font)
        for column, image_path in enumerate(grouped[class_name][:samples_per_class]):
            x = label_width + column * tile_size
            try:
                sheet.paste(_thumbnail(image_path, tile_size - 8), (x + 4, y + 4))
            except (OSError, ValueError):
                draw.rectangle((x + 4, y + 4, x + tile_size - 4, y + tile_size - 4), outline="#cc3333", width=2)
                draw.text((x + 12, y + tile_size // 2), "unreadable", fill="#cc3333", font=font)

    sheet.save(output)
    counts_path = output.with_name("class_counts.csv")
    with counts_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "count"])
        writer.writerows((name, len(paths)) for name, paths in grouped.items())
    return output, counts_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="folder containing one folder per class")
    parser.add_argument("--output", default="artifacts/dataset-preview.png", help="contact-sheet PNG path")
    parser.add_argument("--samples-per-class", type=int, default=5)
    args = parser.parse_args(argv)
    output, counts = create_preview(args.data_dir, args.output, args.samples_per_class)
    print(f"saved preview: {output}")
    print(f"saved counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
