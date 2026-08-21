#!/usr/bin/env python3
"""A deliberately small training loop for reading before Trainer.

Usage:
    python examples/03_minimal_training.py
    python examples/03_minimal_training.py --epochs 3
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    torch.manual_seed(7)
    # Make two visually distinct synthetic classes. This keeps the example
    # downloadable-data free and makes the default loss trend easy to observe.
    labels = torch.arange(32) % 2
    images = labels[:, None, None, None].float().expand(-1, 3, 8, 8).clone()
    images += torch.rand_like(images) * 0.1
    loader = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=True)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(args.epochs):
        total_loss = 0.0
        sample_count = 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            sample_count += labels.numel()
        print(f"epoch {epoch + 1}: loss={total_loss / sample_count:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
