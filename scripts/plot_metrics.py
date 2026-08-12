#!/usr/bin/env python3
"""Plot training curves from a run's metrics.csv.

Usage:
    python scripts/plot_metrics.py artifacts/<run>/metrics.csv [--out loss_curve.png]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=str, help="path to metrics.csv of a run")
    parser.add_argument("--out", type=str, default=None, help="output PNG (default: next to the csv)")
    args = parser.parse_args()

    csv_path = Path(args.metrics_csv)
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, [float(r["train_loss"]) for r in rows], label="train loss", color="tab:blue")
    ax1.plot(epochs, [float(r["val_loss"]) for r in rows], label="val loss", color="tab:red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [float(r["accuracy"]) for r in rows], label="accuracy", color="tab:green")
    ax2.plot(epochs, [float(r["balanced_acc"]) for r in rows], label="balanced acc", color="tab:orange")
    ax2.plot(epochs, [float(r["macro_f1"]) for r in rows], label="macro F1", color="tab:purple")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.set_title("Metrics")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out) if args.out else csv_path.with_name("loss_curve.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
