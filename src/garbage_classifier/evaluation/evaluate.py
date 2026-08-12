"""Evaluate command logic: run a checkpoint against a split and report metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..data import ImageClassificationDataset, collate_fn
from ..data.transforms import build_eval_transform
from ..utils import pick_device
from .metrics import classification_report, error_samples, evaluate_predictions

logger = logging.getLogger("garbage_classifier.evaluate")


def evaluate_checkpoint(
    checkpoint: str | Path,
    cfg: ExperimentConfig,
    split: str = "test",
    tta: bool = False,
    plot: bool = False,
    error_limit: int = 20,
    output_dir: str | Path | None = None,
) -> dict:
    """Evaluate a self-contained checkpoint on a manifest split; prints a report.

    Returns the full metric dict (see ``evaluation.metrics``). When ``plot`` is
    set, a confusion-matrix PNG is saved next to the checkpoint.
    """
    device = pick_device(cfg.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names: list[str] = payload["class_names"]
    # model identity comes from the checkpoint metadata (self-contained); an
    # explicit --config may override data paths but not silently swap the architecture
    ckpt_model = payload.get("config", {}).get("model", {}).get("name", cfg.model.name)
    logger.info("evaluating %s (%s) on %s (classes=%d)", checkpoint, ckpt_model, split, len(class_names))

    dataset = ImageClassificationDataset(
        Path(cfg.data.manifest_dir) / f"{split}.csv", transform=build_eval_transform(cfg.data)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
    )

    from ..inference.predictor import Predictor  # lazy: avoids import cycle

    predictor = Predictor(checkpoint, device=cfg.device)
    predictor.model.load_state_dict(payload["model_state_dict"])

    all_preds, all_labels, all_paths = [], [], []
    sample_paths = [p for p, _ in dataset.samples]
    start = 0
    with torch.no_grad():
        for images, labels in loader:
            probs = predictor.predict_probs(images.to(device), tta=tta)
            preds_batch = probs.argmax(dim=1).cpu()
            all_preds.append(preds_batch)
            all_labels.append(labels)
            end = start + len(labels)
            all_paths.extend(sample_paths[start:end])
            start = end

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = evaluate_predictions(preds, labels, num_classes=len(class_names))

    print(classification_report(metrics, class_names))
    print(f"\nconfusion matrix (rows=true, cols=pred):\n{metrics['confusion']}")

    # prediction CSV + error sample list
    out_dir = Path(output_dir or Path(checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w") as f:
        f.write("path,true,pred\n")
        for p, t, pr in zip(all_paths, labels, preds, strict=True):
            f.write(f"{p},{class_names[t]},{class_names[pr]}\n")
    errors = error_samples(labels, preds, all_paths, limit=error_limit)
    err_csv = out_dir / "errors.csv"
    with err_csv.open("w") as f:
        f.write("path,true,pred\n")
        for e in errors:
            f.write(f"{e['path']},{class_names[e['true']]},{class_names[e['pred']]}\n")
    logger.info("wrote %s and %s", pred_csv, err_csv)

    if plot:
        _save_confusion_plot(metrics["confusion"], class_names, out_dir / "confusion_matrix.png")
    return metrics


def _save_confusion_plot(confusion: list, class_names: list[str], out_path: Path) -> None:
    """Render and save the confusion matrix heatmap (headless-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(confusion, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (acc={cm.trace() / cm.sum():.3f})")
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"confusion matrix plot saved to {out_path}")
