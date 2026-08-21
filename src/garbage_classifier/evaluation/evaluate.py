"""Checkpoint evaluation with verified data and atomically published evidence."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data import ImageClassificationDataset, collate_fn, verify_prepared_data
from ..data.transforms import build_eval_transform
from ..training.checkpoint import CheckpointCompatibilityError, load_checkpoint, restore_config_from_checkpoint
from ..utils import file_sha256, pick_device, publish_directory, write_text_atomic
from .metrics import classification_report, error_samples, evaluate_predictions

logger = logging.getLogger("garbage_classifier.evaluate")
EVALUATION_SCHEMA_VERSION = 2


def _default_output_dir(checkpoint: Path, split: str, tta: bool) -> Path:
    suffix = f"{split}-tta" if tta else split
    return checkpoint.parent / "evaluation" / suffix


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, output.getvalue())


def evaluate_checkpoint(
    checkpoint: str | Path,
    cfg: ExperimentConfig,
    split: str = "test",
    tta: bool = False,
    plot: bool = False,
    error_limit: int = 20,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint only against data matching its recorded identity."""
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"unknown split: {split}")
    if error_limit < 0:
        raise ValueError("error_limit must be non-negative")
    checkpoint_path = Path(checkpoint)
    payload = load_checkpoint(checkpoint_path)
    checkpoint_cfg = restore_config_from_checkpoint(payload)
    metadata = verify_prepared_data(cfg.data.manifest_dir, cfg.data.data_dir)
    if not payload.get("legacy_checkpoint") and payload["manifest_identity"] != metadata.identity:
        raise CheckpointCompatibilityError("checkpoint manifest_identity does not match the prepared dataset")
    class_names = list(payload["class_names"])
    if tuple(class_names) != metadata.classes:
        raise CheckpointCompatibilityError("checkpoint class_names do not match the prepared dataset")
    device = pick_device(cfg.device)
    dataset = ImageClassificationDataset(
        cfg.data.manifest_dir / f"{split}.csv",
        root_dir=cfg.data.data_dir,
        transform=build_eval_transform(checkpoint_cfg.data),
    )
    if not len(dataset):
        raise ValueError(f"evaluation split is empty: {split}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
    )
    from ..inference.predictor import Predictor

    predictor = Predictor(checkpoint_path, device=cfg.device, config_path=config_path)
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, target in loader:
            batch_probabilities = predictor.predict_probs(images.to(device), tta=tta).cpu()
            probabilities.append(batch_probabilities)
            predictions.append(batch_probabilities.argmax(dim=1))
            labels.append(target.cpu())
    probability_array = torch.cat(probabilities).numpy()
    prediction_array = torch.cat(predictions).numpy()
    label_array = torch.cat(labels).numpy()
    paths = [path for path, _label in dataset.samples]
    metrics = evaluate_predictions(prediction_array, label_array, len(class_names), probability_array)
    print(classification_report(metrics, class_names))
    print(f"\nconfusion matrix (rows=true, cols=pred):\n{metrics['confusion']}")

    destination = Path(output_dir) if output_dir is not None else _default_output_dir(checkpoint_path, split, tta)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        ranked_indices = np.argsort(-probability_array, axis=1)
        prediction_rows = []
        for index, (path, true, pred) in enumerate(zip(paths, label_array, prediction_array, strict=True)):
            top = [
                {
                    "class_name": class_names[class_index],
                    "probability": round(float(probability_array[index, class_index]), 8),
                }
                for class_index in ranked_indices[index, : min(5, len(class_names))]
            ]
            prediction_rows.append(
                {
                    "path": path,
                    "true": class_names[int(true)],
                    "true_index": int(true),
                    "pred": class_names[int(pred)],
                    "pred_index": int(pred),
                    "confidence": round(float(probability_array[index, pred]), 8),
                    "true_probability": round(float(probability_array[index, true]), 8),
                    "entropy": round(
                        float(-(probability_array[index] * np.log(np.clip(probability_array[index], 1e-12, 1))).sum()),
                        8,
                    ),
                    "top_k": json.dumps(top, ensure_ascii=False),
                }
            )
        _write_csv(
            stage / "predictions.csv",
            ["path", "true", "true_index", "pred", "pred_index", "confidence", "true_probability", "entropy", "top_k"],
            prediction_rows,
        )
        errors = error_samples(label_array, prediction_array, paths, probability_array, error_limit)
        error_rows = [
            {
                "path": error["path"],
                "true": class_names[error["true"]],
                "pred": class_names[error["pred"]],
                "confidence": round(float(error.get("confidence", 0.0)), 8),
                "true_probability": round(float(error.get("true_probability", 0.0)), 8),
            }
            for error in errors
        ]
        _write_csv(stage / "errors.csv", ["path", "true", "pred", "confidence", "true_probability"], error_rows)
        per_class_rows = [
            {
                "class_index": index,
                "class_name": name,
                "precision": metrics["per_class_precision"][index],
                "recall": metrics["per_class_recall"][index],
                "f1": metrics["per_class_f1"][index],
                "support": metrics["per_class_support"][index],
            }
            for index, name in enumerate(class_names)
        ]
        _write_csv(
            stage / "per_class.csv",
            ["class_index", "class_name", "precision", "recall", "f1", "support"],
            per_class_rows,
        )
        _write_csv(
            stage / "calibration.csv",
            ["lower", "upper", "count", "accuracy", "confidence"],
            metrics.get("calibration_bins", []),
        )
        report = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "split": split,
            "tta": tta,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "manifest_identity": metadata.identity,
            "dataset_schema_version": metadata.schema_version,
            "class_names": class_names,
            "preprocessing": payload.get("preprocessing"),
            "metrics": metrics,
        }
        write_text_atomic(stage / "evaluation.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if plot:
            _save_confusion_plot(metrics["confusion"], class_names, stage / "confusion_matrix.png")
            _save_reliability_plot(metrics.get("calibration_bins", []), stage / "reliability.png")
        publish_directory(stage, destination, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    logger.info("wrote evaluation evidence to %s", destination)
    return metrics


def _save_confusion_plot(confusion: list[list[int]], class_names: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(confusion, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    total = matrix.sum()
    axis.set_title(f"Confusion matrix (acc={matrix.trace() / total if total else 0.0:.3f})")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def _save_reliability_plot(bins: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    populated = [item for item in bins if item["count"]]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    if populated:
        axis.plot(
            [item["confidence"] for item in populated],
            [item["accuracy"] for item in populated],
            marker="o",
            label="model",
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
