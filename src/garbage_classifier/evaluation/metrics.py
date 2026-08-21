"""Classification metrics, calibration diagnostics and confidence-ranked errors."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Build a matrix with rows=true labels and columns=predicted labels."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, preds), 1)
    return matrix


def _calibration(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict[str, Any]:
    confidences = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    records: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (confidences >= lower) & ((confidences < upper) if index < bins - 1 else (confidences <= upper))
        count = int(mask.sum())
        if count:
            accuracy = float(correct[mask].mean())
            confidence = float(confidences[mask].mean())
            ece += abs(accuracy - confidence) * count / len(labels)
        else:
            accuracy = 0.0
            confidence = 0.0
        records.append({"lower": lower, "upper": upper, "count": count, "accuracy": accuracy, "confidence": confidence})
    return {"ece": float(ece), "calibration_bins": records}


def evaluate_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute class metrics and optional probabilistic diagnostics."""
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if preds.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    if np.any(preds < 0) or np.any(preds >= num_classes) or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("predictions and labels must be in [0, num_classes)")
    matrix = confusion_matrix(labels, preds, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    support = matrix.sum(axis=1).astype(np.float64)
    epsilon = 1e-12
    precision = true_positive / np.maximum(true_positive + false_positive, epsilon)
    recall = true_positive / np.maximum(true_positive + false_negative, epsilon)
    f1 = 2 * precision * recall / np.maximum(precision + recall, epsilon)
    total = matrix.sum()
    accuracy = float(true_positive.sum() / total) if total else 0.0
    supported = support > 0
    balanced_accuracy = float(recall[supported].mean()) if np.any(supported) else 0.0
    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1[supported].mean()) if np.any(supported) else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), epsilon)),
        "macro_precision": float(precision[supported].mean()) if np.any(supported) else 0.0,
        "macro_recall": float(recall[supported].mean()) if np.any(supported) else 0.0,
        "weighted_precision": float((precision * support).sum() / max(support.sum(), epsilon)),
        "weighted_recall": float((recall * support).sum() / max(support.sum(), epsilon)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.astype(np.int64).tolist(),
        "confusion": matrix.tolist(),
    }
    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if probability_array.shape != (len(labels), num_classes):
            raise ValueError("probabilities must have shape [samples, num_classes]")
        clipped = np.clip(probability_array, epsilon, 1.0)
        metrics["nll"] = float(-np.log(clipped[np.arange(len(labels)), labels]).mean()) if len(labels) else 0.0
        one_hot = np.eye(num_classes)[labels]
        metrics["brier_score"] = (
            float(np.square(probability_array - one_hot).sum(axis=1).mean()) if len(labels) else 0.0
        )
        top_k = min(5, num_classes)
        top_indices = np.argpartition(probability_array, -top_k, axis=1)[:, -top_k:]
        metrics["top_5_accuracy"] = (
            float(np.mean([label in indices for label, indices in zip(labels, top_indices, strict=True)]))
            if len(labels)
            else 0.0
        )
        metrics.update(_calibration(probability_array, labels))
    return metrics


def classification_report(metrics: dict[str, Any], class_names: list[str]) -> str:
    """Render a compact report without hiding classes that have zero support."""
    lines = [f"{'':12s} {'precision':>10s} {'recall':>8s} {'f1-score':>9s} {'support':>8s}"]
    for index, name in enumerate(class_names):
        lines.append(
            f"{name:12s} {metrics['per_class_precision'][index]:10.3f} "
            f"{metrics['per_class_recall'][index]:8.3f} {metrics['per_class_f1'][index]:9.3f} "
            f"{metrics['per_class_support'][index]:8d}"
        )
    support = metrics["per_class_support"]
    lines.extend(
        [
            "-" * 50,
            f"{'accuracy':12s} {'':10s} {'':8s} {metrics['accuracy']:9.3f} {sum(support):8d}",
            f"{'macro avg':12s} {metrics['macro_precision']:10.3f} {metrics['macro_recall']:8.3f} {metrics['macro_f1']:9.3f}",
            f"{'weighted avg':12s} {metrics['weighted_precision']:10.3f} {metrics['weighted_recall']:8.3f} {metrics['weighted_f1']:9.3f}",
            f"balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        ]
    )
    return "\n".join(lines)


def error_samples(
    labels: np.ndarray,
    preds: np.ndarray,
    paths: list[str],
    probabilities: np.ndarray | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return high-confidence errors first, including diagnostic probabilities when known."""
    if limit < 0:
        raise ValueError("error limit must be non-negative")
    errors: list[dict[str, Any]] = []
    for index, (label, pred, path) in enumerate(zip(labels, preds, paths, strict=True)):
        if label == pred:
            continue
        record: dict[str, Any] = {"path": path, "true": int(label), "pred": int(pred)}
        if probabilities is not None:
            record["confidence"] = float(probabilities[index, pred])
            record["true_probability"] = float(probabilities[index, label])
        errors.append(record)
    errors.sort(key=lambda record: float(record.get("confidence", 0.0)), reverse=True)
    return errors[:limit]
