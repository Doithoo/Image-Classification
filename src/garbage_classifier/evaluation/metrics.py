"""Evaluation metrics — what each number means (learning note).

Given N predictions, we build a confusion matrix ``M`` with
``M[true, pred] += 1`` (rows = true labels, columns = predictions). From it:

  - **accuracy** = trace(M) / sum(M) — fraction correct. Cheap but misleading
    when classes are imbalanced: guessing the majority class already scores well.
  - **balanced accuracy** = mean over classes of recall — every class counts
    equally, so a model that ignores the rare class is exposed.
  - **precision_c** = M[c,c] / (predictions of class c) — "when I say class c,
    how often am I right?"
  - **recall_c**    = M[c,c] / (true class c) — "of the real class-c samples,
    how many did I find?"
  - **F1_c** = harmonic mean of precision and recall — single number for the
    precision/recall trade-off.
  - **macro F1** = mean of per-class F1 (classes equal weight);
    **weighted F1** = F1 weighted by class support (follows the data
    distribution).

This module follows the standard ``matrix[true, pred]`` convention, which makes
the arithmetic above direct and the plotted heatmap intuitive.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Build a [num_classes, num_classes] matrix with rows=true, cols=predicted."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, preds), 1)
    return matrix


def evaluate_predictions(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> dict[str, float]:
    """Compute the full metric set from raw predictions.

    Returns accuracy, balanced accuracy, macro/weighted F1, macro precision/recall
    and per-class precision/recall/F1/support.
    """
    preds = np.asarray(preds).astype(np.int64)
    labels = np.asarray(labels).astype(np.int64)
    matrix = confusion_matrix(labels, preds, num_classes)

    tp = np.diag(matrix).astype(np.float64)  # correct predictions per class
    fp = matrix.sum(axis=0) - tp  # predicted as c, actually other classes
    fn = matrix.sum(axis=1) - tp  # actually c, predicted as other classes
    support = matrix.sum(axis=1).astype(np.float64)  # true count per class

    eps = 1e-12
    precision = tp / np.maximum(tp + fp, eps)
    recall = tp / np.maximum(tp + fn, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)

    total = matrix.sum()
    accuracy = tp.sum() / total if total else 0.0
    balanced_accuracy = float(recall.mean())

    macro_f1 = float(f1.mean())
    weighted_f1 = float((f1 * support).sum() / max(support.sum(), eps))
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    weighted_precision = float((precision * support).sum() / max(support.sum(), eps))
    weighted_recall = float((recall * support).sum() / max(support.sum(), eps))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.astype(np.int64).tolist(),
        "confusion": matrix.tolist(),
    }


def classification_report(metrics: dict[str, float], class_names: list[str]) -> str:
    """Human-readable report mirroring sklearn's format."""
    lines = [f"{'':12s} {'precision':>10s} {'recall':>8s} {'f1-score':>9s} {'support':>8s}"]
    prec, rec, f1, sup = (
        metrics["per_class_precision"],
        metrics["per_class_recall"],
        metrics["per_class_f1"],
        metrics["per_class_support"],
    )
    for i, name in enumerate(class_names):
        lines.append(f"{name:12s} {prec[i]:10.3f} {rec[i]:8.3f} {f1[i]:9.3f} {sup[i]:8d}")
    lines.append("-" * 50)
    lines.append(f"{'accuracy':12s} {'':10s} {'':8s} {metrics['accuracy']:9.3f} {sum(sup):8d}")
    lines.append(
        f"{'macro avg':12s} {metrics['macro_precision']:10.3f} {metrics['macro_recall']:8.3f} {metrics['macro_f1']:9.3f}"
    )
    lines.append(
        f"{'weighted avg':12s} {metrics['weighted_precision']:10.3f} {metrics['weighted_recall']:8.3f} {metrics['weighted_f1']:9.3f}"
    )
    lines.append(f"balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    return "\n".join(lines)


def error_samples(labels: np.ndarray, preds: np.ndarray, paths: list[str], limit: int = 20) -> list[dict[str, str]]:
    """Return the worst misclassified samples: (path, true, pred, prob) if prob given."""
    errors = [i for i in range(len(labels)) if labels[i] != preds[i]]
    # keep relative order; caller can sort by probability
    return [{"path": paths[i], "true": int(labels[i]), "pred": int(preds[i])} for i in errors[:limit]]
