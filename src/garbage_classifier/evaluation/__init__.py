"""Evaluation package: metrics and confusion-matrix reporting."""

from .metrics import (
    classification_report,
    confusion_matrix,
    error_samples,
    evaluate_predictions,
)

__all__ = ["confusion_matrix", "evaluate_predictions", "classification_report", "error_samples"]
