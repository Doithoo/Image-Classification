"""Evaluation package: metrics, evidence publication and run comparison."""

from .comparison import compare_runs, write_comparison
from .evaluate import evaluate_checkpoint
from .metrics import classification_report, confusion_matrix, error_samples, evaluate_predictions

__all__ = [
    "classification_report",
    "compare_runs",
    "confusion_matrix",
    "error_samples",
    "evaluate_checkpoint",
    "evaluate_predictions",
    "write_comparison",
]
