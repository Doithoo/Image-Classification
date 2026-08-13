"""Tests for evaluation metrics (incl. confusion-matrix orientation)."""

import numpy as np

from garbage_classifier.evaluation import classification_report, evaluate_predictions


def test_confusion_matrix_orientation():
    # true=[a,a,b,b], pred=[a,b,a,b] -> matrix[true, pred]
    from garbage_classifier.evaluation.metrics import confusion_matrix

    m = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), num_classes=2)
    assert m.tolist() == [[1, 1], [1, 1]]


def test_perfect_predictions():
    labels = np.array([0, 0, 1, 1, 2])
    preds = labels.copy()
    m = evaluate_predictions(preds, labels, num_classes=3)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0


def test_per_class_metrics_hand_computed():
    # 3 classes, hand-computed example
    labels = np.array([0, 0, 0, 1, 1, 2])
    preds = np.array([0, 0, 1, 1, 1, 2])
    m = evaluate_predictions(preds, labels, num_classes=3)
    assert m["accuracy"] == 5 / 6
    # matrix (rows=true, cols=pred):
    #   class0: TP=2 FN=1 FP=0      -> P=1 R=2/3 F1=0.8
    #   class1: TP=2 FN=0 FP=1      -> P=2/3 R=1 F1=0.8
    #   class2: TP=1                -> P=1 R=1 F1=1
    assert np.isclose(m["per_class_precision"][0], 1.0)
    assert np.isclose(m["per_class_recall"][0], 2 / 3)
    assert np.isclose(m["per_class_precision"][1], 2 / 3)
    assert np.isclose(m["per_class_f1"][0], 0.8)
    assert np.isclose(m["macro_f1"], (0.8 + 0.8 + 1.0) / 3)
    assert np.isclose(m["weighted_precision"], (1.0 * 3 + (2 / 3) * 2 + 1.0) / 6)
    assert np.isclose(m["weighted_recall"], ((2 / 3) * 3 + 1.0 * 2 + 1.0) / 6)
    # balanced accuracy = mean recall = (2/3 + 1 + 1)/3
    assert np.isclose(m["balanced_accuracy"], (2 / 3 + 1 + 1) / 3)


def test_all_wrong():
    labels = np.array([0, 1])
    preds = np.array([1, 0])
    m = evaluate_predictions(preds, labels, num_classes=2)
    assert m["accuracy"] == 0.0
    assert m["balanced_accuracy"] == 0.0


def test_classification_report_output():
    labels = np.array([0, 0, 1])
    preds = np.array([0, 0, 1])
    m = evaluate_predictions(preds, labels, num_classes=2)
    report = classification_report(m, ["a", "b"])
    assert "a" in report and "b" in report
    assert "balanced accuracy" in report


def test_classification_report_uses_weighted_precision_and_recall():
    labels = np.array([0, 0, 0, 1])
    preds = np.array([0, 0, 1, 1])
    metrics = evaluate_predictions(preds, labels, num_classes=2)

    weighted_row = next(
        line for line in classification_report(metrics, ["a", "b"]).splitlines() if "weighted avg" in line
    )

    assert f"{metrics['weighted_precision']:.3f}" in weighted_row
    assert f"{metrics['weighted_recall']:.3f}" in weighted_row
