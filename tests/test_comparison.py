"""Run comparison is constrained by prepared-data identity."""

import csv

import pytest
import yaml

from garbage_classifier.evaluation.comparison import compare_runs, write_comparison


def _run(tmp_path, name, identity, metrics):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "run.yaml").write_text(yaml.safe_dump({"manifest_identity": identity, "device": "cpu"}))
    with (directory / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "accuracy", "balanced_acc", "macro_f1"])
        writer.writeheader()
        writer.writerows(metrics)
    return directory


def test_compare_runs_ranks_validation_metrics_and_writes_new_csv(tmp_path):
    first = _run(tmp_path, "first", "shared", [{"epoch": 1, "accuracy": 0.7, "balanced_acc": 0.6, "macro_f1": 0.5}])
    second = _run(tmp_path, "second", "shared", [{"epoch": 2, "accuracy": 0.6, "balanced_acc": 0.8, "macro_f1": 0.7}])

    rows = compare_runs([first, second], metric="macro_f1")
    output = write_comparison(tmp_path / "comparison.csv", rows)

    assert [row.run_name for row in rows] == ["second", "first"]
    assert "manifest_identity" in output.read_text()
    with pytest.raises(FileExistsError):
        write_comparison(output, rows)


def test_compare_runs_rejects_different_dataset_identity(tmp_path):
    first = _run(tmp_path, "first", "one", [{"epoch": 1, "accuracy": 0.7, "balanced_acc": 0.6, "macro_f1": 0.5}])
    second = _run(tmp_path, "second", "two", [{"epoch": 1, "accuracy": 0.6, "balanced_acc": 0.5, "macro_f1": 0.4}])

    with pytest.raises(ValueError, match="different manifest_identity"):
        compare_runs([first, second])
