from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXAMPLES = [
    "01_tensor_flow.py",
    "02_dataset_batch.py",
    "03_minimal_training.py",
    "04_predict.py",
    "05_export_onnx.py",
]


def test_beginner_examples_have_helpful_command_lines():
    for name in EXAMPLES:
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / name), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert "usage:" in result.stdout.lower(), name


def test_minimal_training_example_exposes_the_core_loop():
    source = (ROOT / "examples/03_minimal_training.py").read_text()
    for expression in ("model(images)", "loss.backward()", "optimizer.step()"):
        assert expression in source


def test_minimal_training_example_shows_learning_progress():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/03_minimal_training.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    losses = [float(value) for value in re.findall(r"loss=([0-9.]+)", result.stdout)]
    assert result.returncode == 0, result.stderr
    assert len(losses) == 2
    assert losses[-1] < losses[0]
